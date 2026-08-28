// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Async Unix domain socket I/O for vhost-user, carrying the operating-system
//! objects the protocol passes alongside its messages.
//!
//! The control channel is an `AF_UNIX` stream on both platforms, and the
//! messages on it are identical. What differs is how a section or an event
//! reaches the peer:
//!
//! * On POSIX the kernel does it: objects ride with the first `sendmsg` as
//!   `SCM_RIGHTS` ancillary data, installed into the receiver atomically with
//!   the message.
//! * On Windows there is no ancillary data, so the frontend duplicates each
//!   object into the backend process and appends the resulting handle values
//!   to the payload. See [`crate::win32`] for that binding, including why an
//!   undelivered message has to take its handles back.

use crate::protocol::VHOST_USER_MAX_FDS;
use crate::protocol::VhostUserMsgHeader;
use pal_async::interest::InterestSlot;
use pal_async::interest::PollEvents;
use pal_async::socket::PolledSocket;
use sparse_mmap::AsMappableRef;
use sparse_mmap::Mappable;
use std::future::poll_fn;
use std::io;
use thiserror::Error;
use unix_socket::UnixStream;
use zerocopy::FromBytes;
use zerocopy::IntoBytes;

#[cfg(unix)]
use std::io::IoSlice;
#[cfg(unix)]
use unix_socket::ScmReceiver;

#[cfg(windows)]
use crate::win32::RemoteHandles;
#[cfg(windows)]
use std::os::windows::io::OwnedHandle;

#[derive(Debug, Error)]
pub enum SocketError {
    #[error("i/o error")]
    Io(#[source] io::Error),
    #[error("connection closed")]
    Closed,
    #[error("payload too large: {0} bytes")]
    PayloadTooLarge(u32),
    /// The message carries an object, but no handle to the backend process was
    /// configured, so there is no process to duplicate it into.
    #[cfg(windows)]
    #[error(
        "cannot pass {0} object(s) to the backend: no backend process handle was configured for \
         this connection"
    )]
    NoPeerProcess(usize),
    /// Duplicating an object into the backend process failed.
    #[cfg(windows)]
    #[error("failed to duplicate an object into the backend process")]
    Duplicate(#[source] io::Error),
}

impl From<io::Error> for SocketError {
    fn from(e: io::Error) -> Self {
        SocketError::Io(e)
    }
}

/// Maximum payload size to accept (4 MB, generous upper bound).
const MAX_PAYLOAD_SIZE: u32 = 4 * 1024 * 1024;

/// Per-connection state for receiving messages.
///
/// On POSIX this owns the control buffer used for `SCM_RIGHTS` fd passing,
/// reused across the connection to avoid a per-message allocation. On Windows
/// nothing is needed: objects sent to this side would arrive in the payload,
/// and no message that carries one is part of a negotiated feature here.
pub struct MessageReceiver {
    #[cfg(unix)]
    scm: ScmReceiver,
}

impl MessageReceiver {
    pub fn new() -> Self {
        Self {
            #[cfg(unix)]
            scm: ScmReceiver::new(VHOST_USER_MAX_FDS),
        }
    }
}

impl Default for MessageReceiver {
    fn default() -> Self {
        Self::new()
    }
}

/// Async vhost-user socket for sending and receiving protocol messages.
pub struct VhostUserSocket {
    socket: parking_lot::Mutex<PolledSocket<UnixStream>>,
    /// The process to duplicate objects into. Required to send a message that
    /// carries one; see [`Self::with_peer_process`].
    #[cfg(windows)]
    peer_process: Option<OwnedHandle>,
}

impl VhostUserSocket {
    /// Wrap a connected `UnixStream` in an async vhost-user socket.
    ///
    /// On Windows this connection cannot pass objects to the peer. That is
    /// correct for a backend, which never sends one, and for a frontend that
    /// only exchanges plain messages; a frontend that has to send guest memory
    /// or doorbells needs [`Self::with_peer_process`].
    pub fn new(socket: PolledSocket<UnixStream>) -> Self {
        Self {
            socket: parking_lot::Mutex::new(socket),
            #[cfg(windows)]
            peer_process: None,
        }
    }

    /// Wrap a connected `UnixStream`, naming the backend process that objects
    /// are duplicated into.
    ///
    /// `peer_process` must carry `PROCESS_DUP_HANDLE`. It cannot be derived
    /// from the socket — Windows `AF_UNIX` reports no peer credentials, and a
    /// process ID would be racy because IDs are reused — so it comes from
    /// whoever launched the backend.
    #[cfg(windows)]
    pub fn with_peer_process(socket: PolledSocket<UnixStream>, peer_process: OwnedHandle) -> Self {
        Self {
            socket: parking_lot::Mutex::new(socket),
            peer_process: Some(peer_process),
        }
    }

    /// Receive a vhost-user message (header + payload + any objects).
    ///
    /// The caller provides a [`MessageReceiver`], typically reused across the
    /// connection. Returns the parsed header, payload bytes, and any objects
    /// the peer passed.
    pub async fn recv_message(
        &self,
        receiver: &mut MessageReceiver,
    ) -> Result<(VhostUserMsgHeader, Vec<u8>, Vec<Mappable>), SocketError> {
        let mut hdr_buf = [0u8; size_of::<VhostUserMsgHeader>()];
        let mut objects = Vec::new();
        let n = self
            .recv_exact(receiver, &mut hdr_buf, &mut objects)
            .await?;
        if n == 0 {
            return Err(SocketError::Closed);
        }

        let hdr = VhostUserMsgHeader::read_from_bytes(&hdr_buf)
            .expect("hdr_buf is exactly the right size");

        // Read payload if any.
        let payload = if hdr.size > 0 {
            if hdr.size > MAX_PAYLOAD_SIZE {
                return Err(SocketError::PayloadTooLarge(hdr.size));
            }
            let mut payload = vec![0u8; hdr.size as usize];
            self.recv_exact_no_objects(receiver, &mut payload).await?;
            payload
        } else {
            Vec::new()
        };

        Ok((hdr, payload, objects))
    }

    /// Send a vhost-user message (header + payload + any objects).
    ///
    /// On Windows each object is duplicated into the backend process and the
    /// resulting handle values are appended to the payload, with the header's
    /// `size` extended to cover them. If the message is not written in full,
    /// those duplicates are closed again in the backend, so that a failed send
    /// leaves nothing behind — the same all-or-nothing property `SCM_RIGHTS`
    /// gets from the kernel.
    pub async fn send_message(
        &self,
        header: &VhostUserMsgHeader,
        payload: &[u8],
        objects: &[impl AsMappableRef],
    ) -> Result<(), SocketError> {
        assert!(
            objects.len() <= VHOST_USER_MAX_FDS,
            "too many objects: {} > {}",
            objects.len(),
            VHOST_USER_MAX_FDS
        );
        self.send_message_inner(header, payload, objects).await
    }
}

#[cfg(unix)]
impl VhostUserSocket {
    async fn send_message_inner(
        &self,
        header: &VhostUserMsgHeader,
        payload: &[u8],
        objects: &[impl AsMappableRef],
    ) -> Result<(), SocketError> {
        let hdr_bytes = header.as_bytes();
        let iov = [IoSlice::new(hdr_bytes), IoSlice::new(payload)];

        let mut sent = 0;
        let total: usize = iov.iter().map(|s| s.len()).sum();

        // Send all data. Objects are only attached to the first sendmsg, since
        // SCM_RIGHTS delivers them alongside the first byte of the message.
        while sent < total {
            let remaining_iov = build_remaining_iov(&iov, sent);
            let attach = sent == 0;

            let n = poll_fn(|cx| {
                self.socket
                    .lock()
                    .poll_io(cx, InterestSlot::Write, PollEvents::OUT, |socket| {
                        if attach {
                            unix_socket::send_with_fds(
                                socket.get().as_fd(),
                                &remaining_iov,
                                objects.iter().map(|o| o.as_fd()),
                            )
                        } else {
                            unix_socket::send_with_fds(socket.get().as_fd(), &remaining_iov, [])
                        }
                    })
            })
            .await?;
            sent += n;
        }
        Ok(())
    }

    /// Receive exactly `buf.len()` bytes, collecting any objects from the first
    /// recvmsg.
    async fn recv_exact(
        &self,
        receiver: &mut MessageReceiver,
        buf: &mut [u8],
        objects: &mut Vec<Mappable>,
    ) -> Result<usize, SocketError> {
        let mut read = 0;
        while read < buf.len() {
            let n = self
                .recv_raw(
                    receiver,
                    &mut buf[read..],
                    if read == 0 { Some(objects) } else { None },
                )
                .await?;
            if n == 0 {
                if read == 0 {
                    return Ok(0);
                }
                return Err(SocketError::Closed);
            }
            read += n;
        }
        Ok(read)
    }

    /// Receive exactly `buf.len()` bytes, ignoring any ancillary data.
    async fn recv_exact_no_objects(
        &self,
        receiver: &mut MessageReceiver,
        buf: &mut [u8],
    ) -> Result<(), SocketError> {
        let mut read = 0;
        while read < buf.len() {
            let n = self.recv_raw(receiver, &mut buf[read..], None).await?;
            if n == 0 {
                return Err(SocketError::Closed);
            }
            read += n;
        }
        Ok(())
    }

    /// Low-level async recv with optional object collection.
    ///
    /// Waits until the socket is readable, then performs the recv. On spurious
    /// readiness (WouldBlock), re-polls automatically.
    ///
    /// The receiver is drained (into `objects`) or cleared before returning, so
    /// it is always empty on exit and therefore empty on the next entry.
    async fn recv_raw(
        &self,
        receiver: &mut MessageReceiver,
        buf: &mut [u8],
        objects: Option<&mut Vec<Mappable>>,
    ) -> Result<usize, SocketError> {
        let result = poll_fn(|cx| {
            self.socket
                .lock()
                .poll_io(cx, InterestSlot::Read, PollEvents::IN, |socket| {
                    receiver.scm.recv(socket.get().as_fd(), buf)
                })
        })
        .await;

        // Hand received objects to the caller, or drop them (closing any stray
        // descriptors a peer sent unexpectedly). Either way the receiver ends
        // up empty, ready for the next call.
        match objects {
            Some(objects) => objects.extend(receiver.scm.drain()),
            None => receiver.scm.clear(),
        }

        Ok(result?)
    }
}

#[cfg(windows)]
impl VhostUserSocket {
    async fn send_message_inner(
        &self,
        header: &VhostUserMsgHeader,
        payload: &[u8],
        objects: &[impl AsMappableRef],
    ) -> Result<(), SocketError> {
        if objects.is_empty() {
            let mut buf = Vec::with_capacity(size_of::<VhostUserMsgHeader>() + payload.len());
            buf.extend_from_slice(header.as_bytes());
            buf.extend_from_slice(payload);
            return self.send_all(&buf).await;
        }

        let peer = self
            .peer_process
            .as_ref()
            .ok_or(SocketError::NoPeerProcess(objects.len()))?;

        // Duplicate first, then frame. `handles` closes whatever it holds in
        // the backend if anything below fails, so a message that is not
        // delivered in full leaves no handle stranded there.
        let mut handles = RemoteHandles::new(peer.as_handle());
        for object in objects {
            handles
                .duplicate(object.as_handle())
                .map_err(SocketError::Duplicate)?;
        }
        let trailer = handles.trailer();

        // The trailer is part of the payload as far as framing is concerned:
        // the header's size covers it, so the peer reads message and handles
        // together.
        let mut hdr = *header;
        hdr.size = header
            .size
            .checked_add(trailer.len() as u32)
            .expect("message size cannot overflow: payload and trailer are both bounded");

        let mut buf =
            Vec::with_capacity(size_of::<VhostUserMsgHeader>() + payload.len() + trailer.len());
        buf.extend_from_slice(hdr.as_bytes());
        buf.extend_from_slice(payload);
        buf.extend_from_slice(&trailer);

        self.send_all(&buf).await?;

        // Delivered: the backend owns these handles now and closes them itself.
        handles.commit();
        Ok(())
    }

    /// Write `buf` in full, waiting for writability as needed.
    async fn send_all(&self, buf: &[u8]) -> Result<(), SocketError> {
        let mut sent = 0;
        while sent < buf.len() {
            let n = poll_fn(|cx| {
                self.socket
                    .lock()
                    .poll_io(cx, InterestSlot::Write, PollEvents::OUT, |socket| {
                        io::Write::write(&mut socket.get(), &buf[sent..])
                    })
            })
            .await?;
            if n == 0 {
                return Err(SocketError::Closed);
            }
            sent += n;
        }
        Ok(())
    }

    /// Receive exactly `buf.len()` bytes.
    ///
    /// `objects` is never extended: a message carrying an object to this side
    /// belongs to a feature that is not negotiated on Windows, so none arrives.
    async fn recv_exact(
        &self,
        receiver: &mut MessageReceiver,
        buf: &mut [u8],
        _objects: &mut Vec<Mappable>,
    ) -> Result<usize, SocketError> {
        let mut read = 0;
        while read < buf.len() {
            let n = self.recv_raw(receiver, &mut buf[read..]).await?;
            if n == 0 {
                if read == 0 {
                    return Ok(0);
                }
                return Err(SocketError::Closed);
            }
            read += n;
        }
        Ok(read)
    }

    async fn recv_exact_no_objects(
        &self,
        receiver: &mut MessageReceiver,
        buf: &mut [u8],
    ) -> Result<(), SocketError> {
        let mut read = 0;
        while read < buf.len() {
            let n = self.recv_raw(receiver, &mut buf[read..]).await?;
            if n == 0 {
                return Err(SocketError::Closed);
            }
            read += n;
        }
        Ok(())
    }

    /// Low-level async recv. Waits until the socket is readable, then reads.
    async fn recv_raw(
        &self,
        _receiver: &mut MessageReceiver,
        buf: &mut [u8],
    ) -> Result<usize, SocketError> {
        let n = poll_fn(|cx| {
            self.socket
                .lock()
                .poll_io(cx, InterestSlot::Read, PollEvents::IN, |socket| {
                    io::Read::read(&mut socket.get(), buf)
                })
        })
        .await?;
        Ok(n)
    }
}

/// Build IoSlice entries for the remaining unsent bytes.
#[cfg(unix)]
fn build_remaining_iov<'a>(original: &'a [IoSlice<'a>], skip: usize) -> Vec<IoSlice<'a>> {
    let mut remaining = skip;
    let mut result = Vec::new();
    for slice in original {
        if remaining >= slice.len() {
            remaining -= slice.len();
        } else {
            result.push(IoSlice::new(&slice[remaining..]));
            remaining = 0;
        }
    }
    result
}
