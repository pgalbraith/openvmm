// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Windows replacement for `SCM_RIGHTS` object passing.
//!
//! POSIX hands guest-memory sections and vring doorbells to the backend as
//! `SCM_RIGHTS` ancillary data, which the kernel installs into the receiver
//! atomically with the message. Windows `AF_UNIX` sockets carry no ancillary
//! data at all, so the objects travel inside the message stream instead:
//! before sending, the frontend duplicates each one into the backend process
//! and puts the resulting handle value in the message.
//!
//! Wherever POSIX attaches K descriptors, this appends K records of
//! [`HANDLE_RECORD_SIZE`] bytes to the payload, in the order the descriptors
//! would have appeared. The header's `size` covers the trailer, so a message
//! and its handles arrive together with no extra framing. Nothing on the wire
//! carries the count: K is implied by the request, exactly as the descriptor
//! count is on POSIX.
//!
//! Duplication is a separate step that runs *before* the send, so the
//! all-or-nothing property of `SCM_RIGHTS` has to be restored by hand. A
//! handle duplicated for a message that is then not delivered would sit in the
//! backend's handle table with no message to announce it, so nothing could
//! ever close it — and for a memory table that pins whole guest RAM sections
//! for the life of the connection. [`RemoteHandles`] owns the duplicates and
//! closes them again on drop, unless [`commit`](RemoteHandles::commit) is
//! called once the message has been written in full.
//!
//! Objects transferred this way are created unnamed: like an `SCM_RIGHTS`
//! descriptor, the peer reaches them only because it was handed a handle. The
//! protocol never carries a name.
//!
//! See "Windows platform support" in QEMU's `docs/interop/vhost-user.rst` for
//! the full binding.

use std::io;
use std::os::windows::io::AsRawHandle;
use std::os::windows::io::BorrowedHandle;
use std::os::windows::io::FromRawHandle;
use std::os::windows::io::OwnedHandle;
use std::ptr::null_mut;
use windows_sys::Win32::Foundation::DUPLICATE_CLOSE_SOURCE;
use windows_sys::Win32::Foundation::DUPLICATE_SAME_ACCESS;
use windows_sys::Win32::Foundation::DuplicateHandle;
use windows_sys::Win32::Foundation::GetHandleInformation;
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// Size of a single handle record in a message trailer.
///
/// Matches `VHOST_USER_WIN32_HANDLE_RECORD_SIZE` on the backend side.
pub const HANDLE_RECORD_SIZE: usize = 8;

/// Handles duplicated into the peer process for one message, and not yet
/// handed over to it.
///
/// Dropping this closes every handle it still holds *in the peer*, which is
/// what makes a message that is never delivered leave nothing behind. Call
/// [`commit`](Self::commit) once the message has been written in full; from
/// that point the backend owns the handles and closes them itself, as a POSIX
/// backend closes a received descriptor.
pub struct RemoteHandles<'a> {
    peer: BorrowedHandle<'a>,
    values: Vec<u64>,
    committed: bool,
}

impl<'a> RemoteHandles<'a> {
    /// Create an empty set targeting `peer`, a process handle carrying
    /// `PROCESS_DUP_HANDLE`.
    pub fn new(peer: BorrowedHandle<'a>) -> Self {
        Self {
            peer,
            values: Vec::new(),
            committed: false,
        }
    }

    /// Duplicate `object` into the peer and record the resulting value.
    ///
    /// `DUPLICATE_SAME_ACCESS` always suffices for what this protocol permits
    /// the backend to do with a section or an event, and the duplicate is
    /// non-inheritable: the backend is given the object, not the right to pass
    /// it on to its own children.
    pub fn duplicate(&mut self, object: BorrowedHandle<'_>) -> io::Result<()> {
        let mut dup = null_mut();
        // SAFETY: `self.peer` and `object` are live borrowed handles, and
        // `dup` is a valid out-pointer. The result is checked before use.
        let ok = unsafe {
            DuplicateHandle(
                GetCurrentProcess(),
                object.as_raw_handle(),
                self.peer.as_raw_handle(),
                &mut dup,
                0,
                0,
                DUPLICATE_SAME_ACCESS,
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        self.values.push(dup as usize as u64);
        Ok(())
    }

    /// The trailer to append to the message payload: one record per handle, in
    /// the order they were duplicated.
    pub fn trailer(&self) -> Vec<u8> {
        let mut trailer = Vec::with_capacity(self.values.len() * HANDLE_RECORD_SIZE);
        for value in &self.values {
            // Native byte order, like every other field of the protocol.
            trailer.extend_from_slice(&value.to_le_bytes());
        }
        trailer
    }

    /// Number of handles held.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Give up ownership: the message carrying these handles was delivered, so
    /// the backend owns them now and closing them here would revoke resources
    /// it has been told about.
    pub fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for RemoteHandles<'_> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        for value in &self.values {
            close_in_peer(self.peer, *value);
        }
    }
}

/// Adopt a handle to the backend process, named by its numeric value.
///
/// Whoever launched the backend passes this in as a handle inherited by this
/// process, so the value is meaningful here. Checking it at the point the
/// device is configured means a wrong value is reported then, rather than at
/// the first message that needs to duplicate something.
///
/// Only liveness is checked. Whether the handle is a *process* handle, and
/// whether it carries `PROCESS_DUP_HANDLE`, surfaces when an object is
/// duplicated through it — as a descriptor of the wrong kind does on POSIX.
pub fn adopt_process_handle(value: u64) -> io::Result<OwnedHandle> {
    let handle = value as usize as *mut std::ffi::c_void;
    if handle.is_null() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "backend process handle is null",
        ));
    }

    let mut flags = 0u32;
    // SAFETY: querying an arbitrary handle value is safe; the call reports
    // validity rather than trapping on a bad one.
    if unsafe { GetHandleInformation(handle, &mut flags) } == 0 {
        return Err(io::Error::last_os_error());
    }

    // SAFETY: `handle` is a live handle in this process, inherited from the
    // launcher, and owned by us from here on.
    Ok(unsafe { OwnedHandle::from_raw_handle(handle) })
}

/// Close a handle inside the peer process.
///
/// The undo of the duplication primitive: `DUPLICATE_CLOSE_SOURCE` with no
/// target closes the source handle without creating one here. Errors are
/// ignored — this runs on a failure path, the peer may already be gone, and
/// there is nothing useful to do about it either way.
fn close_in_peer(peer: BorrowedHandle<'_>, value: u64) {
    // SAFETY: `peer` is a live borrowed handle. `value` names a handle in that
    // process, duplicated there by this type and not yet handed over. A null
    // target with `DUPLICATE_CLOSE_SOURCE` is the documented way to ask for a
    // close without a duplicate.
    unsafe {
        DuplicateHandle(
            peer.as_raw_handle(),
            value as usize as *mut _,
            null_mut(),
            null_mut(),
            0,
            0,
            DUPLICATE_CLOSE_SOURCE,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::windows::io::AsHandle;
    use std::os::windows::io::FromRawHandle;
    use std::os::windows::io::OwnedHandle;
    use windows_sys::Win32::Foundation::GetHandleInformation;
    use windows_sys::Win32::System::Threading::CreateEventW;

    /// An unnamed auto-reset event, the kick contract's mode.
    fn event() -> OwnedHandle {
        // SAFETY: all arguments are simple values; the result is checked.
        let handle = unsafe { CreateEventW(null_mut(), 0, 0, std::ptr::null()) };
        assert!(!handle.is_null(), "{}", io::Error::last_os_error());
        // SAFETY: `handle` is a live event handle this test now owns.
        unsafe { OwnedHandle::from_raw_handle(handle) }
    }

    /// Whether `value` names a live handle in this process. The peer in these
    /// tests is this same process, so a remote handle is inspectable directly.
    fn is_live(value: u64) -> bool {
        let mut flags = 0u32;
        // SAFETY: querying an arbitrary handle value is safe; the call reports
        // validity rather than trapping on a bad one.
        unsafe { GetHandleInformation(value as usize as *mut _, &mut flags) != 0 }
    }

    /// A committed set leaves its handles live in the peer.
    #[test]
    fn commit_keeps_handles() {
        let object = event();
        // SAFETY: the current-process pseudo handle is always valid; it is
        // borrowed only for the duration of this call.
        let peer = unsafe { BorrowedHandle::borrow_raw(GetCurrentProcess()) };

        let value = {
            let mut handles = RemoteHandles::new(peer);
            handles.duplicate(object.as_handle()).unwrap();
            let value = u64::from_le_bytes(handles.trailer().try_into().unwrap());
            assert!(is_live(value));
            handles.commit();
            value
        };

        assert!(is_live(value), "committed handle was closed in the peer");
        // SAFETY: `value` names a live handle owned by this process; adopting
        // it here closes it when the test ends.
        drop(unsafe { OwnedHandle::from_raw_handle(value as usize as *mut _) });
    }

    /// Dropping without committing closes the handles in the peer, so an
    /// undelivered message leaves nothing behind.
    #[test]
    fn drop_closes_handles() {
        let object = event();
        // SAFETY: as above.
        let peer = unsafe { BorrowedHandle::borrow_raw(GetCurrentProcess()) };

        let value = {
            let mut handles = RemoteHandles::new(peer);
            handles.duplicate(object.as_handle()).unwrap();
            let value = u64::from_le_bytes(handles.trailer().try_into().unwrap());
            assert!(is_live(value));
            value
        };

        assert!(!is_live(value), "undelivered handle was left in the peer");
        // The original is untouched by the remote close.
        assert!(is_live(object.as_raw_handle() as usize as u64));
    }

    /// The trailer is one record per handle, in duplication order.
    #[test]
    fn trailer_layout() {
        let a = event();
        let b = event();
        // SAFETY: as above.
        let peer = unsafe { BorrowedHandle::borrow_raw(GetCurrentProcess()) };

        let mut handles = RemoteHandles::new(peer);
        handles.duplicate(a.as_handle()).unwrap();
        handles.duplicate(b.as_handle()).unwrap();

        let trailer = handles.trailer();
        assert_eq!(trailer.len(), 2 * HANDLE_RECORD_SIZE);
        assert_eq!(handles.len(), 2);
        for record in trailer.chunks_exact(HANDLE_RECORD_SIZE) {
            assert!(is_live(u64::from_le_bytes(record.try_into().unwrap())));
        }
    }
}
