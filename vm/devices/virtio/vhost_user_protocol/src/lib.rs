// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Shared vhost-user wire protocol types and async socket I/O.
//!
//! This crate is used by both the vhost-user backend (`vhost_user_backend`)
//! and the vhost-user frontend (`vhost_user_frontend`).

//! On Windows the wire protocol is unchanged, but the operating-system objects
//! that POSIX passes as `SCM_RIGHTS` descriptors travel as handles duplicated
//! into the peer process instead; see [`win32`].

#![cfg(any(target_os = "linux", windows))]
// UNSAFETY: FFI into the Win32 handle-duplication APIs. There is no unsafe
// code on the POSIX side, where the kernel does the transfer.
#![cfg_attr(unix, forbid(unsafe_code))]
#![cfg_attr(windows, expect(unsafe_code))]
#![expect(missing_docs)]

pub mod protocol;
pub mod socket;
#[cfg(windows)]
pub mod win32;

pub use protocol::*;
pub use socket::MessageReceiver;
pub use socket::SocketError;
pub use socket::VhostUserSocket;
