// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Launching a virtiofsd daemon for a test VM, and brokering what a
//! vhost-user frontend needs in order to talk to it.
//!
//! virtiofsd is not built by this repository, so tests find it through
//! [`BINARY_ENV`] and skip when it is unset — the same arrangement QEMU's
//! `vhost-user-fs-test` uses for the same daemon.
//!
//! The interesting part is what a launcher owes the frontend on Windows.
//! There, the objects POSIX passes as `SCM_RIGHTS` descriptors — guest RAM
//! sections and the vring doorbells — are duplicated into the daemon's process
//! instead, which takes a handle to that process carrying `PROCESS_DUP_HANDLE`.
//! Nothing about the socket yields one: Windows `AF_UNIX` reports no peer
//! credentials, and a process ID would be racy because IDs are reused. It has
//! to come from whoever created the daemon, which is this module. See
//! "Windows platform support" in QEMU's `docs/interop/vhost-user.rst`.

use crate::PetriLogFile;
use anyhow::Context as _;
use pal_async::DefaultDriver;
use pal_async::pipe::PolledPipe;
use pal_async::task::Spawn;
use pal_async::task::Task;
use pal_async::timer::PolledTimer;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;
use virtio_resources::vhost_user::VhostUserConnection;
use virtio_resources::vhost_user::VhostUserFsHandle;

/// Environment variable naming the virtiofsd binary to test against.
pub const BINARY_ENV: &str = "PETRI_VIRTIOFSD_BINARY";

/// How long to wait for the daemon to start listening.
const LISTEN_TIMEOUT: Duration = Duration::from_secs(10);

/// A running virtiofsd daemon, killed when this is dropped.
pub struct Virtiofsd {
    child: Option<std::process::Child>,
    socket_path: PathBuf,
    _log_task: Task<anyhow::Result<()>>,
}

/// The configured virtiofsd binary, or `None` if [`BINARY_ENV`] is unset.
///
/// A test that needs the daemon should skip when this returns `None`, rather
/// than fail: not every environment has one built.
pub fn binary_path() -> Option<PathBuf> {
    let path = std::env::var_os(BINARY_ENV)?;
    (!path.is_empty()).then(|| PathBuf::from(path))
}

impl Virtiofsd {
    /// Spawn virtiofsd sharing `shared_dir`, returning once it is listening on
    /// `socket_path`.
    ///
    /// `--sandbox none` is required on Windows, which has no equivalent of
    /// mount namespaces or `chroot`, and avoids needing privileges elsewhere.
    pub async fn spawn(
        driver: &DefaultDriver,
        log_file: PetriLogFile,
        shared_dir: &Path,
        socket_path: &Path,
    ) -> anyhow::Result<Self> {
        let binary = binary_path()
            .with_context(|| format!("{BINARY_ENV} is not set; no virtiofsd to launch"))?;

        // One pipe for both streams, so the daemon's output stays in order in
        // the log.
        let (output_read, output_write) = pal::pipe_pair().context("create virtiofsd log pipe")?;
        let child = std::process::Command::new(&binary)
            .arg("--socket-path")
            .arg(socket_path)
            .arg("--shared-dir")
            .arg(shared_dir)
            .arg("--sandbox")
            .arg("none")
            .arg("--log-level")
            .arg("debug")
            .stdout(output_write.try_clone().context("clone virtiofsd stdout")?)
            .stderr(output_write)
            .spawn()
            .with_context(|| format!("failed to spawn virtiofsd at {}", binary.display()))?;

        let log_task = driver.spawn(
            "virtiofsd output",
            crate::log_task(
                log_file,
                PolledPipe::new(driver, output_read).context("poll virtiofsd log pipe")?,
                "virtiofsd",
            ),
        );

        let mut this = Self {
            child: Some(child),
            socket_path: socket_path.to_path_buf(),
            _log_task: log_task,
        };
        this.wait_for_listen(driver).await?;
        Ok(this)
    }

    /// Wait for the daemon's socket to appear, which it creates on listen.
    async fn wait_for_listen(&mut self, driver: &DefaultDriver) -> anyhow::Result<()> {
        let mut timer = PolledTimer::new(driver);
        let deadline = Instant::now() + LISTEN_TIMEOUT;
        while !self.socket_path.exists() {
            // A daemon that died has a reason in the log; say so rather than
            // waiting out the timeout.
            if let Some(status) = self.child.as_mut().unwrap().try_wait()? {
                anyhow::bail!("virtiofsd exited before listening, with status: {status}");
            }
            if Instant::now() > deadline {
                anyhow::bail!(
                    "timed out waiting for virtiofsd to listen on {}",
                    self.socket_path.display()
                );
            }
            timer.sleep(Duration::from_millis(50)).await;
        }
        Ok(())
    }

    /// Connect a control channel to the daemon, brokering everything the
    /// frontend needs to pass objects over it.
    pub fn connect(&self) -> anyhow::Result<VhostUserConnection> {
        let socket = unix_socket::UnixStream::connect(&self.socket_path)
            .with_context(|| format!("connect to virtiofsd at {}", self.socket_path.display()))?;

        Ok(VhostUserConnection {
            socket,
            #[cfg(windows)]
            backend_process: Some(self.process_handle()?),
        })
    }

    /// A handle to the daemon's process, for the frontend to duplicate guest
    /// memory and the doorbells into.
    ///
    /// This is a duplicate rather than the `Child`'s own handle: the `Child`
    /// keeps ownership so it can still wait on and kill the daemon, and the
    /// copy travels to whichever process ends up running the frontend.
    #[cfg(windows)]
    fn process_handle(&self) -> anyhow::Result<std::os::windows::io::OwnedHandle> {
        use pal::windows::BorrowedHandleExt;
        use std::os::windows::io::AsHandle;

        self.child
            .as_ref()
            .context("virtiofsd process is no longer running")?
            .as_handle()
            .duplicate(false, None)
            .context("duplicate the virtiofsd process handle")
    }

    /// A virtio-fs device backed by this daemon, exposed to the guest under
    /// `tag`.
    pub fn fs_handle(&self, tag: &str) -> anyhow::Result<VhostUserFsHandle> {
        Ok(VhostUserFsHandle {
            connection: self.connect()?,
            tag: tag.to_owned(),
            num_queues: None,
            queue_size: None,
        })
    }

    /// Wait for the daemon to exit on its own.
    ///
    /// It serves one connection and exits when the frontend disconnects, so a
    /// test that shut its VM down cleanly can check the daemon agrees.
    pub fn wait(mut self) -> anyhow::Result<std::process::ExitStatus> {
        self.child
            .take()
            .context("virtiofsd already reaped")?
            .wait()
            .context("wait for virtiofsd")
    }
}

impl Drop for Virtiofsd {
    fn drop(&mut self) {
        // Kill rather than wait: on a failing test the daemon is still serving
        // a connection nobody is going to close, and it would outlive the run.
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
