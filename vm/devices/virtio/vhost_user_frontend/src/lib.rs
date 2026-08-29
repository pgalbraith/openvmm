// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#![cfg(any(target_os = "linux", windows))]

//! vhost-user frontend: a [`VirtioDevice`] implementation that forwards
//! device operations to an external vhost-user backend over a Unix socket.
//!
//! The protocol itself is spoken by the `vhost` crate's
//! [`Frontend`](vhost::vhost_user::Frontend), on both platforms: POSIX passes
//! guest memory sections and vring doorbells as `SCM_RIGHTS` descriptors,
//! while on Windows — where `AF_UNIX` sockets carry no ancillary data — they
//! are duplicated into the backend process instead, which is why the Windows
//! constructor takes a handle to that process. This crate maps openvmm's
//! device model onto that frontend: guest memory regions from [`guestmem`],
//! doorbells from [`pal_event`], and the [`VirtioDevice`] contract on top.
//!
//! The `vhost` frontend does blocking socket I/O, so every control-plane call
//! runs on the blocking thread pool via [`blocking::unblock`].

pub mod resolver;

use anyhow::Context as _;
use blocking::unblock;
use guestmem::GuestMemory;
use inspect::InspectMut;
use vhost::VhostBackend as _;
use vhost::VhostUserMemoryRegionInfo;
use vhost::VringConfigData;
use vhost::vhost_user::Frontend;
use vhost::vhost_user::VhostUserFrontend as _;
use vhost::vhost_user::message::VhostUserConfigFlags;
use vhost::vhost_user::message::VhostUserHeaderFlag;
use vhost::vhost_user::message::VhostUserProtocolFeatures;
use virtio::DeviceTraits;
use virtio::DeviceTraitsSharedMemory;
use virtio::QueueResources;
use virtio::VirtioDevice;
use virtio::queue::QueueState;
use virtio::spec::VirtioDeviceFeatures;
use virtio::spec::VirtioDeviceType;
use vmcore::interrupt::EventProxy;
use vmcore::vm_task::VmTaskDriver;

/// Offset added to GPAs to produce the "userspace VA" coordinate system
/// used in the vhost-user protocol. The vhost-user spec expresses vring
/// addresses as frontend userspace VAs, and the backend translates them
/// back using the region table from SET_MEM_TABLE.
///
/// We don't actually map guest memory into our VA space, so we use GPAs
/// as the base coordinate system. This non-zero offset ensures VA != GPA,
/// so any code path that accidentally skips the translation will produce
/// obviously wrong results instead of silently working.
///
/// Set to the maximum ARM64 physical address space size (2^52) so that
/// there is no possible collision with any valid GPA.
const GPA_TO_VA_OFFSET: u64 = 1 << 52;

/// Configuration for creating a `VhostUserFrontend`.
///
/// Each device-type resolver builds an appropriate `VhostUserConfig`:
/// - FS: `use_backend_config: false`, full config as a patch at offset 0
/// - BLK: `use_backend_config: true`, num_queues patched
/// - Generic: `use_backend_config: true`, no patches
pub struct VhostUserConfig {
    /// Virtio device ID (e.g., BLK, FS).
    pub device_id: VirtioDeviceType,
    /// When true, negotiate `VHOST_USER_PROTOCOL_F_CONFIG` with the
    /// backend and use GET_CONFIG/SET_CONFIG for config reads/writes.
    /// When false, config reads start from zeros (patches still apply)
    /// and writes are dropped.
    pub use_backend_config: bool,
    /// Per-queue sizes. Length determines the queue count; must be
    /// non-empty.
    pub queue_sizes: Vec<u16>,
    /// Sparse patches applied to config reads before returning to the
    /// guest. Each entry is `(byte_offset, replacement_bytes)`. The
    /// base is either GET_CONFIG (when `use_backend_config` is true)
    /// or zeros. Writes pass through to SET_CONFIG unchanged when
    /// `use_backend_config` is true.
    pub config_patches: Vec<(u16, Vec<u8>)>,
}

/// Per-queue tracking state.
struct FrontendQueueState {
    active: bool,
    /// Saved queue params for reading used ring index during stop.
    params: Option<virtio::queue::QueueParams>,
    /// Keeps the interrupt event proxy task alive (if one was needed).
    _event_proxy: Option<EventProxy>,
}

/// A `VirtioDevice` that proxies to a vhost-user backend.
#[derive(InspectMut)]
#[inspect(skip)]
pub struct VhostUserFrontend {
    driver: VmTaskDriver,
    device_traits: DeviceTraits,
    protocol_features: VhostUserProtocolFeatures,
    frontend: Frontend,
    /// Per-queue sizes. `queue_size()` indexes into this.
    queue_sizes: Vec<u16>,
    /// Sparse patches applied to config reads. Each entry is
    /// `(byte_offset, replacement_bytes)`.
    config_patches: Vec<(u16, Vec<u8>)>,
    /// Device feature bits from GET_FEATURES (used to mask guest features).
    device_features_raw: VirtioDeviceFeatures,
    guest_features_sent: bool,
    /// Whether packed ring (VIRTIO_F_RING_PACKED) is active. Set when
    /// guest-negotiated features are sent to the backend.
    packed_ring: bool,
    /// Whether the memory table has been sent to the backend. The memory
    /// table is sent once (on the first `start_queue`) and not resent on
    /// reset, because the guest memory backing is the same file-backed
    /// allocation for the lifetime of the socket connection.
    mem_table_sent: bool,
    queues: Vec<FrontendQueueState>,
    /// Set on the first `start_queue` call, used by `stop_queue` to read
    /// the used index from the guest-visible used ring.
    guest_memory: Option<GuestMemory>,
}

/// Run a blocking vhost-user control-plane call on the blocking thread pool.
///
/// `Frontend` is a cheap handle sharing one connection, so the clone moved
/// into the closure speaks over the same socket; its methods do blocking
/// socket I/O, which must stay off the async executor threads.
async fn call<R: 'static + Send>(
    frontend: &Frontend,
    f: impl FnOnce(&mut Frontend) -> vhost::Result<R> + 'static + Send,
) -> vhost::Result<R> {
    let mut frontend = frontend.clone();
    unblock(move || f(&mut frontend)).await
}

// UNSAFETY: needed to transfer ownership of duplicated OS objects into the
// `vhost` crate's wrapper types, which only offer raw-descriptor
// constructors.
#[expect(unsafe_code)]
mod convert {
    use vmm_sys_util::eventfd::EventFd;

    /// Duplicate `event` into the [`EventFd`] wrapper the `vhost` crate
    /// sends doorbells from. The wire only carries the underlying object, so
    /// the wrapper's read/write semantics never come into play here.
    pub(crate) fn event_to_eventfd(event: &pal_event::Event) -> std::io::Result<EventFd> {
        #[cfg(unix)]
        {
            use std::os::fd::AsFd;
            use std::os::fd::FromRawFd;
            use std::os::fd::IntoRawFd;
            let fd = event.as_fd().try_clone_to_owned()?;
            // SAFETY: `fd` is an owned duplicate whose ownership transfers to
            // the returned `EventFd`, which closes it on drop.
            Ok(unsafe { EventFd::from_raw_fd(fd.into_raw_fd()) })
        }
        #[cfg(windows)]
        {
            use std::os::windows::io::AsHandle;
            use std::os::windows::io::FromRawHandle;
            use std::os::windows::io::IntoRawHandle;
            let handle = event.as_handle().try_clone_to_owned()?;
            // SAFETY: `handle` is an owned duplicate whose ownership
            // transfers to the returned `EventFd`, which closes it on drop.
            Ok(unsafe { EventFd::from_raw_handle(handle.into_raw_handle()) })
        }
    }

    /// Rehouse a connected socket in the stream type the `vhost` crate's
    /// Windows frontend takes.
    #[cfg(windows)]
    pub(crate) fn stream_to_uds(stream: unix_socket::UnixStream) -> uds_windows::UnixStream {
        use std::os::windows::io::FromRawSocket;
        use std::os::windows::io::IntoRawSocket;
        let socket: std::os::windows::io::OwnedSocket = stream.into();
        // SAFETY: `socket` is owned and its ownership transfers to the
        // returned stream, which closes it on drop.
        unsafe { uds_windows::UnixStream::from_raw_socket(socket.into_raw_socket()) }
    }
}

impl VhostUserFrontend {
    /// Create from a connected socket.
    #[cfg(unix)]
    pub async fn from_stream(
        driver: VmTaskDriver,
        stream: unix_socket::UnixStream,
        config: VhostUserConfig,
    ) -> anyhow::Result<Self> {
        let frontend = Frontend::from_stream(stream, config.queue_sizes.len() as u64);
        Self::new(driver, frontend, config).await
    }

    /// Create from a connected socket and a handle to the backend process.
    ///
    /// The handle needs `PROCESS_DUP_HANDLE` access: guest memory sections
    /// and vring doorbells are duplicated into the backend process rather
    /// than passed as descriptors, since Windows `AF_UNIX` sockets carry no
    /// ancillary data. It must come from whoever launched the backend — it
    /// cannot be derived from the socket, and a process ID would be racy
    /// because IDs are reused.
    #[cfg(windows)]
    pub async fn from_stream(
        driver: VmTaskDriver,
        stream: unix_socket::UnixStream,
        backend_process: std::os::windows::io::OwnedHandle,
        config: VhostUserConfig,
    ) -> anyhow::Result<Self> {
        let frontend = Frontend::from_stream(
            convert::stream_to_uds(stream),
            config.queue_sizes.len() as u64,
            backend_process,
        );
        Self::new(driver, frontend, config).await
    }

    /// Create from an already-constructed `vhost` frontend, running the
    /// vhost-user handshake.
    pub async fn new(
        driver: VmTaskDriver,
        frontend: Frontend,
        config: VhostUserConfig,
    ) -> anyhow::Result<Self> {
        // 1. GET_FEATURES
        let device_features_raw = VirtioDeviceFeatures::from_bits(
            call(&frontend, |f| f.get_features())
                .await
                .context("GET_FEATURES failed")?,
        );
        tracing::trace!(features = %format!("0x{:x}", device_features_raw.into_bits()), "GET_FEATURES");

        // 2. Negotiate protocol features (only if the backend advertises them).
        let negotiated_proto = if device_features_raw.vhost_user_protocol_features() {
            let advertised = call(&frontend, |f| f.get_protocol_features())
                .await
                .context("GET_PROTOCOL_FEATURES failed")?;
            let mut wanted = VhostUserProtocolFeatures::MQ
                | VhostUserProtocolFeatures::REPLY_ACK
                | VhostUserProtocolFeatures::RESET_DEVICE;
            if config.use_backend_config {
                wanted |= VhostUserProtocolFeatures::CONFIG;
            }
            let negotiated = advertised & wanted;
            call(&frontend, move |f| f.set_protocol_features(negotiated))
                .await
                .context("SET_PROTOCOL_FEATURES failed")?;
            // From here on, ask for and wait on REPLY_ACK acknowledgments if
            // they were negotiated. (The negotiation message itself went out
            // before this flag was set, as it must.)
            if negotiated.contains(VhostUserProtocolFeatures::REPLY_ACK) {
                frontend.set_hdr_flags(VhostUserHeaderFlag::NEED_REPLY);
            }
            negotiated
        } else {
            VhostUserProtocolFeatures::empty()
        };

        // 3. SET_OWNER
        call(&frontend, |f| f.set_owner())
            .await
            .context("SET_OWNER failed")?;

        // 4. GET_QUEUE_NUM (requires MQ protocol feature)
        let backend_max_queues = if negotiated_proto.contains(VhostUserProtocolFeatures::MQ) {
            Some(
                call(&frontend, |f| f.get_queue_num())
                    .await
                    .context("GET_QUEUE_NUM failed despite MQ being negotiated")?
                    as u16,
            )
        } else {
            None
        };
        tracing::trace!(?backend_max_queues, "GET_QUEUE_NUM");

        // Validate the requested queue count against the backend.
        anyhow::ensure!(
            !config.queue_sizes.is_empty(),
            "queue_sizes must be non-empty"
        );
        let max_queues = u16::try_from(config.queue_sizes.len()).map_err(|_| {
            anyhow::anyhow!(
                "queue_sizes has {} entries, exceeding maximum supported queue count {}",
                config.queue_sizes.len(),
                u16::MAX
            )
        })?;
        if let Some(backend_max) = backend_max_queues {
            anyhow::ensure!(
                max_queues <= backend_max,
                "requested {max_queues} queues but backend supports at most {backend_max}"
            );
        }
        let queue_sizes = config.queue_sizes;

        // Build DeviceTraits from the wire features.
        let device_features = device_features_raw.with_vhost_user_protocol_features(false);

        // Determine the config register length.
        //
        // When the backend supports GET_CONFIG, use the vhost-user max
        // config size (256); reads beyond the backend's actual config
        // space will return zeros. Otherwise, derive the length from
        // the patches (the guest only sees patched fields).
        let device_register_length = if negotiated_proto.contains(VhostUserProtocolFeatures::CONFIG)
        {
            256
        } else {
            config
                .config_patches
                .iter()
                .map(|(off, data)| *off as u32 + data.len() as u32)
                .max()
                .unwrap_or(0)
        };

        let device_traits = DeviceTraits {
            device_id: config.device_id,
            device_features,
            max_queues,
            device_register_length,
            shared_memory: DeviceTraitsSharedMemory::default(),
        };

        let queues = (0..max_queues)
            .map(|_| FrontendQueueState {
                active: false,
                params: None,
                _event_proxy: None,
            })
            .collect();

        Ok(Self {
            driver,
            device_traits,
            protocol_features: negotiated_proto,
            frontend,
            queue_sizes,
            config_patches: config.config_patches,
            device_features_raw,
            guest_features_sent: false,
            mem_table_sent: false,
            packed_ring: false,
            queues,
            guest_memory: None,
        })
    }
}

impl VirtioDevice for VhostUserFrontend {
    fn traits(&self) -> DeviceTraits {
        self.device_traits.clone()
    }

    fn queue_size(&self, queue_index: u16) -> u16 {
        self.queue_sizes[queue_index as usize]
    }

    async fn read_registers_u32(&mut self, offset: u16) -> u32 {
        let mut buf = if self
            .protocol_features
            .contains(VhostUserProtocolFeatures::CONFIG)
        {
            tracing::trace!(offset, "GET_CONFIG");
            match call(&self.frontend, move |f| {
                f.get_config(offset as u32, 4, VhostUserConfigFlags::empty(), &[0u8; 4])
            })
            .await
            {
                Ok((_, data)) if data.len() >= 4 => {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&data[..4]);
                    b
                }
                Ok(_) => [0u8; 4],
                Err(e) => {
                    tracelimit::warn_ratelimited!(
                        error = &e as &dyn std::error::Error,
                        offset,
                        "GET_CONFIG failed"
                    );
                    [0u8; 4]
                }
            }
        } else {
            [0u8; 4]
        };

        // Apply config patches to the read buffer.
        for (patch_offset, patch_data) in &self.config_patches {
            let p_start = *patch_offset as usize;
            let p_end = p_start + patch_data.len();
            let r_start = offset as usize;
            let r_end = r_start + 4;
            // Check for overlap.
            if p_start < r_end && p_end > r_start {
                let overlap_start = p_start.max(r_start);
                let overlap_end = p_end.min(r_end);
                let buf_offset = overlap_start - r_start;
                let patch_src_offset = overlap_start - p_start;
                let len = overlap_end - overlap_start;
                buf[buf_offset..buf_offset + len]
                    .copy_from_slice(&patch_data[patch_src_offset..patch_src_offset + len]);
            }
        }

        u32::from_le_bytes(buf)
    }

    async fn write_registers_u32(&mut self, offset: u16, val: u32) {
        if !self
            .protocol_features
            .contains(VhostUserProtocolFeatures::CONFIG)
        {
            return;
        }

        tracing::trace!(offset, "SET_CONFIG");
        if let Err(e) = call(&self.frontend, move |f| {
            f.set_config(
                offset as u32,
                VhostUserConfigFlags::empty(),
                &val.to_le_bytes(),
            )
        })
        .await
        {
            tracelimit::warn_ratelimited!(
                error = &e as &dyn std::error::Error,
                offset,
                "SET_CONFIG failed"
            );
        }
    }

    async fn start_queue(
        &mut self,
        idx: u16,
        resources: QueueResources,
        features: &VirtioDeviceFeatures,
        initial_state: Option<QueueState>,
    ) -> anyhow::Result<()> {
        // Send SET_MEM_TABLE before the first queue is started.
        //
        // The memory table is sent once and persists across device
        // resets. The backend retains the memory mapping (it is
        // connection-scoped, not device-scoped), and the guest memory
        // backing doesn't change for the lifetime of the connection.
        if !self.mem_table_sent {
            let sharing = resources.guest_memory.sharing().ok_or_else(|| {
                anyhow::anyhow!(
                    "vhost-user requires file-backed guest memory (sharing() returned None)"
                )
            })?;
            let exported_regions = sharing
                .get_regions()
                .await
                .map_err(|e| anyhow::anyhow!(e))?;

            tracing::trace!(region_count = exported_regions.len(), "SET_MEM_TABLE");
            for (i, region) in exported_regions.iter().enumerate() {
                tracing::trace!(
                    i,
                    gpa = %format!("0x{:x}", region.guest_address),
                    size = %format!("0x{:x}", region.size),
                    userspace_addr = %format!("0x{:x}", region.guest_address + GPA_TO_VA_OFFSET),
                    mmap_offset = %format!("0x{:x}", region.file_offset),
                    "SET_MEM_TABLE region",
                );
            }
            // The region table is built inside the closure so that the
            // backing file objects stay alive for the duration of the send;
            // the table itself carries only their raw descriptors.
            call(&self.frontend, move |f| {
                let regions = exported_regions
                    .iter()
                    .map(|region| {
                        #[cfg(unix)]
                        let raw = {
                            use std::os::fd::AsRawFd;
                            region.file.as_raw_fd()
                        };
                        #[cfg(windows)]
                        let raw = {
                            use std::os::windows::io::AsRawHandle;
                            region.file.as_raw_handle()
                        };
                        VhostUserMemoryRegionInfo {
                            guest_phys_addr: region.guest_address,
                            memory_size: region.size,
                            userspace_addr: region.guest_address + GPA_TO_VA_OFFSET,
                            mmap_offset: region.file_offset,
                            mmap_handle: raw,
                        }
                    })
                    .collect::<Vec<_>>();
                f.set_mem_table(&regions)
            })
            .await
            .context("SET_MEM_TABLE failed")?;
            self.mem_table_sent = true;
            self.guest_memory = Some(resources.guest_memory.clone());
        }

        // Send SET_FEATURES with the guest-negotiated features before the
        // first queue is started.  The backend needs this to know which
        // features are active.
        if !self.guest_features_sent {
            // Mask to only include features the backend actually advertised.
            // The VMM transport may add features (e.g., RING_PACKED) that
            // the backend doesn't support. Always include PROTOCOL_FEATURES
            // if it was negotiated — backends (e.g., virtiofsd) treat its
            // absence in SET_FEATURES as de-negotiation.
            let negotiated = VirtioDeviceFeatures::from_bits(
                features.into_bits() & self.device_features_raw.into_bits(),
            );
            let on_wire = negotiated.with_vhost_user_protocol_features(true);
            tracing::trace!(
                idx,
                features = %format!("0x{:x}", on_wire.into_bits()),
                "SET_FEATURES (guest-negotiated)",
            );
            call(&self.frontend, move |f| f.set_features(on_wire.into_bits()))
                .await
                .context("SET_FEATURES failed")?;
            self.guest_features_sent = true;
            self.packed_ring = negotiated.ring_packed();
        }

        let packed_ring = self.packed_ring;

        let base = initial_state.map(|s| s.avail_index).unwrap_or_else(|| {
            // For packed ring, the initial wrap counter is 1 (encoded in bit 15).
            if packed_ring { 0x8000 } else { 0 }
        });

        // For packed ring, SET_VRING_BASE packs both avail and used state:
        //   bits 0-14: last avail index
        //   bit 15: avail wrap counter
        //   bits 16-30: last used index
        //   bit 31: used wrap counter
        // For fresh start, used == avail. For save/restore, used comes from
        // the saved state.
        let vring_base = if packed_ring {
            let used = initial_state.map(|s| s.used_index).unwrap_or(0x8000);
            (base as u32) | ((used as u32) << 16)
        } else {
            base as u32
        };

        tracing::trace!(
            idx,
            size = resources.params.size,
            desc = %format!("0x{:x}", resources.params.desc_addr),
            avail = %format!("0x{:x}", resources.params.avail_addr),
            used = %format!("0x{:x}", resources.params.used_addr),
            base,
            has_event_interrupt = resources.notify.event().is_some(),
            "start_queue",
        );

        // SET_VRING_NUM
        let size = resources.params.size;
        call(&self.frontend, move |f| f.set_vring_num(idx.into(), size))
            .await
            .context("SET_VRING_NUM failed")?;

        // SET_VRING_ADDR — addresses must be in the VA coordinate system
        // (GPA + GPA_TO_VA_OFFSET) matching what we sent in SET_MEM_TABLE.
        let vring_config = VringConfigData {
            queue_max_size: size,
            queue_size: size,
            flags: 0,
            desc_table_addr: resources.params.desc_addr + GPA_TO_VA_OFFSET,
            used_ring_addr: resources.params.used_addr + GPA_TO_VA_OFFSET,
            avail_ring_addr: resources.params.avail_addr + GPA_TO_VA_OFFSET,
            log_addr: None,
        };
        call(&self.frontend, move |f| {
            f.set_vring_addr(idx.into(), &vring_config)
        })
        .await
        .context("SET_VRING_ADDR failed")?;

        // SET_VRING_BASE. The u16 trait method covers a split ring's avail
        // index; a packed ring's initial state needs the full 32-bit field.
        tracing::trace!(idx, vring_base = %format!("0x{vring_base:x}"), "SET_VRING_BASE");
        call(&self.frontend, move |f| {
            if packed_ring {
                f.set_vring_base_raw(idx.into(), vring_base)
            } else {
                f.set_vring_base(idx.into(), vring_base as u16)
            }
        })
        .await
        .context("SET_VRING_BASE failed")?;

        // SET_VRING_KICK — pass the kick event to the backend.
        let kick = convert::event_to_eventfd(&resources.event)
            .context("failed to duplicate kick event")?;
        call(&self.frontend, move |f| f.set_vring_kick(idx.into(), &kick))
            .await
            .context("SET_VRING_KICK failed")?;

        // SET_VRING_CALL — pass an interrupt event to the backend.
        //
        // If the transport's interrupt is already event-backed, pass it
        // directly. Otherwise, create an async proxy that bridges a new
        // event to Interrupt::deliver() (needed for e.g. MSI-X function-
        // backed interrupts where the transport has side effects like
        // updating the ISR register).
        let (call_event, event_proxy) = resources.notify.event_or_proxy(&self.driver)?;
        let call_fd = convert::event_to_eventfd(&call_event)
            .context("failed to duplicate interrupt event")?;
        call(&self.frontend, move |f| {
            f.set_vring_call(idx.into(), &call_fd)
        })
        .await
        .context("SET_VRING_CALL failed")?;

        // SET_VRING_ENABLE
        call(&self.frontend, move |f| {
            f.set_vring_enable(idx.into(), true)
        })
        .await
        .context("SET_VRING_ENABLE failed")?;

        if let Some(q) = self.queues.get_mut(idx as usize) {
            q.active = true;
            q.params = Some(resources.params);
            q._event_proxy = event_proxy;
        }
        Ok(())
    }

    async fn stop_queue(&mut self, idx: u16) -> Option<QueueState> {
        let q = self.queues.get_mut(idx as usize)?;
        if !q.active {
            return None;
        }

        // Disable the queue before stopping it. QEMU sends
        // SET_VRING_ENABLE(0) before GET_VRING_BASE to ensure the
        // backend's data plane stops processing kicks before the
        // control plane tears down the queue.
        if let Err(e) = call(&self.frontend, move |f| {
            f.set_vring_enable(idx.into(), false)
        })
        .await
        {
            tracelimit::warn_ratelimited!(
                error = &e as &dyn std::error::Error,
                idx,
                "SET_VRING_ENABLE(0) failed during stop_queue"
            );
        }

        // GET_VRING_BASE implicitly stops the queue on the backend.
        // For packed ring, the reply packs both avail and used state:
        //   bits 0-15: avail state (index + wrap counter)
        //   bits 16-31: used state (index + wrap counter)
        // For split ring, only the low 16 bits matter (avail index),
        // and used_index is read from the guest-visible used ring.
        let vring_base = match call(&self.frontend, move |f| f.get_vring_base(idx.into())).await {
            Ok(base) => base,
            Err(e) => {
                tracelimit::warn_ratelimited!(
                    error = &e as &dyn std::error::Error,
                    idx,
                    "GET_VRING_BASE failed during stop_queue; marking queue inactive"
                );
                q.active = false;
                q.params = None;
                q._event_proxy = None;
                return None;
            }
        };

        let (avail_index, used_index) = if self.packed_ring {
            (vring_base as u16, (vring_base >> 16) as u16)
        } else {
            let used = q
                .params
                .as_ref()
                .map(|params| {
                    read_used_index(
                        self.guest_memory
                            .as_ref()
                            .expect("memory set in start_queue"),
                        params,
                    )
                })
                .unwrap_or(0);
            (vring_base as u16, used)
        };

        q.active = false;
        q.params = None;
        q._event_proxy = None;
        Some(QueueState {
            avail_index,
            used_index,
        })
    }

    async fn reset(&mut self) {
        // Stop all active queues.
        for idx in 0..self.queues.len() {
            if self.queues[idx].active {
                if let Err(e) = call(&self.frontend, move |f| f.set_vring_enable(idx, false)).await
                {
                    tracelimit::warn_ratelimited!(
                        error = &e as &dyn std::error::Error,
                        idx,
                        "SET_VRING_ENABLE(0) failed during reset"
                    );
                }
                if let Err(e) = call(&self.frontend, move |f| f.get_vring_base(idx)).await {
                    tracelimit::warn_ratelimited!(
                        error = &e as &dyn std::error::Error,
                        idx,
                        "GET_VRING_BASE failed during reset"
                    );
                }
                self.queues[idx].active = false;
                self.queues[idx].params = None;
                self.queues[idx]._event_proxy = None;
            }
        }
        self.guest_features_sent = false;
        self.packed_ring = false;
        // Send RESET_DEVICE if negotiated.
        if self
            .protocol_features
            .contains(VhostUserProtocolFeatures::RESET_DEVICE)
        {
            if let Err(e) = call(&self.frontend, |f| f.reset_device()).await {
                tracelimit::warn_ratelimited!(
                    error = &e as &dyn std::error::Error,
                    "RESET_DEVICE failed during reset"
                );
            }
        }
    }

    fn supports_save_restore(&self) -> bool {
        true
    }
}

/// Read the used_index from the used ring in guest memory.
///
/// The used ring starts at `params.used_addr`. The `idx` field is at
/// offset 2 (after the flags field) and is a 16-bit LE value.
fn read_used_index(mem: &GuestMemory, params: &virtio::queue::QueueParams) -> u16 {
    let mut buf = [0u8; 2];
    // used ring layout: { flags: u16, idx: u16, ... }
    if mem.read_at(params.used_addr + 2, &mut buf).is_ok() {
        u16::from_le_bytes(buf)
    } else {
        0
    }
}

// These tests drive a real backend over a loopback socket, which means
// `vhost_user_backend` — openvmm's own backend, and Linux-only. The frontend
// half runs through the `vhost` crate, so they also cross-check the two
// implementations against each other. The Windows transport's own contract is
// covered by the `vhost` crate's tests; end to end it is exercised against an
// external backend.
#[cfg(all(test, target_os = "linux"))]
// UNSAFETY: Implementing GuestMemoryAccess for test-only ShareableGuestMemory.
#[expect(unsafe_code)]
mod tests {
    use super::*;
    use guestmem::GuestMemorySharing;
    use guestmem::ProvideShareableRegions;
    use guestmem::ShareableRegion;
    use guestmem::ShareableRegionError;
    use pal_async::DefaultDriver;
    use pal_async::async_test;
    use pal_async::socket::PolledSocket;
    use pal_async::task::Spawn;
    use pal_event::Event;
    use sparse_mmap::Mappable;
    use sparse_mmap::SparseMapping;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU32;
    use std::sync::atomic::Ordering;
    use test_with_tracing::test;
    use unix_socket::UnixStream;
    use vhost_user_backend::VhostUserDeviceServer;
    use vhost_user_protocol::VhostUserSocket;
    use virtio::DEFAULT_QUEUE_SIZE;
    use virtio::DeviceTraits;
    use virtio::DeviceTraitsSharedMemory;
    use virtio::QueueResources;
    use virtio::VirtioDevice;
    use virtio::queue::QueueParams;
    use virtio::queue::QueueState;
    use virtio::spec::VirtioDeviceFeatures;
    use vmcore::interrupt::Interrupt;
    use vmcore::vm_task::SingleDriverBackend;
    use vmcore::vm_task::VmTaskDriverSource;

    /// File-backed guest memory that supports `sharing()`.
    struct ShareableGuestMemory {
        mapping: SparseMapping,
        fd: Arc<Mappable>,
        size: u64,
    }

    impl ShareableGuestMemory {
        fn new(size: usize) -> Self {
            let fd = sparse_mmap::alloc_shared_memory(size, "test-guest-memory")
                .expect("alloc_shared_memory failed");
            let mapping = SparseMapping::new(size).expect("SparseMapping::new failed");
            mapping
                .map_file(0, size, fd.try_clone().unwrap(), 0, true)
                .expect("map_file failed");
            Self {
                mapping,
                fd: Arc::new(fd),
                size: size as u64,
            }
        }

        fn into_guest_memory(self) -> GuestMemory {
            GuestMemory::new("test-shareable", self)
        }
    }

    // SAFETY: SparseMapping's pointer is valid for the lifetime of the mapping
    // and the fd is a shareable file descriptor.
    unsafe impl guestmem::GuestMemoryAccess for ShareableGuestMemory {
        fn mapping(&self) -> Option<std::ptr::NonNull<u8>> {
            std::ptr::NonNull::new(self.mapping.as_ptr().cast())
        }

        fn max_address(&self) -> u64 {
            self.size
        }

        fn sharing(&self) -> Option<GuestMemorySharing> {
            Some(GuestMemorySharing::new(TestRegionProvider {
                fd: self.fd.clone(),
                size: self.size,
            }))
        }
    }

    struct TestRegionProvider {
        fd: Arc<Mappable>,
        size: u64,
    }

    impl ProvideShareableRegions for TestRegionProvider {
        async fn get_regions(&self) -> Result<Vec<ShareableRegion>, ShareableRegionError> {
            Ok(vec![ShareableRegion {
                guest_address: 0,
                size: self.size,
                file: self.fd.clone(),
                file_offset: 0,
            }])
        }
    }

    /// A mock VirtioDevice for the backend side of the dog-food test.
    struct MockBackendDevice {
        traits: DeviceTraits,
        started_queues: Vec<u16>,
    }

    impl MockBackendDevice {
        fn new() -> Self {
            Self {
                traits: DeviceTraits {
                    device_id: VirtioDeviceType::BLK,
                    device_features: VirtioDeviceFeatures::new(),
                    max_queues: 2,
                    device_register_length: 0,
                    shared_memory: DeviceTraitsSharedMemory::default(),
                },
                started_queues: Vec::new(),
            }
        }
    }

    impl InspectMut for MockBackendDevice {
        fn inspect_mut(&mut self, _req: inspect::Request<'_>) {}
    }

    impl VirtioDevice for MockBackendDevice {
        fn traits(&self) -> DeviceTraits {
            self.traits.clone()
        }

        async fn read_registers_u32(&mut self, _offset: u16) -> u32 {
            0
        }

        async fn write_registers_u32(&mut self, _offset: u16, _val: u32) {}

        async fn start_queue(
            &mut self,
            idx: u16,
            _resources: QueueResources,
            _features: &VirtioDeviceFeatures,
            _initial_state: Option<QueueState>,
        ) -> anyhow::Result<()> {
            self.started_queues.push(idx);
            Ok(())
        }

        async fn stop_queue(&mut self, idx: u16) -> Option<QueueState> {
            if self.started_queues.contains(&idx) {
                self.started_queues.retain(|&x| x != idx);
                Some(QueueState {
                    avail_index: 42,
                    used_index: 0,
                })
            } else {
                None
            }
        }

        async fn reset(&mut self) {
            self.started_queues.clear();
        }
    }

    fn socket_pair() -> (UnixStream, UnixStream) {
        let (a, b) = socket2::Socket::pair(socket2::Domain::UNIX, socket2::Type::STREAM, None)
            .expect("socketpair failed");
        (a.into(), b.into())
    }

    /// Create a frontend+backend pair over a socketpair. Returns the frontend
    /// and a handle to the backend task (drop the frontend to let it finish).
    async fn setup_frontend_backend(
        driver: &DefaultDriver,
    ) -> (VhostUserFrontend, GuestMemory, pal_async::task::Task<()>) {
        setup_frontend_backend_with_config(
            driver,
            MockBackendDevice::new(),
            VhostUserConfig {
                device_id: VirtioDeviceType::BLK,
                use_backend_config: true,
                queue_sizes: vec![DEFAULT_QUEUE_SIZE; 2],
                config_patches: vec![],
            },
        )
        .await
    }

    /// Build dummy QueueResources with the given interrupt.
    fn dummy_queue_resources(notify: Interrupt, guest_memory: GuestMemory) -> QueueResources {
        QueueResources {
            params: QueueParams {
                size: 16,
                enable: true,
                desc_addr: 0x0000,
                avail_addr: 0x1000,
                used_addr: 0x2000,
            },
            notify,
            event: Event::new(),
            guest_memory,
        }
    }

    #[async_test]
    async fn frontend_backend_dogfood(driver: DefaultDriver) {
        let (mut frontend, _guest_memory, backend_task) = setup_frontend_backend(&driver).await;

        // Verify traits.
        let traits = frontend.traits();
        assert_eq!(traits.device_id, VirtioDeviceType::BLK);
        assert_eq!(traits.max_queues, 2);

        // Reset.
        frontend.reset().await;
        assert!(frontend.supports_save_restore());

        drop(frontend);
        backend_task.await;
    }

    /// Test start_queue + stop_queue with an event-backed interrupt.
    /// When the interrupt is event-backed, the event is passed directly
    /// to SET_VRING_CALL with no proxy task needed.
    #[async_test]
    async fn start_stop_queue_event_interrupt(driver: DefaultDriver) {
        let (mut frontend, guest_memory, backend_task) = setup_frontend_backend(&driver).await;

        let features = VirtioDeviceFeatures::new();
        let resources =
            dummy_queue_resources(Interrupt::from_event(Event::new()), guest_memory.clone());

        frontend
            .start_queue(0, resources, &features, None)
            .await
            .expect("start_queue failed");

        // Event-backed interrupt — no proxy needed.
        assert!(frontend.queues[0]._event_proxy.is_none());

        // Stop the queue and verify we get state back.
        let state = frontend.stop_queue(0).await;
        assert!(state.is_some());

        // Stopping again should return None.
        let state2 = frontend.stop_queue(0).await;
        assert!(state2.is_none());

        drop(frontend);
        backend_task.await;
    }

    /// Test start_queue + stop_queue with a function-backed interrupt.
    /// Verifies the proxy works with non-event interrupts (e.g., MSI-X).
    #[async_test]
    async fn start_stop_queue_fn_interrupt(driver: DefaultDriver) {
        let (mut frontend, guest_memory, backend_task) = setup_frontend_backend(&driver).await;

        let features = VirtioDeviceFeatures::new();
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();
        let resources = dummy_queue_resources(
            Interrupt::from_fn(move || {
                counter_clone.fetch_add(1, Ordering::Relaxed);
            }),
            guest_memory.clone(),
        );

        // Verify this is NOT event-backed — will exercise the proxy path.
        assert!(resources.notify.event().is_none());

        frontend
            .start_queue(0, resources, &features, None)
            .await
            .expect("start_queue failed");

        // Stop the queue — this drops the InterruptEvent and its proxy task.
        let state = frontend.stop_queue(0).await;
        assert!(state.is_some());

        drop(frontend);
        backend_task.await;
    }

    /// Test that the interrupt proxy actually forwards signals.
    #[async_test]
    async fn interrupt_proxy_delivers(driver: DefaultDriver) {
        let (mut frontend, guest_memory, backend_task) = setup_frontend_backend(&driver).await;

        let features = VirtioDeviceFeatures::new();
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();
        let resources = dummy_queue_resources(
            Interrupt::from_fn(move || {
                counter_clone.fetch_add(1, Ordering::Relaxed);
            }),
            guest_memory.clone(),
        );

        frontend
            .start_queue(0, resources, &features, None)
            .await
            .expect("start_queue failed");

        // The proxy task is waiting on the event we sent via SET_VRING_CALL.
        // The backend holds a clone of that event. When the backend signals it,
        // the proxy should call our counter fn.
        //
        // We can't directly poke the backend's event from here, but we can
        // verify the proxy was set up since the interrupt is fn-backed.
        assert!(frontend.queues[0]._event_proxy.is_some());

        let state = frontend.stop_queue(0).await;
        assert!(state.is_some());

        // Proxy should be torn down.
        assert!(frontend.queues[0]._event_proxy.is_none());

        drop(frontend);
        backend_task.await;
    }

    /// Test that reset clears the guest_features_sent flag and stops queues.
    #[async_test]
    async fn reset_clears_state(driver: DefaultDriver) {
        let (mut frontend, guest_memory, backend_task) = setup_frontend_backend(&driver).await;

        let features = VirtioDeviceFeatures::new();

        // Start queue 0.
        let resources =
            dummy_queue_resources(Interrupt::from_event(Event::new()), guest_memory.clone());
        frontend
            .start_queue(0, resources, &features, None)
            .await
            .expect("start_queue failed");

        assert!(frontend.guest_features_sent);

        // Reset should stop all queues and clear the features flag.
        frontend.reset().await;

        assert!(!frontend.guest_features_sent);
        assert!(!frontend.queues[0].active);

        // Stopping a non-active queue returns None.
        assert!(frontend.stop_queue(0).await.is_none());

        // Can start again after reset (SET_FEATURES will be re-sent).
        let resources2 =
            dummy_queue_resources(Interrupt::from_event(Event::new()), guest_memory.clone());
        frontend
            .start_queue(0, resources2, &features, None)
            .await
            .expect("start_queue after reset failed");

        assert!(frontend.guest_features_sent);

        drop(frontend);
        backend_task.await;
    }

    /// Create a frontend+backend pair with a custom device. The device's
    /// traits determine feature negotiation (e.g., packed ring support).
    async fn setup_frontend_backend_with_device(
        driver: &DefaultDriver,
        device: impl VirtioDevice + 'static,
    ) -> (VhostUserFrontend, GuestMemory, pal_async::task::Task<()>) {
        setup_frontend_backend_with_config(
            driver,
            device,
            VhostUserConfig {
                device_id: VirtioDeviceType::BLK,
                use_backend_config: true,
                queue_sizes: vec![DEFAULT_QUEUE_SIZE; 2],
                config_patches: vec![],
            },
        )
        .await
    }

    /// Create a frontend+backend pair with a custom device and config.
    async fn setup_frontend_backend_with_config(
        driver: &DefaultDriver,
        device: impl VirtioDevice + 'static,
        config: VhostUserConfig,
    ) -> (VhostUserFrontend, GuestMemory, pal_async::task::Task<()>) {
        let (frontend_stream, backend_stream) = socket_pair();

        let backend_polled = PolledSocket::new(driver, backend_stream).unwrap();
        let backend_socket = VhostUserSocket::new(backend_polled);

        let server = VhostUserDeviceServer::new(Box::new(device));

        let backend_task = driver.spawn("backend", async move {
            server.serve_connection(backend_socket).await.unwrap();
        });

        let guest_memory = ShareableGuestMemory::new(65536).into_guest_memory();

        let vm_driver = VmTaskDriverSource::new(SingleDriverBackend::new(driver.clone())).simple();
        let frontend = VhostUserFrontend::from_stream(vm_driver, frontend_stream, config)
            .await
            .expect("frontend handshake failed");

        (frontend, guest_memory, backend_task)
    }

    /// Mock device that returns specific avail/used indices on stop and
    /// optionally advertises packed ring.
    struct SaveRestoreMockDevice {
        traits: DeviceTraits,
        started_queues: Vec<u16>,
        stop_avail: u16,
        stop_used: u16,
    }

    impl SaveRestoreMockDevice {
        fn new(packed_ring: bool, stop_avail: u16, stop_used: u16) -> Self {
            let features = VirtioDeviceFeatures::new().with_ring_packed(packed_ring);
            Self {
                traits: DeviceTraits {
                    device_id: VirtioDeviceType::BLK,
                    device_features: features,
                    max_queues: 2,
                    device_register_length: 0,
                    shared_memory: DeviceTraitsSharedMemory::default(),
                },
                started_queues: Vec::new(),
                stop_avail,
                stop_used,
            }
        }
    }

    impl InspectMut for SaveRestoreMockDevice {
        fn inspect_mut(&mut self, _req: inspect::Request<'_>) {}
    }

    impl VirtioDevice for SaveRestoreMockDevice {
        fn traits(&self) -> DeviceTraits {
            self.traits.clone()
        }

        async fn read_registers_u32(&mut self, _offset: u16) -> u32 {
            0
        }

        async fn write_registers_u32(&mut self, _offset: u16, _val: u32) {}

        async fn start_queue(
            &mut self,
            idx: u16,
            _resources: QueueResources,
            _features: &VirtioDeviceFeatures,
            _initial_state: Option<QueueState>,
        ) -> anyhow::Result<()> {
            self.started_queues.push(idx);
            Ok(())
        }

        async fn stop_queue(&mut self, idx: u16) -> Option<QueueState> {
            if self.started_queues.contains(&idx) {
                self.started_queues.retain(|&x| x != idx);
                Some(QueueState {
                    avail_index: self.stop_avail,
                    used_index: self.stop_used,
                })
            } else {
                None
            }
        }

        async fn reset(&mut self) {
            self.started_queues.clear();
        }
    }

    /// Split ring: stop_queue returns avail_index from GET_VRING_BASE,
    /// used_index from the guest-visible used ring (not from the backend).
    #[async_test]
    async fn stop_queue_split_ring_state(driver: DefaultDriver) {
        let device = SaveRestoreMockDevice::new(
            false, // split ring
            100,   // avail_index the backend will return
            999,   // used_index — should NOT appear in the frontend result
        );
        let (mut frontend, guest_memory, backend_task) =
            setup_frontend_backend_with_device(&driver, device).await;

        // Write a known used_index into the guest-visible used ring.
        // used ring layout: { flags: u16, idx: u16, ... }
        let used_addr: u64 = 0x2000;
        let used_idx_value: u16 = 77;
        guest_memory
            .write_at(used_addr + 2, &used_idx_value.to_le_bytes())
            .unwrap();

        let features = VirtioDeviceFeatures::new(); // no packed ring
        let resources =
            dummy_queue_resources(Interrupt::from_event(Event::new()), guest_memory.clone());
        frontend
            .start_queue(0, resources, &features, None)
            .await
            .expect("start_queue failed");

        let state = frontend
            .stop_queue(0)
            .await
            .expect("stop_queue should return state");
        // avail_index comes from GET_VRING_BASE reply.
        assert_eq!(state.avail_index, 100);
        // used_index comes from reading the guest used ring, not the backend.
        assert_eq!(state.used_index, used_idx_value);

        drop(frontend);
        backend_task.await;
    }

    /// Packed ring: stop_queue returns both avail and used state from
    /// GET_VRING_BASE (the used ring in guest memory is not used).
    #[async_test]
    async fn stop_queue_packed_ring_state(driver: DefaultDriver) {
        let device = SaveRestoreMockDevice::new(
            true, // packed ring
            200,  // avail_index (with wrap counter bits)
            300,  // used_index (with wrap counter bits)
        );
        let (mut frontend, guest_memory, backend_task) =
            setup_frontend_backend_with_device(&driver, device).await;

        // Features must include packed ring so the frontend knows.
        let features = VirtioDeviceFeatures::new().with_ring_packed(true);
        let resources =
            dummy_queue_resources(Interrupt::from_event(Event::new()), guest_memory.clone());
        frontend
            .start_queue(0, resources, &features, None)
            .await
            .expect("start_queue failed");

        assert!(frontend.packed_ring);

        let state = frontend
            .stop_queue(0)
            .await
            .expect("stop_queue should return state");
        // Both avail and used come from GET_VRING_BASE.
        assert_eq!(state.avail_index, 200);
        assert_eq!(state.used_index, 300);

        drop(frontend);
        backend_task.await;
    }

    /// When `use_backend_config` is false with a full config patch, the
    /// frontend serves config reads from the patch and does not negotiate
    /// VHOST_USER_PROTOCOL_F_CONFIG.
    #[async_test]
    async fn frontend_owned_config_space(driver: DefaultDriver) {
        use virtio::spec::fs as virtio_fs;
        use zerocopy::IntoBytes;

        // Build a virtio-fs config with a known tag.
        let mut config = virtio_fs::Config {
            tag: [0; virtio_fs::TAG_LEN],
            num_request_queues: 1.into(),
        };
        let tag = b"myfs";
        config.tag[..tag.len()].copy_from_slice(tag);
        let config_bytes = config.as_bytes().to_vec();

        let (mut frontend, _guest_memory, backend_task) = setup_frontend_backend_with_config(
            &driver,
            MockBackendDevice::new(),
            VhostUserConfig {
                device_id: VirtioDeviceType::FS,
                use_backend_config: false,
                queue_sizes: vec![1024; 2], // hiprio + 1 request queue
                config_patches: vec![(0, config_bytes.clone())],
            },
        )
        .await;

        // CONFIG should NOT be negotiated.
        assert!(
            !frontend
                .protocol_features
                .contains(VhostUserProtocolFeatures::CONFIG)
        );

        // Config register length should match the provided config.
        assert_eq!(
            frontend.traits().device_register_length,
            config_bytes.len() as u32
        );

        // read_registers_u32 at offset 0 should return the first 4 tag bytes.
        let val = frontend.read_registers_u32(0).await;
        assert_eq!(val, u32::from_le_bytes(*b"myfs"));

        // read_registers_u32 at the tag[4..8] region should be zero-padded.
        let val = frontend.read_registers_u32(4).await;
        assert_eq!(val, 0);

        // read_registers_u32 at the num_request_queues offset (36).
        let val = frontend.read_registers_u32(virtio_fs::TAG_LEN as u16).await;
        assert_eq!(val, 1);

        // write_registers_u32 should be a no-op (no panic, no backend call).
        frontend.write_registers_u32(0, 0xdeadbeef).await;

        drop(frontend);
        backend_task.await;
    }

    /// FS with num_request_queues=2: 3 total queues (1 hiprio + 2 request),
    /// queue_size returns 1024 for all, config reads back num_request_queues=2.
    #[async_test]
    async fn fs_multi_queue_config(driver: DefaultDriver) {
        use virtio::spec::fs as virtio_fs;
        use zerocopy::IntoBytes;

        let num_request_queues: u16 = 2;
        let queue_size: u16 = 1024;
        let total_queues = 1 + num_request_queues as usize; // hiprio + request

        let mut fs_config = virtio_fs::Config {
            tag: [0; virtio_fs::TAG_LEN],
            num_request_queues: (num_request_queues as u32).into(),
        };
        let tag = b"testfs";
        fs_config.tag[..tag.len()].copy_from_slice(tag);

        let config = VhostUserConfig {
            device_id: VirtioDeviceType::FS,
            use_backend_config: false,
            queue_sizes: vec![queue_size; total_queues],
            config_patches: vec![(0, fs_config.as_bytes().to_vec())],
        };

        // Need a mock device with enough queues (3).
        let mut mock = MockBackendDevice::new();
        mock.traits.max_queues = 4;

        let (mut frontend, _guest_memory, backend_task) =
            setup_frontend_backend_with_config(&driver, mock, config).await;

        // Verify total queue count.
        assert_eq!(frontend.traits().max_queues, total_queues as u16);

        // Verify queue_size returns 1024 for all queues.
        for i in 0..total_queues {
            assert_eq!(frontend.queue_size(i as u16), queue_size);
        }

        // Verify config space reads back num_request_queues=2.
        let val = frontend.read_registers_u32(virtio_fs::TAG_LEN as u16).await;
        assert_eq!(val, num_request_queues as u32);

        drop(frontend);
        backend_task.await;
    }

    /// BLK with num_queues=2, queue_size=512: 2 queues, queue_size returns
    /// 512 for all, config patch overrides num_queues.
    #[async_test]
    async fn blk_multi_queue_config(driver: DefaultDriver) {
        use virtio::spec::blk;

        let num_queues: u16 = 2; // MockBackendDevice supports max 2
        let queue_size: u16 = 512;
        let num_queues_offset = core::mem::offset_of!(blk::VirtioBlkConfig, num_queues) as u16;

        let config = VhostUserConfig {
            device_id: VirtioDeviceType::BLK,
            use_backend_config: true,
            queue_sizes: vec![queue_size; num_queues as usize],
            config_patches: vec![(num_queues_offset, num_queues.to_le_bytes().to_vec())],
        };

        // Backend needs config space so CONFIG protocol feature is
        // negotiated and GET_CONFIG works.
        let mut mock = MockBackendDevice::new();
        mock.traits.device_register_length = 64;

        let (mut frontend, _guest_memory, backend_task) =
            setup_frontend_backend_with_config(&driver, mock, config).await;

        // Verify queue count.
        assert_eq!(frontend.traits().max_queues, num_queues);

        // Verify queue_size returns 512 for all queues.
        for i in 0..num_queues {
            assert_eq!(frontend.queue_size(i), queue_size);
        }

        // Verify the config patch is applied: reading num_queues from
        // config space should return the patched value.
        let val = frontend.read_registers_u32(num_queues_offset).await;
        assert_eq!(val, num_queues as u32);

        drop(frontend);
        backend_task.await;
    }

    /// Generic with queue_sizes=[256, 512]: 2 queues with per-queue sizes.
    #[async_test]
    async fn generic_per_queue_sizes(driver: DefaultDriver) {
        let queue_sizes = vec![256u16, 512u16];

        let config = VhostUserConfig {
            device_id: VirtioDeviceType::BLK, // device type doesn't matter for this test
            use_backend_config: true,
            queue_sizes: queue_sizes.clone(),
            config_patches: vec![],
        };

        let (frontend, _guest_memory, backend_task) =
            setup_frontend_backend_with_config(&driver, MockBackendDevice::new(), config).await;

        // Verify queue count.
        assert_eq!(frontend.traits().max_queues, 2);

        // Verify per-queue sizes.
        assert_eq!(frontend.queue_size(0), 256);
        assert_eq!(frontend.queue_size(1), 512);

        drop(frontend);
        backend_task.await;
    }

    /// Requesting more queues than the backend supports should fail.
    #[async_test]
    async fn queue_count_exceeds_backend(driver: DefaultDriver) {
        // MockBackendDevice supports max_queues=2.
        let config = VhostUserConfig {
            device_id: VirtioDeviceType::BLK,
            use_backend_config: true,
            queue_sizes: vec![256; 4], // 4 > 2
            config_patches: vec![],
        };

        let (frontend_stream, backend_stream) = socket_pair();

        let backend_polled = PolledSocket::new(&driver, backend_stream).unwrap();
        let backend_socket = VhostUserSocket::new(backend_polled);

        let server = VhostUserDeviceServer::new(Box::new(MockBackendDevice::new()));
        let backend_task = driver.spawn("backend", async move {
            // The backend will see the connection drop when the frontend
            // rejects the queue count. Ignore the serve error.
            let _ = server.serve_connection(backend_socket).await;
        });

        let vm_driver = VmTaskDriverSource::new(SingleDriverBackend::new(driver.clone())).simple();
        let result = VhostUserFrontend::from_stream(vm_driver, frontend_stream, config).await;

        let err = result
            .err()
            .expect("should fail when queue count exceeds backend");
        let err_msg = format!("{err}");
        assert!(
            err_msg.contains("4 queues"),
            "error should mention requested count: {err_msg}"
        );

        backend_task.await;
    }
}
