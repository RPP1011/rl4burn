//! Cloud GPU provider trait and shared types.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// GPU type
// ---------------------------------------------------------------------------

/// Common GPU models available on cloud marketplaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuType {
    RtxA4000,
    RtxA5000,
    RtxA6000,
    Rtx3090,
    Rtx4090,
    A100Sxm,
    A100Pcie,
    H100Sxm,
    H100Pcie,
    L40,
    L40s,
    Other,
}

impl fmt::Display for GpuType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::RtxA4000 => "RTX A4000",
            Self::RtxA5000 => "RTX A5000",
            Self::RtxA6000 => "RTX A6000",
            Self::Rtx3090 => "RTX 3090",
            Self::Rtx4090 => "RTX 4090",
            Self::A100Sxm => "A100 SXM",
            Self::A100Pcie => "A100 PCIe",
            Self::H100Sxm => "H100 SXM",
            Self::H100Pcie => "H100 PCIe",
            Self::L40 => "L40",
            Self::L40s => "L40S",
            Self::Other => "Other",
        };
        write!(f, "{name}")
    }
}

// ---------------------------------------------------------------------------
// Instance requirements
// ---------------------------------------------------------------------------

/// Requirements for a GPU instance to run a training job.
#[derive(Debug, Clone)]
pub struct InstanceRequirements {
    /// Minimum GPU VRAM in GiB.
    pub min_gpu_ram_gib: f64,
    /// Minimum number of GPUs.
    pub num_gpus: u32,
    /// Preferred GPU types (empty = any).
    pub gpu_types: Vec<GpuType>,
    /// Minimum system RAM in GiB.
    pub min_ram_gib: f64,
    /// Minimum disk space in GiB.
    pub min_disk_gib: f64,
    /// Docker image to launch.
    pub docker_image: String,
    /// Maximum price per hour in USD (0.0 = no limit).
    pub max_price_per_hour: f64,
    /// On-start command to execute when the instance boots.
    pub on_start_cmd: Option<String>,
    /// Extra provider-specific key-value options.
    pub extra: HashMap<String, String>,
}

impl Default for InstanceRequirements {
    fn default() -> Self {
        Self {
            min_gpu_ram_gib: 24.0,
            num_gpus: 1,
            gpu_types: Vec::new(),
            min_ram_gib: 32.0,
            min_disk_gib: 50.0,
            docker_image: "pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel".into(),
            max_price_per_hour: 0.0,
            on_start_cmd: None,
            extra: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Offer / instance
// ---------------------------------------------------------------------------

/// An available GPU offer from a provider.
#[derive(Debug, Clone)]
pub struct GpuOffer {
    /// Provider-specific offer ID.
    pub offer_id: String,
    /// GPU model.
    pub gpu_type: GpuType,
    /// Number of GPUs in this offer.
    pub num_gpus: u32,
    /// GPU VRAM per GPU in GiB.
    pub gpu_ram_gib: f64,
    /// System RAM in GiB.
    pub ram_gib: f64,
    /// Disk space in GiB.
    pub disk_gib: f64,
    /// Price per hour in USD.
    pub price_per_hour: f64,
    /// Provider name (e.g., "vast.ai", "runpod").
    pub provider: &'static str,
    /// Provider-specific metadata.
    pub meta: HashMap<String, String>,
}

/// Status of a running instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstanceStatus {
    /// Instance is being created/provisioned.
    Creating,
    /// Instance is running and ready.
    Running,
    /// Instance is being stopped.
    Stopping,
    /// Instance has stopped.
    Stopped,
    /// Instance encountered an error.
    Error,
    /// Status is unknown or not recognized.
    Unknown,
}

impl fmt::Display for InstanceStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Creating => "creating",
            Self::Running => "running",
            Self::Stopping => "stopping",
            Self::Stopped => "stopped",
            Self::Error => "error",
            Self::Unknown => "unknown",
        };
        write!(f, "{s}")
    }
}

/// A running cloud GPU instance.
#[derive(Debug, Clone)]
pub struct Instance {
    /// Provider-specific instance ID.
    pub instance_id: String,
    /// Current status.
    pub status: InstanceStatus,
    /// SSH connection string (e.g., "ssh -p 12345 root@host").
    pub ssh_connection: Option<String>,
    /// Public IP address.
    pub ip_address: Option<String>,
    /// SSH port.
    pub ssh_port: Option<u16>,
    /// Provider name.
    pub provider: &'static str,
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors from cloud provider operations.
#[derive(Debug, thiserror::Error)]
pub enum CloudError {
    /// The provider API returned an error.
    #[error("API error from {provider}: {message}")]
    Api {
        provider: &'static str,
        message: String,
    },
    /// Network or connectivity issue.
    #[error("Network error: {0}")]
    Network(String),
    /// Authentication failure (bad API key, expired token).
    #[error("Auth error for {provider}: {message}")]
    Auth {
        provider: &'static str,
        message: String,
    },
    /// No offers matched the given requirements.
    #[error("No matching offers found for the given requirements")]
    NoOffers,
    /// The requested instance was not found.
    #[error("Instance {instance_id} not found on {provider}")]
    NotFound {
        provider: &'static str,
        instance_id: String,
    },
}

/// Result alias for cloud provider operations.
pub type CloudResult<T> = std::result::Result<T, CloudError>;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Trait for cloud GPU providers.
///
/// Implementations wrap the provider's REST API and handle authentication,
/// offer search, instance creation, status polling, and teardown.
///
/// # Example
///
/// ```rust,ignore
/// use rl4burn::deploy::{CloudProvider, InstanceRequirements, VastAiProvider};
///
/// let provider = VastAiProvider::new("your-api-key");
/// let reqs = InstanceRequirements {
///     min_gpu_ram_gib: 24.0,
///     num_gpus: 1,
///     docker_image: "pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel".into(),
///     ..Default::default()
/// };
///
/// let offers = provider.search_offers(&reqs)?;
/// let instance = provider.launch(&offers[0])?;
/// let status = provider.status(&instance.instance_id)?;
/// provider.stop(&instance.instance_id)?;
/// ```
pub trait CloudProvider {
    /// Human-readable provider name (e.g., "vast.ai").
    fn name(&self) -> &'static str;

    /// Search for GPU offers matching the given requirements.
    fn search_offers(&self, reqs: &InstanceRequirements) -> CloudResult<Vec<GpuOffer>>;

    /// Launch an instance from an offer.
    fn launch(&self, offer: &GpuOffer) -> CloudResult<Instance>;

    /// Query the current status of an instance.
    fn status(&self, instance_id: &str) -> CloudResult<Instance>;

    /// Stop (destroy) a running instance.
    fn stop(&self, instance_id: &str) -> CloudResult<()>;
}
