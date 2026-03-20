//! RunPod GPU cloud provider.
//!
//! Wraps the RunPod GraphQL API (<https://docs.runpod.io/api/graphql>)
//! for searching GPU pods, launching instances, querying status, and stopping
//! instances.
//!
//! # Authentication
//!
//! Pass your API key via [`RunPodProvider::new`]. Generate a key at
//! <https://www.runpod.io/console/user/settings>.
//!
//! # Usage
//!
//! ```rust,ignore
//! use rl4burn::deploy::{RunPodProvider, CloudProvider, InstanceRequirements, GpuType};
//!
//! let provider = RunPodProvider::new("your-api-key");
//! let reqs = InstanceRequirements {
//!     gpu_types: vec![GpuType::Rtx4090],
//!     num_gpus: 1,
//!     docker_image: "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel".into(),
//!     ..Default::default()
//! };
//! let offers = provider.search_offers(&reqs)?;
//! let instance = provider.launch(&offers[0])?;
//! ```

use super::provider::*;
use super::vastai::HttpRequest;

/// RunPod cloud GPU provider.
///
/// Interacts with the RunPod GraphQL API to manage GPU pods.
/// Like [`VastAiProvider`](super::VastAiProvider), HTTP calls are delegated
/// to a user-supplied function to keep this crate dependency-free.
#[derive(Debug, Clone)]
pub struct RunPodProvider {
    api_key: String,
    base_url: String,
    http_fn: Option<fn(&HttpRequest) -> Result<String, String>>,
}

impl RunPodProvider {
    /// Create a new RunPod provider with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.runpod.io/graphql".into(),
            http_fn: None,
        }
    }

    /// Set a custom base URL (useful for testing).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Register an HTTP executor function for making real API calls.
    pub fn with_http(mut self, f: fn(&HttpRequest) -> Result<String, String>) -> Self {
        self.http_fn = Some(f);
        self
    }

    /// Build the GraphQL query for searching available GPU types.
    pub fn build_search_query(&self, reqs: &InstanceRequirements) -> String {
        let gpu_filter = if !reqs.gpu_types.is_empty() {
            let names: Vec<&str> = reqs
                .gpu_types
                .iter()
                .map(|g| runpod_gpu_id(g))
                .collect();
            format!(
                "gpuTypeId: {{in: [{}]}}",
                names
                    .iter()
                    .map(|n| format!("\\\"{n}\\\""))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        } else {
            String::new()
        };

        let min_ram = if reqs.min_gpu_ram_gib > 0.0 {
            format!("minMemoryInGb: {}", reqs.min_gpu_ram_gib as u32)
        } else {
            String::new()
        };

        let filters: Vec<&str> = [gpu_filter.as_str(), min_ram.as_str()]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect();

        let filter_str = if filters.is_empty() {
            String::new()
        } else {
            format!("input: {{{}}}", filters.join(", "))
        };

        format!(
            r#"{{
  "query": "query {{ gpuTypes({filter_str}) {{ id displayName memoryInGb maxGpuCount securePrice communityPrice }} }}"
}}"#
        )
    }

    /// Build the GraphQL mutation for creating a pod.
    pub fn build_launch_mutation(&self, offer: &GpuOffer) -> String {
        let image = offer
            .meta
            .get("docker_image")
            .cloned()
            .unwrap_or_else(|| "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel".into());
        let disk = offer.disk_gib as u32;
        let gpu_count = offer.num_gpus;
        let gpu_type_id = offer
            .meta
            .get("gpu_type_id")
            .cloned()
            .unwrap_or_else(|| offer.offer_id.clone());

        format!(
            r#"{{
  "query": "mutation {{ podFindAndDeployOnDemand(input: {{ name: \\\"rl4burn-training\\\", imageName: \\\"{image}\\\", gpuTypeId: \\\"{gpu_type_id}\\\", gpuCount: {gpu_count}, volumeInGb: {disk}, containerDiskInGb: 20 }}) {{ id imageName desiredStatus }} }}"
}}"#
        )
    }

    /// Build the GraphQL query for pod status.
    pub fn build_status_query(&self, pod_id: &str) -> String {
        format!(
            r#"{{
  "query": "query {{ pod(input: {{ podId: \\\"{pod_id}\\\" }}) {{ id desiredStatus runtime {{ uptimeInSeconds ports {{ ip isIpPublic privatePort publicPort }} }} }} }}"
}}"#
        )
    }

    /// Build the GraphQL mutation for stopping a pod.
    pub fn build_stop_mutation(&self, pod_id: &str) -> String {
        format!(
            r#"{{
  "query": "mutation {{ podTerminate(input: {{ podId: \\\"{pod_id}\\\" }}) }}"
}}"#
        )
    }

    fn graphql_request(&self, body: String) -> HttpRequest {
        HttpRequest {
            method: "POST",
            url: format!("{}?api_key={}", self.base_url, self.api_key),
            headers: vec![("Content-Type", "application/json".into())],
            body: Some(body),
        }
    }

    fn exec(&self, req: &HttpRequest) -> CloudResult<String> {
        match &self.http_fn {
            Some(f) => f(req).map_err(|e| CloudError::Network(e)),
            None => Err(CloudError::Network(format!(
                "No HTTP executor registered. Request: {} {}",
                req.method, req.url
            ))),
        }
    }
}

impl CloudProvider for RunPodProvider {
    fn name(&self) -> &'static str {
        "runpod"
    }

    fn search_offers(&self, reqs: &InstanceRequirements) -> CloudResult<Vec<GpuOffer>> {
        let query = self.build_search_query(reqs);
        let req = self.graphql_request(query);
        let body = self.exec(&req)?;

        parse_runpod_gpu_types(&body, reqs)
    }

    fn launch(&self, offer: &GpuOffer) -> CloudResult<Instance> {
        let mutation = self.build_launch_mutation(offer);
        let req = self.graphql_request(mutation);
        let body = self.exec(&req)?;

        parse_runpod_pod(&body)
    }

    fn status(&self, instance_id: &str) -> CloudResult<Instance> {
        let query = self.build_status_query(instance_id);
        let req = self.graphql_request(query);
        let body = self.exec(&req)?;

        parse_runpod_pod(&body)
    }

    fn stop(&self, instance_id: &str) -> CloudResult<()> {
        let mutation = self.build_stop_mutation(instance_id);
        let req = self.graphql_request(mutation);
        self.exec(&req)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn runpod_gpu_id(gpu: &GpuType) -> &'static str {
    match gpu {
        GpuType::RtxA4000 => "NVIDIA RTX A4000",
        GpuType::RtxA5000 => "NVIDIA RTX A5000",
        GpuType::RtxA6000 => "NVIDIA RTX A6000",
        GpuType::Rtx3090 => "NVIDIA GeForce RTX 3090",
        GpuType::Rtx4090 => "NVIDIA GeForce RTX 4090",
        GpuType::A100Sxm => "NVIDIA A100 80GB SXM",
        GpuType::A100Pcie => "NVIDIA A100 80GB PCIe",
        GpuType::H100Sxm => "NVIDIA H100 80GB SXM",
        GpuType::H100Pcie => "NVIDIA H100 80GB PCIe",
        GpuType::L40 => "NVIDIA L40",
        GpuType::L40s => "NVIDIA L40S",
        GpuType::Other => "OTHER",
    }
}

fn parse_gpu_type_from_runpod(name: &str) -> GpuType {
    let lower = name.to_lowercase();
    if lower.contains("a4000") {
        GpuType::RtxA4000
    } else if lower.contains("a5000") {
        GpuType::RtxA5000
    } else if lower.contains("a6000") {
        GpuType::RtxA6000
    } else if lower.contains("3090") {
        GpuType::Rtx3090
    } else if lower.contains("4090") {
        GpuType::Rtx4090
    } else if lower.contains("a100") && lower.contains("sxm") {
        GpuType::A100Sxm
    } else if lower.contains("a100") {
        GpuType::A100Pcie
    } else if lower.contains("h100") && lower.contains("sxm") {
        GpuType::H100Sxm
    } else if lower.contains("h100") {
        GpuType::H100Pcie
    } else if lower.contains("l40s") {
        GpuType::L40s
    } else if lower.contains("l40") {
        GpuType::L40
    } else {
        GpuType::Other
    }
}

/// Minimal JSON field extraction (same approach as vastai module).
fn extract_string_field(json: &str, field: &str) -> Option<String> {
    let pattern = format!("\"{}\"", field);
    let pos = json.find(&pattern)?;
    let rest = &json[pos + pattern.len()..];
    let rest = rest.trim_start().strip_prefix(':')?;
    let rest = rest.trim_start();
    if rest.starts_with('"') {
        let rest = &rest[1..];
        let end = rest.find('"')?;
        Some(rest[..end].to_string())
    } else {
        let end = rest.find(|c: char| c == ',' || c == '}' || c == ' ' || c == '\n')?;
        Some(rest[..end].to_string())
    }
}

fn extract_number_field(json: &str, field: &str) -> Option<f64> {
    extract_string_field(json, field)?.parse().ok()
}

fn parse_runpod_gpu_types(json: &str, reqs: &InstanceRequirements) -> CloudResult<Vec<GpuOffer>> {
    let mut offers = Vec::new();

    for chunk in json.split("\"id\"").skip(1) {
        let id = (|| -> Option<String> {
            let rest = chunk.trim_start().strip_prefix(':')?;
            let rest = rest.trim_start();
            if rest.starts_with('"') {
                let rest = &rest[1..];
                rest.find('"').map(|end| rest[..end].to_string())
            } else {
                None
            }
        })()
        .unwrap_or_default();

        let display_name = extract_string_field(chunk, "displayName").unwrap_or_default();
        let memory = extract_number_field(chunk, "memoryInGb").unwrap_or(0.0);
        let max_gpus = extract_number_field(chunk, "maxGpuCount").unwrap_or(1.0) as u32;
        let secure_price = extract_number_field(chunk, "securePrice").unwrap_or(0.0);
        let community_price = extract_number_field(chunk, "communityPrice").unwrap_or(0.0);

        // Use community price if available, else secure
        let price = if community_price > 0.0 {
            community_price
        } else {
            secure_price
        };

        if id.is_empty() || memory < reqs.min_gpu_ram_gib {
            continue;
        }

        if reqs.max_price_per_hour > 0.0 && price > reqs.max_price_per_hour {
            continue;
        }

        let gpu_type = parse_gpu_type_from_runpod(&display_name);
        let num_gpus = reqs.num_gpus.min(max_gpus);

        let mut meta = std::collections::HashMap::new();
        meta.insert("gpu_type_id".into(), id.clone());
        meta.insert("display_name".into(), display_name);

        offers.push(GpuOffer {
            offer_id: id,
            gpu_type,
            num_gpus,
            gpu_ram_gib: memory,
            ram_gib: 0.0, // RunPod doesn't expose system RAM in gpuTypes query
            disk_gib: reqs.min_disk_gib,
            price_per_hour: price * num_gpus as f64,
            provider: "runpod",
            meta,
        });
    }

    if offers.is_empty() {
        return Err(CloudError::NoOffers);
    }

    // Sort by price
    offers.sort_by(|a, b| a.price_per_hour.partial_cmp(&b.price_per_hour).unwrap());
    Ok(offers)
}

fn parse_runpod_pod(json: &str) -> CloudResult<Instance> {
    // Look for pod data in the response
    let id = extract_string_field(json, "id").unwrap_or_default();
    let status_str = extract_string_field(json, "desiredStatus").unwrap_or_default();

    let status = match status_str.as_str() {
        "RUNNING" => InstanceStatus::Running,
        "CREATED" | "BUILDING" => InstanceStatus::Creating,
        "STOPPED" | "EXITED" | "TERMINATED" => InstanceStatus::Stopped,
        "ERROR" => InstanceStatus::Error,
        _ => InstanceStatus::Unknown,
    };

    // Try to extract SSH connection info from runtime.ports
    let ip = extract_string_field(json, "ip");
    let public_port = extract_number_field(json, "publicPort").map(|p| p as u16);

    let ssh_connection = ip.as_ref().map(|addr| {
        let port = public_port.unwrap_or(22);
        format!("ssh -p {port} root@{addr}")
    });

    Ok(Instance {
        instance_id: id,
        status,
        ssh_connection,
        ip_address: ip,
        ssh_port: public_port,
        provider: "runpod",
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_name() {
        let p = RunPodProvider::new("test-key");
        assert_eq!(p.name(), "runpod");
    }

    #[test]
    fn build_search_query_basic() {
        let p = RunPodProvider::new("key");
        let reqs = InstanceRequirements::default();
        let q = p.build_search_query(&reqs);
        assert!(q.contains("gpuTypes"));
        assert!(q.contains("memoryInGb"));
    }

    #[test]
    fn build_search_query_with_gpu_filter() {
        let p = RunPodProvider::new("key");
        let reqs = InstanceRequirements {
            gpu_types: vec![GpuType::Rtx4090],
            ..Default::default()
        };
        let q = p.build_search_query(&reqs);
        assert!(q.contains("RTX 4090"));
    }

    #[test]
    fn build_launch_mutation_uses_offer() {
        let p = RunPodProvider::new("key");
        let mut meta = std::collections::HashMap::new();
        meta.insert("gpu_type_id".into(), "NVIDIA GeForce RTX 4090".into());
        let offer = GpuOffer {
            offer_id: "NVIDIA GeForce RTX 4090".into(),
            gpu_type: GpuType::Rtx4090,
            num_gpus: 1,
            gpu_ram_gib: 24.0,
            ram_gib: 0.0,
            disk_gib: 50.0,
            price_per_hour: 0.44,
            provider: "runpod",
            meta,
        };
        let m = p.build_launch_mutation(&offer);
        assert!(m.contains("podFindAndDeployOnDemand"));
        assert!(m.contains("RTX 4090"));
    }

    #[test]
    fn status_query_includes_pod_id() {
        let p = RunPodProvider::new("key");
        let q = p.build_status_query("abc123");
        assert!(q.contains("abc123"));
    }

    #[test]
    fn stop_mutation_includes_pod_id() {
        let p = RunPodProvider::new("key");
        let m = p.build_stop_mutation("abc123");
        assert!(m.contains("podTerminate"));
        assert!(m.contains("abc123"));
    }

    #[test]
    fn graphql_request_includes_api_key() {
        let p = RunPodProvider::new("secret-key");
        let req = p.graphql_request("{}".into());
        assert!(req.url.contains("secret-key"));
        assert_eq!(req.method, "POST");
    }

    #[test]
    fn custom_base_url() {
        let p = RunPodProvider::new("key").with_base_url("http://localhost:9090/graphql");
        let req = p.graphql_request("{}".into());
        assert!(req.url.starts_with("http://localhost:9090/graphql"));
    }

    #[test]
    fn parse_gpu_types_from_names() {
        assert_eq!(
            parse_gpu_type_from_runpod("NVIDIA GeForce RTX 4090"),
            GpuType::Rtx4090
        );
        assert_eq!(
            parse_gpu_type_from_runpod("NVIDIA A100 80GB SXM"),
            GpuType::A100Sxm
        );
        assert_eq!(
            parse_gpu_type_from_runpod("NVIDIA H100 80GB PCIe"),
            GpuType::H100Pcie
        );
    }
}
