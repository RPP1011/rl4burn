//! Vast.ai GPU marketplace provider.
//!
//! Wraps the Vast.ai REST API (<https://vast.ai/docs/rest/introduction>)
//! for searching offers, launching instances, querying status, and stopping
//! instances.
//!
//! # Authentication
//!
//! Pass your API key via [`VastAiProvider::new`]. You can find your key at
//! <https://cloud.vast.ai/account/>.
//!
//! # Usage
//!
//! ```rust,ignore
//! use rl4burn::deploy::{VastAiProvider, CloudProvider, InstanceRequirements};
//!
//! let provider = VastAiProvider::new("your-api-key");
//! let reqs = InstanceRequirements::default();
//! let offers = provider.search_offers(&reqs)?;
//! println!("Cheapest: {} at ${}/hr", offers[0].gpu_type, offers[0].price_per_hour);
//! ```

use super::provider::*;

/// Vast.ai cloud GPU provider.
///
/// Interacts with the Vast.ai REST API to manage GPU instances.
/// All methods construct the appropriate HTTP request parameters but delegate
/// actual HTTP calls to the user's HTTP client through the [`VastAiProvider::with_http`]
/// hook, keeping this crate dependency-free.
#[derive(Debug, Clone)]
pub struct VastAiProvider {
    api_key: String,
    base_url: String,
    /// Optional HTTP executor. When `None`, methods return structured request
    /// descriptions that the caller can execute with any HTTP client.
    http_fn: Option<fn(&HttpRequest) -> Result<String, String>>,
}

/// An HTTP request that the provider needs executed.
///
/// This is returned by provider methods so callers can use their preferred
/// HTTP client (reqwest, ureq, curl, etc.) without this crate depending on one.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    /// HTTP method (GET, POST, PUT, DELETE).
    pub method: &'static str,
    /// Full URL.
    pub url: String,
    /// Request headers.
    pub headers: Vec<(&'static str, String)>,
    /// Optional JSON body.
    pub body: Option<String>,
}

impl VastAiProvider {
    /// Create a new Vast.ai provider with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://console.vast.ai/api/v0".into(),
            http_fn: None,
        }
    }

    /// Set a custom base URL (useful for testing).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Register an HTTP executor function for making real API calls.
    ///
    /// Without this, the provider methods will return `CloudError::Network`
    /// with a message containing the request details.
    pub fn with_http(mut self, f: fn(&HttpRequest) -> Result<String, String>) -> Self {
        self.http_fn = Some(f);
        self
    }

    /// Build the search query JSON for the Vast.ai offers endpoint.
    pub fn build_search_query(&self, reqs: &InstanceRequirements) -> String {
        let mut filters = Vec::new();

        filters.push(format!(
            "\"gpu_ram\": {{\"gte\": {:.1}}}",
            reqs.min_gpu_ram_gib
        ));
        filters.push(format!("\"num_gpus\": {{\"gte\": {}}}", reqs.num_gpus));

        if reqs.min_ram_gib > 0.0 {
            filters.push(format!("\"cpu_ram\": {{\"gte\": {:.1}}}", reqs.min_ram_gib));
        }
        if reqs.min_disk_gib > 0.0 {
            filters.push(format!(
                "\"disk_space\": {{\"gte\": {:.1}}}",
                reqs.min_disk_gib
            ));
        }
        if reqs.max_price_per_hour > 0.0 {
            filters.push(format!(
                "\"dph_total\": {{\"lte\": {:.4}}}",
                reqs.max_price_per_hour
            ));
        }

        if !reqs.gpu_types.is_empty() {
            let names: Vec<String> = reqs
                .gpu_types
                .iter()
                .map(|g| format!("\"{}\"", vastai_gpu_name(g)))
                .collect();
            filters.push(format!("\"gpu_name\": {{\"in\": [{}]}}", names.join(", ")));
        }

        format!("{{{}}}", filters.join(", "))
    }

    /// Build the HTTP request for searching offers.
    pub fn search_request(&self, reqs: &InstanceRequirements) -> HttpRequest {
        let query = self.build_search_query(reqs);
        HttpRequest {
            method: "GET",
            url: format!(
                "{}/bundles?q={}&order=dph_total&type=on-demand",
                self.base_url, query
            ),
            headers: vec![("Authorization", format!("Bearer {}", self.api_key))],
            body: None,
        }
    }

    /// Build the HTTP request for launching an instance from an offer.
    pub fn launch_request(&self, offer: &GpuOffer) -> HttpRequest {
        let image = offer
            .meta
            .get("docker_image")
            .cloned()
            .unwrap_or_else(|| "pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel".into());
        let on_start = offer
            .meta
            .get("on_start_cmd")
            .cloned()
            .unwrap_or_default();

        let body = format!(
            r#"{{"client_id": "me", "image": "{image}", "disk": {:.1}, "onstart": "{on_start}"}}"#,
            offer.disk_gib,
        );

        HttpRequest {
            method: "PUT",
            url: format!(
                "{}/asks/{}/",
                self.base_url, offer.offer_id
            ),
            headers: vec![
                ("Authorization", format!("Bearer {}", self.api_key)),
                ("Content-Type", "application/json".into()),
            ],
            body: Some(body),
        }
    }

    /// Build the HTTP request for querying instance status.
    pub fn status_request(&self, instance_id: &str) -> HttpRequest {
        HttpRequest {
            method: "GET",
            url: format!("{}/instances/{}/", self.base_url, instance_id),
            headers: vec![("Authorization", format!("Bearer {}", self.api_key))],
            body: None,
        }
    }

    /// Build the HTTP request for stopping an instance.
    pub fn stop_request(&self, instance_id: &str) -> HttpRequest {
        HttpRequest {
            method: "DELETE",
            url: format!("{}/instances/{}/", self.base_url, instance_id),
            headers: vec![("Authorization", format!("Bearer {}", self.api_key))],
            body: None,
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

impl CloudProvider for VastAiProvider {
    fn name(&self) -> &'static str {
        "vast.ai"
    }

    fn search_offers(&self, reqs: &InstanceRequirements) -> CloudResult<Vec<GpuOffer>> {
        let req = self.search_request(reqs);
        let body = self.exec(&req)?;

        // Minimal JSON parsing — in production, users would use serde_json.
        // We parse the "offers" array from the Vast.ai response format.
        parse_vastai_offers(&body)
    }

    fn launch(&self, offer: &GpuOffer) -> CloudResult<Instance> {
        let req = self.launch_request(offer);
        let body = self.exec(&req)?;

        parse_vastai_instance(&body, "vast.ai")
    }

    fn status(&self, instance_id: &str) -> CloudResult<Instance> {
        let req = self.status_request(instance_id);
        let body = self.exec(&req)?;

        parse_vastai_instance(&body, "vast.ai")
    }

    fn stop(&self, instance_id: &str) -> CloudResult<()> {
        let req = self.stop_request(instance_id);
        self.exec(&req)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn vastai_gpu_name(gpu: &GpuType) -> &'static str {
    match gpu {
        GpuType::RtxA4000 => "RTX A4000",
        GpuType::RtxA5000 => "RTX A5000",
        GpuType::RtxA6000 => "RTX A6000",
        GpuType::Rtx3090 => "RTX 3090",
        GpuType::Rtx4090 => "RTX 4090",
        GpuType::A100Sxm => "A100 SXM",
        GpuType::A100Pcie => "A100 PCIe",
        GpuType::H100Sxm => "H100 SXM",
        GpuType::H100Pcie => "H100 PCIe",
        GpuType::L40 => "L40",
        GpuType::L40s => "L40S",
        GpuType::Other => "Other",
    }
}

/// Parse Vast.ai JSON offer list. This is a minimal parser — for production
/// use, bring in `serde_json`.
fn parse_vastai_offers(json: &str) -> CloudResult<Vec<GpuOffer>> {
    // Vast.ai returns {"offers": [{...}, ...]}
    // We do basic string extraction for the key fields.
    let mut offers = Vec::new();

    // Split on offer boundaries (each offer is a JSON object in the array)
    for chunk in json.split("\"id\"").skip(1) {
        let id = extract_number(chunk).unwrap_or_default();
        let gpu_name = extract_string_field(chunk, "gpu_name").unwrap_or_default();
        let num_gpus = extract_number_field(chunk, "num_gpus").unwrap_or(1.0) as u32;
        let gpu_ram = extract_number_field(chunk, "gpu_ram").unwrap_or(0.0);
        let cpu_ram = extract_number_field(chunk, "cpu_ram").unwrap_or(0.0);
        let disk = extract_number_field(chunk, "disk_space").unwrap_or(0.0);
        let price = extract_number_field(chunk, "dph_total").unwrap_or(0.0);

        if id.is_empty() {
            continue;
        }

        offers.push(GpuOffer {
            offer_id: id,
            gpu_type: parse_gpu_type(&gpu_name),
            num_gpus,
            gpu_ram_gib: gpu_ram,
            ram_gib: cpu_ram,
            disk_gib: disk,
            price_per_hour: price,
            provider: "vast.ai",
            meta: std::collections::HashMap::new(),
        });
    }

    if offers.is_empty() {
        return Err(CloudError::NoOffers);
    }

    Ok(offers)
}

fn parse_vastai_instance(json: &str, provider: &'static str) -> CloudResult<Instance> {
    let id = extract_string_field(json, "id")
        .or_else(|| extract_string_field(json, "new_contract"))
        .unwrap_or_default();
    let status_str = extract_string_field(json, "actual_status").unwrap_or_default();
    let ssh_host = extract_string_field(json, "ssh_host");
    let ssh_port = extract_number_field(json, "ssh_port").map(|p| p as u16);
    let ip = extract_string_field(json, "public_ipaddr");

    let status = match status_str.as_str() {
        "running" => InstanceStatus::Running,
        "loading" | "creating" => InstanceStatus::Creating,
        "exited" | "stopped" => InstanceStatus::Stopped,
        "error" => InstanceStatus::Error,
        _ => InstanceStatus::Unknown,
    };

    let ssh_connection = ssh_host.as_ref().map(|host| {
        let port = ssh_port.unwrap_or(22);
        format!("ssh -p {port} root@{host}")
    });

    Ok(Instance {
        instance_id: id,
        status,
        ssh_connection,
        ip_address: ip,
        ssh_port,
        provider,
    })
}

fn parse_gpu_type(name: &str) -> GpuType {
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

// Minimal JSON field extraction (no serde dependency)
fn extract_string_field(json: &str, field: &str) -> Option<String> {
    let pattern = format!("\"{}\"", field);
    let pos = json.find(&pattern)?;
    let rest = &json[pos + pattern.len()..];
    // Skip : and whitespace
    let rest = rest.trim_start().strip_prefix(':')?;
    let rest = rest.trim_start();
    if rest.starts_with('"') {
        let rest = &rest[1..];
        let end = rest.find('"')?;
        Some(rest[..end].to_string())
    } else {
        // Number value — return as string
        let end = rest.find(|c: char| c == ',' || c == '}' || c == ' ')?;
        Some(rest[..end].to_string())
    }
}

fn extract_number_field(json: &str, field: &str) -> Option<f64> {
    let s = extract_string_field(json, field)?;
    s.parse().ok()
}

fn extract_number(s: &str) -> Option<String> {
    // Extract the first number-like sequence after a colon
    let rest = s.trim_start().strip_prefix(':')?;
    let rest = rest.trim_start().strip_prefix('"').unwrap_or(rest.trim_start());
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '.')?;
    if end == 0 {
        return None;
    }
    Some(rest[..end].to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_name() {
        let p = VastAiProvider::new("test-key");
        assert_eq!(p.name(), "vast.ai");
    }

    #[test]
    fn build_search_query_basic() {
        let p = VastAiProvider::new("key");
        let reqs = InstanceRequirements {
            min_gpu_ram_gib: 24.0,
            num_gpus: 1,
            ..Default::default()
        };
        let q = p.build_search_query(&reqs);
        assert!(q.contains("\"gpu_ram\""));
        assert!(q.contains("\"num_gpus\""));
    }

    #[test]
    fn build_search_query_with_gpu_filter() {
        let p = VastAiProvider::new("key");
        let reqs = InstanceRequirements {
            gpu_types: vec![GpuType::Rtx4090, GpuType::A100Sxm],
            ..Default::default()
        };
        let q = p.build_search_query(&reqs);
        assert!(q.contains("RTX 4090"));
        assert!(q.contains("A100 SXM"));
    }

    #[test]
    fn build_search_query_with_price_limit() {
        let p = VastAiProvider::new("key");
        let reqs = InstanceRequirements {
            max_price_per_hour: 1.50,
            ..Default::default()
        };
        let q = p.build_search_query(&reqs);
        assert!(q.contains("\"dph_total\""));
        assert!(q.contains("1.5"));
    }

    #[test]
    fn search_request_url() {
        let p = VastAiProvider::new("my-key");
        let reqs = InstanceRequirements::default();
        let req = p.search_request(&reqs);
        assert_eq!(req.method, "GET");
        assert!(req.url.contains("/bundles"));
        assert!(req.headers[0].1.contains("my-key"));
    }

    #[test]
    fn launch_request_uses_offer_id() {
        let p = VastAiProvider::new("key");
        let offer = GpuOffer {
            offer_id: "12345".into(),
            gpu_type: GpuType::Rtx4090,
            num_gpus: 1,
            gpu_ram_gib: 24.0,
            ram_gib: 32.0,
            disk_gib: 50.0,
            price_per_hour: 0.50,
            provider: "vast.ai",
            meta: std::collections::HashMap::new(),
        };
        let req = p.launch_request(&offer);
        assert_eq!(req.method, "PUT");
        assert!(req.url.contains("12345"));
    }

    #[test]
    fn parse_gpu_types() {
        assert_eq!(parse_gpu_type("RTX 4090"), GpuType::Rtx4090);
        assert_eq!(parse_gpu_type("A100 SXM 80GB"), GpuType::A100Sxm);
        assert_eq!(parse_gpu_type("A100 PCIe"), GpuType::A100Pcie);
        assert_eq!(parse_gpu_type("H100 SXM"), GpuType::H100Sxm);
        assert_eq!(parse_gpu_type("L40S"), GpuType::L40s);
        assert_eq!(parse_gpu_type("Unknown GPU"), GpuType::Other);
    }

    #[test]
    fn instance_requirements_default() {
        let reqs = InstanceRequirements::default();
        assert_eq!(reqs.min_gpu_ram_gib, 24.0);
        assert_eq!(reqs.num_gpus, 1);
        assert!(reqs.gpu_types.is_empty());
    }

    #[test]
    fn instance_status_display() {
        assert_eq!(InstanceStatus::Running.to_string(), "running");
        assert_eq!(InstanceStatus::Creating.to_string(), "creating");
        assert_eq!(InstanceStatus::Stopped.to_string(), "stopped");
    }

    #[test]
    fn custom_base_url() {
        let p = VastAiProvider::new("key").with_base_url("http://localhost:8080");
        let req = p.status_request("123");
        assert!(req.url.starts_with("http://localhost:8080"));
    }
}
