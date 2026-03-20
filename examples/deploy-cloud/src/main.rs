//! Cloud GPU deployment example — search for cheap GPU offers on Vast.ai and RunPod.
//!
//! This example shows how to use rl4burn's `deploy` module to:
//!
//! 1. Define hardware requirements for a training job.
//! 2. Search multiple providers for matching GPU offers.
//! 3. Compare prices across providers.
//! 4. Build launch/status/stop requests you can execute with any HTTP client.
//!
//! ## Running
//!
//! ```bash
//! # Set API keys (optional — the example works without them for demonstration)
//! export VASTAI_API_KEY="your-vast-ai-key"
//! export RUNPOD_API_KEY="your-runpod-key"
//!
//! cargo run -p deploy-cloud
//! ```
//!
//! Without real API keys, the example demonstrates request construction and
//! shows what the provider calls would look like.

use rl4burn::deploy::{
    CloudProvider, GpuType, InstanceRequirements, RunPodProvider, VastAiProvider,
};

fn main() {
    eprintln!("=== rl4burn Cloud GPU Deployment Example ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Define what you need
    // -----------------------------------------------------------------------
    let reqs = InstanceRequirements {
        min_gpu_ram_gib: 24.0,
        num_gpus: 1,
        gpu_types: vec![GpuType::Rtx4090, GpuType::A100Pcie, GpuType::A100Sxm],
        min_ram_gib: 32.0,
        min_disk_gib: 100.0,
        docker_image: "pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel".into(),
        max_price_per_hour: 2.00,
        on_start_cmd: Some("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && source ~/.cargo/env".into()),
        ..Default::default()
    };

    eprintln!("Requirements:");
    eprintln!("  GPU VRAM:  >= {} GiB", reqs.min_gpu_ram_gib);
    eprintln!("  GPU count: {}", reqs.num_gpus);
    eprintln!(
        "  GPU types: {:?}",
        reqs.gpu_types
            .iter()
            .map(|g| g.to_string())
            .collect::<Vec<_>>()
    );
    eprintln!("  System RAM: >= {} GiB", reqs.min_ram_gib);
    eprintln!("  Disk:      >= {} GiB", reqs.min_disk_gib);
    eprintln!("  Max price: ${}/hr", reqs.max_price_per_hour);
    eprintln!("  Docker:    {}", reqs.docker_image);
    eprintln!();

    // -----------------------------------------------------------------------
    // Step 2: Set up providers
    // -----------------------------------------------------------------------
    let vastai_key = std::env::var("VASTAI_API_KEY").unwrap_or_else(|_| "demo-key".into());
    let runpod_key = std::env::var("RUNPOD_API_KEY").unwrap_or_else(|_| "demo-key".into());

    let vastai = VastAiProvider::new(&vastai_key);
    let runpod = RunPodProvider::new(&runpod_key);

    let providers: Vec<&dyn CloudProvider> = vec![&vastai, &runpod];

    // -----------------------------------------------------------------------
    // Step 3: Show what the search requests look like
    // -----------------------------------------------------------------------
    eprintln!("--- Vast.ai search query ---");
    let vastai_query = vastai.build_search_query(&reqs);
    eprintln!("{vastai_query}\n");

    let vastai_req = vastai.search_request(&reqs);
    eprintln!("Vast.ai URL: {} {}", vastai_req.method, vastai_req.url);
    eprintln!();

    eprintln!("--- RunPod search query ---");
    let runpod_query = runpod.build_search_query(&reqs);
    eprintln!("{runpod_query}\n");

    // -----------------------------------------------------------------------
    // Step 4: Attempt live search (will fail with demo keys but shows the flow)
    // -----------------------------------------------------------------------
    eprintln!("--- Searching providers ---");
    for provider in &providers {
        match provider.search_offers(&reqs) {
            Ok(offers) => {
                eprintln!(
                    "[{}] Found {} offers:",
                    provider.name(),
                    offers.len()
                );
                for (i, offer) in offers.iter().take(5).enumerate() {
                    eprintln!(
                        "  #{}: {} x{} ({:.0} GiB) — ${:.3}/hr [offer {}]",
                        i + 1,
                        offer.gpu_type,
                        offer.num_gpus,
                        offer.gpu_ram_gib,
                        offer.price_per_hour,
                        offer.offer_id,
                    );
                }
            }
            Err(e) => {
                eprintln!("[{}] {}", provider.name(), e);
                eprintln!("  (This is expected with demo API keys)");
            }
        }
        eprintln!();
    }

    // -----------------------------------------------------------------------
    // Step 5: Show launch/status/stop request construction
    // -----------------------------------------------------------------------
    eprintln!("--- Example request construction (Vast.ai) ---");

    // Construct a mock offer to show what a launch request looks like
    let mock_offer = rl4burn::deploy::GpuOffer {
        offer_id: "123456".into(),
        gpu_type: GpuType::Rtx4090,
        num_gpus: 1,
        gpu_ram_gib: 24.0,
        ram_gib: 64.0,
        disk_gib: 100.0,
        price_per_hour: 0.45,
        provider: "vast.ai",
        meta: std::collections::HashMap::new(),
    };

    let launch_req = vastai.launch_request(&mock_offer);
    eprintln!("Launch: {} {}", launch_req.method, launch_req.url);
    if let Some(body) = &launch_req.body {
        eprintln!("  Body: {body}");
    }

    let status_req = vastai.status_request("789");
    eprintln!("Status: {} {}", status_req.method, status_req.url);

    let stop_req = vastai.stop_request("789");
    eprintln!("Stop:   {} {}", stop_req.method, stop_req.url);
    eprintln!();

    eprintln!("--- Example request construction (RunPod) ---");
    let launch_mut = runpod.build_launch_mutation(&mock_offer);
    eprintln!("Launch mutation:\n{launch_mut}");

    let status_q = runpod.build_status_query("pod-abc123");
    eprintln!("Status query:\n{status_q}");

    eprintln!();
    eprintln!("Done! To run with real providers, set VASTAI_API_KEY and RUNPOD_API_KEY");
    eprintln!("and register an HTTP executor with .with_http(fn).");
}
