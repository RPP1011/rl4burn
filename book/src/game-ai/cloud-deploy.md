# Cloud GPU Deployment

rl4burn provides a `deploy` module with provider-agnostic abstractions for
launching training jobs on cloud GPU marketplaces. Currently supported:

- **Vast.ai** — peer-to-peer GPU marketplace with competitive pricing
- **RunPod** — managed GPU cloud with on-demand and spot pods

Enable the feature in your `Cargo.toml`:

```toml
[dependencies]
rl4burn = { git = "https://github.com/RPP1011/rl4burn", features = ["deploy"] }
```

## The `CloudProvider` Trait

All providers implement a common interface:

```rust,ignore
use rl4burn::deploy::{CloudProvider, InstanceRequirements, GpuOffer, Instance};

pub trait CloudProvider {
    fn name(&self) -> &'static str;
    fn search_offers(&self, reqs: &InstanceRequirements) -> CloudResult<Vec<GpuOffer>>;
    fn launch(&self, offer: &GpuOffer) -> CloudResult<Instance>;
    fn status(&self, instance_id: &str) -> CloudResult<Instance>;
    fn stop(&self, instance_id: &str) -> CloudResult<()>;
}
```

The trait is HTTP-client agnostic — providers produce structured `HttpRequest`
values that you execute with your preferred client (reqwest, ureq, curl, etc.).

## Specifying Requirements

```rust,ignore
use rl4burn::deploy::{InstanceRequirements, GpuType};

let reqs = InstanceRequirements {
    min_gpu_ram_gib: 24.0,
    num_gpus: 1,
    gpu_types: vec![GpuType::Rtx4090, GpuType::A100Pcie],
    min_ram_gib: 32.0,
    min_disk_gib: 100.0,
    docker_image: "pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel".into(),
    max_price_per_hour: 1.00,
    on_start_cmd: Some("apt-get update && apt-get install -y cargo".into()),
    ..Default::default()
};
```

## Vast.ai

```rust,ignore
use rl4burn::deploy::{VastAiProvider, CloudProvider};

let provider = VastAiProvider::new(std::env::var("VASTAI_API_KEY").unwrap());

// Search for offers
let offers = provider.search_offers(&reqs)?;
println!("Found {} offers, cheapest: ${}/hr",
    offers.len(), offers[0].price_per_hour);

// Launch the cheapest offer
let instance = provider.launch(&offers[0])?;
println!("Instance {} status: {}", instance.instance_id, instance.status);

// Check status
let inst = provider.status(&instance.instance_id)?;
if let Some(ssh) = &inst.ssh_connection {
    println!("Connect: {}", ssh);
}

// Cleanup
provider.stop(&instance.instance_id)?;
```

## RunPod

```rust,ignore
use rl4burn::deploy::{RunPodProvider, CloudProvider};

let provider = RunPodProvider::new(std::env::var("RUNPOD_API_KEY").unwrap());

let offers = provider.search_offers(&reqs)?;
let instance = provider.launch(&offers[0])?;
println!("Pod {} status: {}", instance.instance_id, instance.status);

provider.stop(&instance.instance_id)?;
```

## Comparing Providers

You can write provider-agnostic code by working with `&dyn CloudProvider`:

```rust,ignore
fn cheapest_offer(
    providers: &[&dyn CloudProvider],
    reqs: &InstanceRequirements,
) -> Option<(GpuOffer, &'static str)> {
    let mut best: Option<(GpuOffer, &str)> = None;
    for provider in providers {
        if let Ok(offers) = provider.search_offers(reqs) {
            for offer in offers {
                if best.as_ref().map_or(true, |(b, _)| offer.price_per_hour < b.price_per_hour) {
                    best = Some((offer, provider.name()));
                }
            }
        }
    }
    best
}
```

## HTTP Client Integration

The providers are designed to be dependency-free. Each method can either:

1. **Execute requests directly** — register an HTTP function with `.with_http(fn)`.
2. **Return request descriptions** — use `.search_request()`, `.launch_request()`, etc.
   to get `HttpRequest` structs you execute yourself.

Example with a custom HTTP function:

```rust,ignore
fn my_http(req: &rl4burn::deploy::vastai::HttpRequest) -> Result<String, String> {
    // Use reqwest, ureq, curl, etc.
    todo!()
}

let provider = VastAiProvider::new("key").with_http(my_http);
```

## Supported GPU Types

The `GpuType` enum covers common training GPUs:

| Variant | GPU |
|---------|-----|
| `Rtx3090` | NVIDIA RTX 3090 (24 GB) |
| `Rtx4090` | NVIDIA RTX 4090 (24 GB) |
| `RtxA4000` | NVIDIA RTX A4000 (16 GB) |
| `RtxA5000` | NVIDIA RTX A5000 (24 GB) |
| `RtxA6000` | NVIDIA RTX A6000 (48 GB) |
| `A100Pcie` | NVIDIA A100 PCIe (40/80 GB) |
| `A100Sxm` | NVIDIA A100 SXM (80 GB) |
| `H100Pcie` | NVIDIA H100 PCIe (80 GB) |
| `H100Sxm` | NVIDIA H100 SXM (80 GB) |
| `L40` | NVIDIA L40 (48 GB) |
| `L40s` | NVIDIA L40S (48 GB) |
