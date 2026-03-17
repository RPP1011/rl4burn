# CSPL (Curriculum Self-Play Learning)

JueWu's 3-phase training pipeline for scaling to many heroes/unit types.

## The problem

Training a single policy that handles 40+ heroes in all combinations doesn't converge (480+ hours without success).

## The solution: three phases

| Phase | What | Duration |
|-------|------|----------|
| **1. Specialists** | Train small models on fixed team compositions | ~72h |
| **2. Distillation** | Merge all specialists into one big model | ~48h |
| **3. Generalization** | Continue RL with random compositions | ~144h |

## API

```rust,ignore
use rl4burn::{CsplPipeline, CsplConfig, CsplPhase};

let mut pipeline = CsplPipeline::new(CsplConfig {
    phase1_steps: 100_000,
    phase2_steps: 50_000,
    phase3_steps: 200_000,
    n_specialists: 10,
});

loop {
    let phase_changed = pipeline.step();

    match pipeline.current_phase() {
        CsplPhase::SpecialistTraining => { /* train specialists via self-play */ }
        CsplPhase::Distillation => { /* distill into student */ }
        CsplPhase::Generalization => { /* continue RL with random compositions */ }
    }

    if pipeline.is_complete() { break; }
}
```
