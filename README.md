# HAT SuperTransformer Reproduction and Scalability Study

This repository stores experiment logs, summaries, configs, patches, and candidate architectures from a reproduction study of MIT Han Lab's Hardware-Aware Transformers (HAT).

## Scope

This repo is not a full mirror of the original HAT source code.
It contains the experiment artifacts needed for reporting and reproducibility:

- 50K SuperTransformer from-scratch training summary
- BLEU / SacreBLEU results
- CPU / GPU latency profiling
- FLOPs / Params profiling
- Runtime compatibility patches
- Search pipeline fallback results
- Predictor-only Top-K candidate configs
- Environment and checkpoint manifests

## Key Results

| Metric | Result |
| --- | ---: |
| SuperTransformer training | 50,000 updates completed |
| Training time | 28,627.2 sec ≈ 7.95 hr |
| SacreBLEU best / last | 32.79 / 32.90 |
| V100 latency | 88.5603 ms |
| CPU latency | 193.24 ms |
| FLOPs | 1.48G |
| Params | 30.73M |
| Training peak GPU memory | about 25.7GB |

## Important Note

Large binary files such as checkpoints (`*.pt`) and datasets are intentionally not committed to this repository.
See `manifests/checkpoint_manifest.md` for checkpoint file names, sizes, and SHA256 hashes.

## Directory Structure

```text
reports/       Experiment logs and final evaluation outputs
configs/       HAT configs used in the experiments
candidates/    Predictor-only top-k candidate configs
patches/       Runtime compatibility patches
scripts/       Reproduction helper scripts
summaries/     Human-readable experiment summaries
docs/          Environment and hardware information
manifests/     Checkpoint and artifact manifests
```
