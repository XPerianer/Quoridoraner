# Transformer throughput optimization on NVIDIA L4

Measured 2026-11-01 on a separate NVIDIA L4, PyTorch 2.11.0+cu130. The original ResNet runtime was left untouched. These are synthetic network benchmarks, not new strength measurements or end-to-end self-play rates.

## Recommendation

Use inference microbatches of 128 with the existing 192×6 checkpoint. For the next fresh architecture experiment, try width 256, four blocks, four heads, expansion 2, inference microbatches 64, and training microbatches 128 with effective batch 1024. This retains or increases parameter capacity while reducing serial depth. Its eventual playing strength remains untested.

The smaller 128×6 model is faster still, but halves the parameters and is a less direct test of the higher-ceiling hypothesis. Changing head count or width/depth requires fresh training; microbatching can use existing weights.

## Architecture sweep

Compiled CUDA-graph bf16 inference; eager bf16 training including backward, gradient clipping and AdamW. Warmup excluded; 50 inference repeats and 20 training repeats. Training inputs/targets reside on GPU; data preparation and transfer excluded.

| Model (width × blocks / heads) | Parameters | Inference batch 1024, ms | Inference batch 2048, ms | Training batch 1024, ms |
|---|---:|---:|---:|---:|
| Transformer 192×6 / 6, original | 1,850,696 | 25.44 | 54.51 | 141.40 |
| Transformer 192×6 / 3 | 1,850,696 | 24.13 | 49.00 | 135.62 |
| Transformer 256×4 / 4 | 2,185,864 | 21.96 | 43.98 | 125.25 |
| Transformer 128×6 / 2 | 854,536 | 13.94 | 32.61 | 84.58 |
| ResNet 128×10 | 3,188,012 | 10.86 | 26.31 | 54.22 |

Fewer heads alone helped modestly. The wider, shallower model has 18% more parameters and 13% greater training-step throughput than the original transformer. All transformer candidates remain slower than the measured ResNet at full batch.

## Microbatching

Splitting inference includes output clones before reusable CUDA graph buffers are overwritten, plus concatenation. Default model batch1024: 25.56→14.58 ms with chunk128 (1.75× throughput); batch2048: 54.77→29.09 ms (1.88×). Synthetic output parity was exact for these runs. Production wrapper validation passed eager parity, graph replay and in-place checkpoint weight refresh; production timings were 14.86/30.09 ms at1024/2048.

For 256×4, chunk64 was fastest among tested sizes: 13.19 ms at1024 and26.92 ms at2048. This is around1.9–2.0× faster than the original unchunked transformer, and approaches the unchunked ResNet inference baseline. The ResNet was not microbatch-tuned in this sweep.

Training accumulates gradients, clips once and performs one AdamW update per effective batch1024:

| Model | Full batch, ms | Chunk128, ms | Chunk256, ms |
|---|---:|---:|---:|
| Original 192×6 | 141.45 | 115.61 | 104.33 |
| Candidate 256×4 | 125.38 | 91.71 | 98.49 |

The combined 256×4/chunk128 configuration provides1.54× training-step throughput relative to the original transformer. It remains slower than ResNet's54.22ms. Full-gradient relative L2 differences were0.21–0.27% under bf16. Gradient accumulation preserves the full-batch objective mathematically; floating-point reductions differ. Peak allocated GPU memory for default training decreased from2905MiB to831MiB atchunk256; candidate from2709MiB to457MiB atchunk128. These memory numbers are process-specific, not total device usage.

## Attention backend

Automatic eager dispatch already uses Flash Attention for forward/backward. Forcing flash or cuDNN did not materially improve timings. The profiler reports eager dispatch; it does not establish the compiled inference kernel choice. No evidence supports an accidental eager math-attention fallback.

## Implementation and use

Opt-in runtime settings in `training/az.py`, applied only to transformer models:

```sh
# Existing checkpoint, same architecture and effective batch:
AZ_TRANSFORMER_INFERENCE_BATCH=128 AZ_TRANSFORMER_TRAIN_BATCH=256 \
  python training/az.py rl ... --batch-size 1024

# Fresh 256×4 candidate (set architecture during pretraining):
AZ_TRANSFORMER_INFERENCE_BATCH=64 AZ_TRANSFORMER_TRAIN_BATCH=128 \
  python training/az.py pretrain ... --architecture transformer \
  --width 256 --blocks 4 --heads 4 --expansion 2 --batch-size 1024
```

Use the same environment settings for subsequent RL. Defaults remain unchanged; zero disables splitting. Environment settings are runtime controls and are not stored in checkpoints, so record them with experiment commands. The inference setting is read when each Engine architecture cache is first created.

Training uses the full batch's policy-mask denominator and value-example count, including chunks with zero supervised policy rows. Effective batch size, optimizer step count, and learning-rate scheduler cadence stay unchanged. No BatchNorm model is training-microbatched by these settings.

## Validation and limits

46 local `test_az*.py` tests passed, including sparse/zero-policy masks, uneven chunk sizes, gradient equivalence, reused inference buffers, checkpoint compatibility and compilation. L4 synthetic benchmarks include finite gradients, output parity, graph replay and weight refresh checks. Separate production sparse-policy bf16 validation is saved in `production-training-validation.json`.

No new learning curve or playing-strength test was run. Network throughput does not translate linearly into self-play or wall-clock learning improvement; CPU search, feature generation, data transfer and launch overhead remain. A fair next strength comparison needs freshly trained weights for architecture changes and matched GPU-hours, using these recorded runtime settings.

Raw measurements and logs are adjacent to this report. Benchmark entry points: `training/benchmark_transformer.py` and `training/benchmark_transformer_microbatch.py`.
