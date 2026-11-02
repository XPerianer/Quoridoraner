# Active kickstart runs (launched 2026-11-01 ~22:05 UTC)

- `qkick-resnet` (L4): ResNet 256×20, simplified inputs, `run.sh resnet 8`.
- `qkick-transformer` (L4): transformer 512×6 / 8 heads, simplified inputs, `run.sh transformer 8`.
- Teacher and anchor: h013. λ 1 → 0 over 4 training hours. 8 training hours, then final matches.
- Local backups every ≤20 minutes: `monitor.py <arm>` → `artifacts/<arm>/`, status in `monitor-<arm>.json`.
- Throughput benchmark behind the size choices: `bench/` (bench-transformers.json lacks the last 512×12 rows;
  its 1,024-row inference was 7.9k positions/s, training 648 ms with teacher).
- If a VM disappears: new L4, upload sources + `/content/teacher.pt` + `artifacts/<arm>/snapshots/state-latest.pt.gz`
  to `/content/kick-<arm>/snapshots/`, re-run `run.sh <arm> 8` (resumes from state-latest).

## VM loss and resume (2026-11-02)

Both L4s disappeared between about 00:10 and 00:40 UTC (no visible cause; ~2 h after creation). Resumed on new L4s
at about 00:48 UTC from the local `state-latest.pt.gz` backups: ResNet from iteration 13 (1.76 h; one iteration
lost), transformer from iteration 15 (1.78 h). The Colab upload API rejected the large state files (HTTP 500);
they were streamed over the `colab ssh` proxy and verified by sha256. Training hours continue from the saved
state, so λ and the 8-hour budget are unaffected; only wall-clock time is lost.
The first VMs' `rl.log` text was overwritten locally by the resumed log; per-iteration metrics are unaffected
(they live in the resumable state). The monitor now extracts to a side directory and keeps older logs.
