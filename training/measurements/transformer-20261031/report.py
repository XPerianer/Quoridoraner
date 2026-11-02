"""Generate the experiment's Markdown report from downloaded artifacts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    bench = json.loads((ROOT / 'benchmark.json').read_text())
    metrics = json.loads((ROOT / 'transformer-metrics.json').read_text())
    pre = json.loads((ROOT / 'transformer-pretrain.json').read_text())
    resnets = json.loads((ROOT / 'resnet-checkpoints.json').read_text())
    reference = next(r for r in resnets if 'h001' in r['file'])
    elapsed = sum(r['seconds'] for r in metrics)
    lines = ['# One-hour transformer versus ResNet', '',
             '**Recommendation: keep the 128×10 ResNet for the current L4 training pipeline.** '
             'The tested transformer scored 28.6% against the one-hour ResNet (400 games; paired 95% interval 24.8–32.5%), '
             'about −159 descriptive Elo. It also used 5.0% more RL time and has slower inference. '
             'This is evidence against this particular 6×192 transformer and training recipe, not every transformer design.', '',
             'Separate NVIDIA L4 experiment; the existing `qfinal` training process was not modified.', '',
             '## Training budgets', '',
             '| Model | RL minutes | Games | Optimizer steps |',
             '|---|---:|---:|---:|',
             f"| ResNet 128×10 | {reference['spent_seconds']/60:.2f} | {reference['totals']['games']:,} | {reference['totals']['steps']:,} |",
             f"| Transformer 6×192 | {elapsed/60:.2f} | {sum(r['games'] for r in metrics):,} | {sum(r['steps'] for r in metrics):,} |", '',
             'RL budget includes self-play and optimizer updates; evaluation and compilation outside that accounting are separate. '
             'Both use pure self-play, randomized 40/400 playouts, AdamW 3e-4, batch 1,024, replay 600,000 and reuse 4.', '',
             'The requested duration was matched to the recorded ResNet checkpoint, but the transformer finished its final '
             'iteration at 64.30 minutes versus 61.22 minutes for ResNet (+5.0%). This small mismatch favors the transformer. '
             'ResNet generated 1.62× as many games and performed 1.60× as many optimizer steps. '
             'An exact-duration rerun is unlikely to reverse this result; the measured score interval is well below 50%.', '',
             f"Transformer supervised training: {pre['log'][-1]['seconds']/60:.2f} minutes, four epochs. "
             'The historical ResNet log records 15.84 minutes on different hardware. The original supervised dataset was unavailable; '
             'its 100,000-game generation recipe was recreated. Replay seed data was copied identically. '
             '**This comparison matches RL time, not total lifetime compute.**', '',
             '## Playing strength', '',
             'Scores are from the transformer’s perspective; cutoffs receive half a point. '
             'Intervals bootstrap paired openings, not individual games; constant observed pair scores use conservative bounds. '
             'Intervals do not include training-seed uncertainty.', '',
             '| Opponent / search | Games | W / L / cutoff | Score | Paired 95% CI | Descriptive Elo |',
             '|---|---:|---|---:|---|---:|']
    for path in sorted((ROOT / 'eval').glob('*.json')):
        data = json.loads(path.read_text())
        s = data['summary']
        elo = s['descriptive_elo']
        elo = f'{elo:+.0f}' if isinstance(elo, (int, float)) else elo
        lo, hi = s['score_ci95']
        lines.append(f"| {path.stem} | {s['games']} | {s['wins']} / {s['losses']} / {s['cutoffs']} | "
                     f"{s['score']:.1%} | {lo:.1%}–{hi:.1%} | {elo} |")
    lines += ['', 'Only the h001 match is the primary matched-RL-budget comparison. Later ResNet checkpoints are context. '
              'One-simulation matches are a policy-dominated diagnostic. Equal search simulations do not mean equal thinking time.', '',
              '## Inference cost', '',
              'Synthetic steady-state production CUDA-graph timings; compilation excluded. These are network-only timings, not full search latency.', '',
              '| Architecture | Parameters | Batch | Milliseconds | Positions/s |', '|---|---:|---:|---:|---:|']
    for model in bench['models']:
        for row in model['inference']:
            lines.append(f"| {model['architecture']} | {model['parameters']:,} | {row['batch']} | {row['wall_ms']:.3f} | {row['positions_per_second']:,.0f} |")
    lines += ['', '## Limits', '',
              'One seed and one compact transformer configuration cannot establish that an architecture family is universally better. '
              'Pretraining initialization differs, shared hyperparameters may favor one model, and cutoff adjudication can affect strength estimates. '
              'A deployment choice should also consider equal-time matches if inference costs differ materially.', '',
              'Raw match JSON includes every game, paired uncertainty, checkpoint hashes, and settings. '
              'See PLAN.md, run.sh, source-sha256.txt, resnet-checkpoints.json and benchmark.json for provenance.', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
