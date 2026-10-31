"""Write an evidence-only report from the L4 ablation queue's actual outputs."""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def summarize(root):
    plan = read(root / 'plan.json')
    if (root / 'tablebase/metrics.json').exists():
        plan['arms']['tablebase'] = ['--race-solver', 'tablebase']
    def evaluation(arm, label):
        return read(root / f'{arm}-{label}.json') or read(root / arm / f'{label}.json')
    lines = ['# L4 RL ablation results', '',
             'The network is unchanged (128 channels, 10 residual blocks). These are optimizer and search ablations.', '',
             f"Budget: {plan['minutes']:g} minutes of self-play and training per arm; one training seed ({plan['seed']}). "
             f"Schedules use {plan['schedule_steps']:,} updates. Evaluation uses {plan['pairs']} paired openings.", '',
             '## Training throughput', '',
             '| Arm | Iterations | Actual min | Games | Positions | Updates | Games/s | Median iteration s |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    summary = {}
    for arm in plan['arms']:
        metrics = read(root / arm / 'metrics.json')
        if not metrics:
            status = 'rejected on inference cost' if arm == 'race_solver' else 'pending'
            lines.append(f'| {arm} | {status} | | | | | | |')
            continue
        seconds = sum(m['seconds'] for m in metrics)
        games = sum(m['games'] for m in metrics)
        row = dict(iterations=len(metrics), seconds=seconds, games=games,
                   positions=sum(m['positions'] for m in metrics), steps=sum(m['steps'] for m in metrics),
                   games_per_second=games / seconds, median_iteration_seconds=statistics.median(
                       m['seconds'] for m in (metrics[1:] or metrics)),
                   play_seconds=sum(m['play_seconds'] for m in metrics),
                   policy_rows=sum(m.get('policy_rows', 0) for m in metrics),
                   value_only_rows=sum(m.get('value_only_rows', 0) for m in metrics),
                   search_stats=metrics[-1]['search_stats'],
                   caps={cap: sum(m.get('playout_caps', {}).get(cap, 0) for m in metrics) for cap in ['40', '400']})
        summary[arm] = row
        lines.append(f"| {arm} | {row['iterations']} | {seconds / 60:.2f} | {games:,} | "
                     f"{row['positions']:,} | {row['steps']:,} | {row['games_per_second']:.2f} | "
                     f"{row['median_iteration_seconds']:.1f} |")
    lines += ['', '## Held-out strength', '',
              'Score = wins + half draws, divided by games. Intervals resample opening pairs, retaining both colors.', '',
              '| Arm | Network vs N2000 | PUCT128 vs N2000 | PUCT128 vs N32000 | H2H vs baseline |',
              '|---|---:|---:|---:|---:|']
    for arm in plan['arms']:
        cells = []
        for label in ['s1-n2000', 's128-n2000', 's128-n32000', 'vs-baseline']:
            result = evaluation(arm, label)
            if result is None:
                cells.append('rejected on cost' if arm == 'race_solver' else '—' if arm == 'baseline' and label == 'vs-baseline' else 'pending')
            else:
                lo, hi = result['pair_bootstrap_95']
                cells.append(f"{result['score']:.3f} [{lo:.3f}, {hi:.3f}]")
        lines.append(f"| {arm} | " + ' | '.join(cells) + ' |')
    lines += ['', '## Paired change against the stronger fixed opponent', '',
              'Each difference compares the same openings and colors with baseline PUCT128 vs negamax-32000.', '',
              '| Arm | Score change | Paired bootstrap 95% interval |', '|---|---:|---:|']
    baseline = read(root / 'baseline-s128-n32000.json')
    if baseline:
        base = np.array([g['score'] for g in baseline['games']]).reshape(-1, 2).mean(1)
        for arm in plan['arms']:
            other = evaluation(arm, 's128-n32000')
            if arm == 'baseline' or other is None:
                continue
            delta = np.array([g['score'] for g in other['games']]).reshape(-1, 2).mean(1) - base
            rng = np.random.default_rng(43)
            samples = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1)
            lo, hi = np.quantile(samples, [0.025, 0.975])
            lines.append(f'| {arm} | {100 * delta.mean():+.1f} pp | [{100 * lo:+.1f}, {100 * hi:+.1f}] pp |')
    overhead = read(root / 'optimizer-overhead.json')
    if overhead:
        times = {name: statistics.median(r['milliseconds_per_step'] for r in overhead if r['optimizer'] == name)
                 for name in ['adamw', 'muon']}
        ratio = times['muon'] / times['adamw']
        lines += ['', '## Optimizer overhead', '',
                  f"Identical replay batches, fresh initial model, 20 warmup updates per repeat: "
                  f"AdamW {times['adamw']:.2f} ms/update; Muon {times['muon']:.2f} ms/update "
                  f"(median, Muon/AdamW = {ratio:.3f}).",
                  f"Muon needs {100 * (1 - 1 / ratio):.1f}% fewer updates to offset this measured update-time ratio "
                  'in training alone. This calculation does not establish that it actually needs fewer updates.']
    if 'playout_40_400' in summary:
        row = summary['playout_40_400']
        caps = row['caps']
        lines += ['', '## Randomized cap diagnostics', '',
                  f"Finished games contain {caps['40']:,} low-cap and {caps['400']:,} high-cap learning moves. "
                  f"Replay additions: {row['policy_rows']:,} policy-bearing rows and {row['value_only_rows']:,} "
                  'value-only rows (the latter also include negamax moves). Retained-tree visits count toward caps; '
                  '130 nominal simulations per move is not a measurement of fresh network work.']
    if 'race_solver' in summary:
        s = summary['race_solver']['search_stats']
        lines += ['', '## Race proof diagnostics', '',
                  f"Beyond the existing depth-4 proof: {s['race_queries']:,} extended queries, "
                  f"{s['race_solved']:,} proved, {s['race_unknown']:,} unknown, {s['race_nodes']:,} nodes, "
                  f"{s['race_seconds']:.2f} aggregate CPU-thread seconds. This is not wall time.",
                  'The solver searches the remaining game horizon with a 2,048-node cap. Exact returned values '
                  'respect pawn jumps, threefold repetitions, and the true move cutoff. Unknown results use the '
                  'network; this is not an exhaustive tablebase guaranteeing a result for every zero-wall position.']
    race = read(root / 'race-inference.json')
    if race and race['rows']:
        times = {mode: statistics.median(r['seconds'] for r in race['rows'] if r['mode'] == mode)
                 for mode in ['legacy', 'exact']}
        lines += ['', '## Solver inference microbenchmark', '',
                  f"{race['positions']} zero-wall replay positions, {race['simulations']} simulations each: "
                  f"median legacy {times['legacy']:.3f}s, extended proof {times['exact']:.3f}s. "
                  f"Speed ratio legacy/extended = {times['legacy'] / times['exact']:.3f}.",
                  'These isolated queries use the replay row’s current-position repetition count and otherwise fresh '
                  'history; the full training arm uses complete histories. This measures runtime, not playing strength.']
    table = read(root / 'tablebase/inference.json')
    if table and table['rows']:
        times = {mode: statistics.median(r['seconds'] for r in table['rows'] if r['mode'] == mode)
                 for mode in ['legacy', 'tablebase']}
        lines += ['', '## Exact tablebase inference', '',
                  f"Median search batch: legacy {times['legacy']:.3f}s; tablebase {times['tablebase']:.3f}s. "
                  f"Legacy/tablebase speed ratio {times['legacy']/times['tablebase']:.3f}.",
                  'This is a separate implementation from the bounded proof extension. It solves the fixed-wall pawn graph '
                  'with repetition-aware draw sinks and exact distance-to-mate cutoff handling. '
                  'Source provenance is in `tablebase/source.json`.']
    enabled = read(root / 'tablebase/solver-on-s128-n32000.json')
    disabled = evaluation('tablebase', 's128-n32000')
    if enabled and disabled:
        delta = (np.array([g['score'] for g in enabled['games']]) -
                 np.array([g['score'] for g in disabled['games']])).reshape(-1, 2).mean(1)
        rng = np.random.default_rng(44)
        lo, hi = np.quantile(delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1), [0.025, 0.975])
        lines += ['', '## Solver enabled during play', '',
                  f"Same tablebase-trained model, PUCT128 vs negamax32000: solver off {disabled['score']:.3f}, "
                  f"solver on {enabled['score']:.3f}; paired change {100*delta.mean():+.1f} pp "
                  f"[95% interval {100*lo:+.1f}, {100*hi:+.1f} pp]."]
    confirmation = read(root / 'confirmation/confirm-vs-baseline.json')
    if confirmation:
        lo, hi = confirmation['pair_bootstrap_95']
        lines += ['', '## Fresh-opening confirmation', '',
                  f"Playout randomization vs baseline on seed 2028, {len(confirmation['games'])} games: "
                  f"score {confirmation['score']:.3f}, paired-bootstrap 95% interval [{lo:.3f}, {hi:.3f}]. "
                  'This uses the same trained weights with fresh openings; it is not a second training seed.']
        for name in ['baseline', 'playout_40_400']:
            result = read(root / f'confirmation/confirm-{name}-strong.json')
            if result:
                lines.append(f"- {name}, PUCT128 vs negamax32000, seed 2028: {result['score']:.3f}.")
    lines += ['', '## Limits of the estimate', '',
              '- One training seed per arm: intervals cover opening variation, not variation between training runs.',
              '- Multiple-arm screening creates selection bias. A small apparent winner needs an independent-seed confirmation.',
              '- Short-run results do not establish the best long-run schedule; the final schedule horizon must match the final run.',
              '- Training losses use different replay distributions and cannot substantiate a 30–40% validation-step reduction.',
              '- All network comparisons use the baseline evaluation search. The isolated solver benchmark measures runtime, not playing strength.',
              '- No combined optimizer + randomized-cap + solver arm has been tested; individual benefits cannot simply be added.', '',
              'Exact arguments, source hashes, checkpoint hash and environment are in `plan.json` and `environment.json`.']
    decision = root / 'DECISION.md'
    if decision.exists():
        lines[2:2] = [decision.read_text(), '']
    (root / 'summary.json').write_text(json.dumps(summary, indent=2))
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    summarize(parser.parse_args().root)
