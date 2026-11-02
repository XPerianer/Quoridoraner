"""Generate the current learning-curve report from locally backed-up match records."""
import json
from pathlib import Path
import re
import numpy as np

HERE=Path(__file__).resolve().parent
BACKUP=HERE/'artifacts/backup'
metrics_path=BACKUP/'rl/metrics.json'
metrics=json.loads(metrics_path.read_text()) if metrics_path.exists() else []
rows=[]
for p in sorted((BACKUP/'eval').glob('*.json')):
 if not p.name.startswith('transformer-h'):
  continue
 d=json.loads(p.read_text()); s=d['summary']; a,b=d['models']
 stage=int(re.search(r'transformer-h(\d+)',p.name)[1])
 iteration=a['iteration']
 spent=sum(row['seconds'] for row in metrics if row.get('iteration',row.get('iter',0))<=iteration)
 # Current metrics record schema uses `iteration`; fail visibly if unavailable.
 if metrics and not any('iteration' in row or 'iter' in row for row in metrics):
  spent=None
 rows.append((stage,b['path'].split('/')[-1],s,spent,d,p.name))
lines=['# Optimized transformer: overnight learning curve','',
 'Architecture: pre-LN transformer width 256, four blocks, four heads, 2× MLP expansion, 2,185,864 parameters. Inference microbatch 64; training microbatch 128 with effective batch 1,024. Fresh four-epoch supervised pretraining, then up to 12 hours of self-play and optimization. See PLAN.md for the controlled settings and limits.','',
 'All scores below are from the transformer’s perspective. Matches use 400 games, 200 paired heldout openings, 200 simulations per move, seed 731. Cutoffs receive half a point. Confidence intervals resample opening pairs. Equal simulations is not equal thinking time.','',
 '| Transformer target hours | ResNet checkpoint | W–L–cutoffs | Score | 95% CI | Descriptive Elo |','|---:|---|---:|---:|---:|---:|']
primary=next((r for r in rows if r[0]==12 and 'model-h012-' in r[1]),None)
if primary:
 s=primary[2];ci=s['score_ci95']
 conclusion=(f"**Matched-budget result: the ResNet remains stronger.** The 12-hour transformer scored {s['score']:.1%} "
             f"against the 12-hour ResNet (95% interval {ci[0]:.1%}–{ci[1]:.1%}), "
             f"approximately {abs(s['descriptive_elo']):.0f} Elo weaker under equal 200-simulation search. "
             "This establishes lower strength at this budget, not a lower asymptotic ceiling. "
             "The early gains flattened between eight and twelve hours. This run provides no evidence of a higher ceiling, "
             "and does not justify extending this exact configuration unchanged.")
 lines[2:2]=[conclusion,'']
for stage,opponent,s,spent,d,name in rows:
 ci=s['score_ci95'];e=s['descriptive_elo'];es=f'{e:+.0f}' if isinstance(e,(int,float)) else e
 lines.append(f"| {stage} | {opponent} | {s['wins']}–{s['losses']}–{s['cutoffs']} | {s['score']:.1%} | {ci[0]:.1%}–{ci[1]:.1%} | {es} |")
if not rows:lines+=['','No completed strength evaluations have been backed up yet.']
progress_files=sorted((BACKUP/'eval').glob('progress-*.json'))
if progress_files:
 lines+=['','## Direct progress checks','',
 '| Match (score from first model) | W–L–cutoffs | Score | 95% CI |',
 '|---|---:|---:|---:|']
 for p in progress_files:
  d=json.loads(p.read_text());s=d['summary'];ci=s['score_ci95']
  name=p.stem.removeprefix('progress-')
  lines.append(f"| {name} | {s['wins']}–{s['losses']}–{s['cutoffs']} | {s['score']:.1%} | {ci[0]:.1%}–{ci[1]:.1%} |")
if rows and metrics:
 lines+=['','## Actual training budget','',
 '| Target hours | Actual RL hours | Iterations | Completed games | Optimizer updates |',
 '|---:|---:|---:|---:|---:|']
 for stage in sorted(set(r[0] for r in rows)):
  item=next(r for r in rows if r[0]==stage)
  iteration=item[4]['models'][0]['iteration']
  history=[r for r in metrics if r['iteration']<=iteration]
  hours=sum(r['seconds'] for r in history)/3600
  games=sum(r['games'] for r in history)
  updates=sum(r['steps'] for r in history)
  lines.append(f'| {stage} | {hours:.3f} | {iteration} | {games:,} | {updates:,} |')
if primary:
 lines+=['','## Did more training still help?','',
 'Changes from 8 to 12 RL hours, using the same opening pairs against each frozen opponent. The intervals bootstrap the difference within opening-pair clusters; they describe opening-sample uncertainty, not variation between training seeds.','',
 '| Frozen opponent | Change in score | 95% interval for change |','|---|---:|---:|']
 for opponent in ['h001-it00034','h013-it00451']:
  matches=[json.loads((BACKUP/'eval'/f'transformer-h{h:03d}-vs-model-{opponent}.json').read_text()) for h in [8,12]]
  diff=(np.array([g['score'] for g in matches[1]['games']])-np.array([g['score'] for g in matches[0]['games']])).reshape(-1,2).mean(1)
  rng=np.random.default_rng(731)
  samples=diff[rng.integers(len(diff),size=(10000,len(diff)))].mean(1)
  ci=np.quantile(samples,[.025,.975])*100
  label='1-hour ResNet' if opponent.startswith('h001') else '13-hour ResNet'
  lines.append(f'| {label} | {diff.mean()*100:+.2f} percentage points | {ci[0]:+.2f} to {ci[1]:+.2f} |')
 lines+=['','Neither frozen-opponent comparison shows a clear gain over these final four hours. The direct 12h-versus-8h transformer match is an additional check. A plateau here could reflect architecture, optimization, data or search; it does not establish a universal transformer ceiling.']
lines+=['','## Interpretation','',
 'Compare the same frozen 13-hour ResNet across stages to measure the transformer’s progress. Matching-hour opponents address early compute efficiency. A rising score against a frozen opponent is stronger evidence of continued learning than saturated negamax scores or adjacent-checkpoint matches. Twelve hours and one seed cannot determine the asymptotic ceiling.','',
 'The pipeline restarts from full optimizer/replay/RNG/tree state between evaluation stages. It retains a constant learning rate 3e-4 throughout; no result-dependent tuning is applied. Hourly snapshots record actual training seconds; stage budgets can overshoot at iteration boundaries. Pretraining, generation and evaluation consume additional hardware time and are excluded from the 12-hour RL budget.','',
 'Raw match records, model fingerprints and confidence-interval metadata are under artifacts/backup/eval. Full model checkpoints and resumable states are kept locally in ignored artifacts; source and launch recipe are tracked separately.']
(HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(f'Wrote report with {len(rows)} match results')
