# /// script
# requires-python = ">=3.11"
# dependencies = ["altair>=5", "polars>=1", "loguru"]
# ///
"""HTML report of a snapshot folder written by colab_final.sh: losses and strength over training time.

    uv run training/report_final.py SNAPSHOT_DIR [--out report.html]

Reads SNAPSHOT_DIR/metrics.json and config.json. Strength chains the hourly `prev_eval` matches
(each kept model against the previous one, starting from the hour-0 starting network) into Elo
relative to that starting network.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess

import altair as alt
from loguru import logger
import polars as pl

BLUE = '#2a78d6'
SMOOTH = 25  # iterations in the rolling mean of per-iteration series
Z = 1.96


def iterations(metrics):
    frame = pl.DataFrame([dict(iteration=r['iteration'], seconds=r['seconds'], games=r['games'],
                               positions=r['positions'], steps=r['steps'], policy=r['policy'], value=r['value'],
                               mean_plies=r['mean_plies'], first_wins=r['selfplay_first_wins'],
                               eval_score=r['eval']['score'] if 'eval' in r else None,
                               eval_games=r['eval']['games'] if 'eval' in r else None) for r in metrics])
    return frame.with_columns(
        hours=pl.col('seconds').cum_sum() / 3600, total_games=pl.col('games').cum_sum(),
        total_positions=pl.col('positions').cum_sum(), total_steps=pl.col('steps').cum_sum(),
        first_win_rate=pl.col('first_wins') / pl.col('games'),
        policy_smooth=pl.col('policy').rolling_mean(SMOOTH, min_samples=1),
        value_smooth=pl.col('value').rolling_mean(SMOOTH, min_samples=1))


def strength(metrics, frame):
    """Chained Elo from the hourly matches; each step's variance adds up along the chain."""
    rows, previous = [], 'model-h000-it00000'
    for r, hours, games in zip(metrics, frame['hours'], frame['total_games']):
        if 'prev_eval' not in r:
            continue
        match = r['prev_eval']
        assert match['opponent'] == previous, (match['opponent'], previous)
        previous = f"model-h{int(hours):03d}-it{r['iteration']:05d}"
        points = match['wins'] + 0.5 * match['cutoffs']
        score = (points + 0.5) / (match['games'] + 1)  # keeps 0 and 1 finite
        rows.append(dict(snapshot=previous.removeprefix('model-'), opponent=match['opponent'].removeprefix('model-'),
                         iteration=r['iteration'], hours=hours, total_games=games, match_games=match['games'],
                         wins=match['wins'], losses=match['losses'], cutoffs=match['cutoffs'],
                         score=points / match['games'],
                         score_se=math.sqrt(score * (1 - score) / match['games']),
                         elo_step=400 * math.log10(score / (1 - score)),
                         elo_step_se=400 / math.log(10) / math.sqrt(match['games'] * score * (1 - score))))
    start = dict(snapshot='h000-it00000', opponent=None, iteration=0, hours=0.0, total_games=0, match_games=0, wins=0,
                 losses=0, cutoffs=0, score=None, score_se=None, elo_step=0.0, elo_step_se=0.0)
    return pl.DataFrame([start] + rows).with_columns(
        elo=pl.col('elo_step').cum_sum(), elo_se=pl.col('elo_step_se').pow(2).cum_sum().sqrt(),
        score_low=(pl.col('score') - Z * pl.col('score_se')).clip(0, 1),
        score_high=(pl.col('score') + Z * pl.col('score_se')).clip(0, 1)).with_columns(
        elo_low=pl.col('elo') - Z * pl.col('elo_se'), elo_high=pl.col('elo') + Z * pl.col('elo_se'))


def hours_axis():
    return alt.X('hours:Q', title='Training hours (self-play + training)', scale=alt.Scale(nice=False))


def elo_chart(table):
    base = alt.Chart(table).encode(hours_axis())
    tooltip = ['snapshot', alt.Tooltip('total_games:Q', format=','), alt.Tooltip('elo:Q', format='.0f'),
               alt.Tooltip('elo_low:Q', format='.0f'), alt.Tooltip('elo_high:Q', format='.0f')]
    return (base.mark_area(color=BLUE, opacity=0.15).encode(alt.Y('elo_low:Q'), alt.Y2('elo_high:Q'))
            + base.mark_line(color=BLUE, strokeWidth=2).encode(alt.Y('elo:Q', title='Elo vs starting network'))
            + base.mark_point(color=BLUE, filled=True, size=40).encode(alt.Y('elo:Q'), tooltip=tooltip)
            ).properties(width='container', height=280,
                         title='Strength relative to the starting network (chained hourly matches, 95% band)')


def step_chart(table):
    base = alt.Chart(table.filter(pl.col('score').is_not_null())).encode(hours_axis())
    tooltip = ['snapshot', 'opponent', 'wins', 'losses', 'cutoffs', alt.Tooltip('score:Q', format='.3f')]
    rule = alt.Chart(pl.DataFrame(dict(y=[0.5]))).mark_rule(color='#8a8984', strokeDash=[4, 4]).encode(y='y:Q')
    return (rule + base.mark_rule(color=BLUE).encode(alt.Y('score_low:Q'), alt.Y2('score_high:Q'))
            + base.mark_point(color=BLUE, filled=True, size=40).encode(
                alt.Y('score:Q', title='Score vs previous hour', scale=alt.Scale(domain=[0, 1])), tooltip=tooltip)
            ).properties(width='container', height=220, title='Each hourly model against the previous one (95% CI)')


def series_chart(frame, raw, smooth, title, y_title):
    base = alt.Chart(frame.select('hours', 'iteration', 'total_games', raw, smooth)).encode(hours_axis())
    tooltip = ['iteration', alt.Tooltip('total_games:Q', format=','), alt.Tooltip(f'{raw}:Q', format='.4f'),
               alt.Tooltip(f'{smooth}:Q', format='.4f')]
    return (base.mark_line(color=BLUE, opacity=0.25, strokeWidth=1).encode(alt.Y(f'{raw}:Q', title=y_title, scale=alt.Scale(zero=False)))
            + base.mark_line(color=BLUE, strokeWidth=2).encode(alt.Y(f'{smooth}:Q'), tooltip=tooltip)
            ).properties(width='container', height=200, title=title)


def points_chart(frame, column, title, y_title, domain=None):
    data = frame.filter(pl.col(column).is_not_null())
    y = alt.Y(f'{column}:Q', title=y_title, scale=alt.Scale(domain=domain) if domain else alt.Undefined)
    return alt.Chart(data).mark_point(color=BLUE, filled=True, size=20, opacity=0.6).encode(
        hours_axis(), y, tooltip=['iteration', alt.Tooltip(f'{column}:Q', format='.3f')]
    ).properties(width='container', height=200, title=title)


def table_html(table):
    columns = ['snapshot', 'hours', 'total_games', 'match_games', 'wins', 'losses', 'cutoffs', 'score', 'elo', 'elo_se']
    head = ''.join(f'<th>{c}</th>' for c in columns)
    cell = lambda v: '' if v is None else f'{v:,.2f}' if isinstance(v, float) else f'{v:,}' if isinstance(v, int) else v
    body = ''.join('<tr>' + ''.join(f'<td>{cell(row[c])}</td>' for c in columns) + '</tr>'
                   for row in table.select(columns).iter_rows(named=True))
    return f'<table class="table is-narrow is-striped is-size-7"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


PAGE = """<!DOCTYPE html>
<html data-theme="light"><head><meta charset="utf-8"><title>{title}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@1.0.2/css/bulma.min.css">
<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
<style>.chart {{ width: 100%; margin-bottom: 2rem; }}</style></head>
<body><section class="section"><div class="container">
<h1 class="title">{title}</h1>
<p class="subtitle is-6">Report written {now} · git {commit} · source {source}</p>
<p>{summary}</p>
<details class="mb-5"><summary>Run configuration ({n_config} settings)</summary><pre>{config}</pre></details>
<h2 class="title is-4">Strength</h2>
<p>Elo adds up the hourly head-to-head results ({pairs} paired games each, {sims} simulations, held-out openings).
Chained results tend to overstate total progress against third parties; a round robin of the kept models
(<code>az.py h2h</code>) is the stronger test.</p>
<div id="elo" class="chart"></div><div id="step" class="chart"></div>
<details class="mb-5"><summary>All {n_snapshots} hourly matches (table)</summary>{table}</details>
<h2 class="title is-4">Training loss</h2>
<p>Per iteration (light) and {smooth}-iteration rolling mean. Policy loss counts only 400-simulation rows.</p>
<div id="policy" class="chart"></div><div id="value" class="chart"></div>
<h2 class="title is-4">Other signals</h2>
<div id="negamax" class="chart"></div>
<details><summary>Self-play game length and first-player win rate (2 charts)</summary>
<div id="plies" class="chart"></div><div id="first" class="chart"></div></details>
</div></section>
<script>
const specs = {specs};
for (const [id, spec] of Object.entries(specs)) vegaEmbed('#' + id, spec, {{actions: false}});
</script></body></html>
"""


@logger.catch(reraise=True)
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('snapshots', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    out = args.out or args.snapshots / 'report.html'
    logger.add(out.parent / 'logs' / 'report_{time}.log')
    metrics = json.loads((args.snapshots / 'metrics.json').read_text())
    config = json.loads((args.snapshots / 'config.json').read_text())
    frame = iterations(metrics)
    table = strength(metrics, frame)
    logger.info(f'{len(metrics)} iterations, {frame["total_games"][-1]:,} games, {len(table) - 1} hourly matches')
    alt.data_transformers.disable_max_rows()
    specs = dict(
        elo=elo_chart(table), step=step_chart(table),
        policy=series_chart(frame, 'policy', 'policy_smooth', 'Policy loss (cross-entropy)', 'Policy loss'),
        value=series_chart(frame, 'value', 'value_smooth', 'Value loss (mean squared error)', 'Value loss'),
        negamax=points_chart(frame, 'eval_score', f"Score vs negamax ({config['negamax_nodes'][0]:,} nodes, "
                             f"{2 * config['eval_pairs']} games per point)", 'Score', [0, 1]),
        plies=points_chart(frame, 'mean_plies', 'Mean plies per self-play game', 'Plies'),
        first=points_chart(frame, 'first_win_rate', 'First player win rate in self-play', 'Win rate', [0, 1]))
    keys = ('init', 'games', 'negamax_fraction', 'simulations', 'playout_cap', 'optimizer', 'lr', 'batch_size',
            'replay', 'reuse', 'budget_minutes', 'prev_eval_pairs')
    summary = (f"{len(metrics):,} iterations over {frame['hours'][-1]:.1f} training hours: "
               f"{frame['total_games'][-1]:,} self-play games, {frame['total_positions'][-1]:,} positions, "
               f"{frame['total_steps'][-1]:,} optimizer steps. "
               + ', '.join(f'{k} = {config[k]}' for k in keys) + '.')
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                            cwd=Path(__file__).parent, check=True).stdout.strip()
    out.write_text(PAGE.format(
        title='Final RL run', now=datetime.now().strftime('%Y-%m-%d %H:%M'), commit=commit, source=args.snapshots,
        summary=summary, n_config=len(config), config=json.dumps(config, indent=1), pairs=2 * config['prev_eval_pairs'],
        sims=config['eval_simulations'], n_snapshots=len(table) - 1, table=table_html(table), smooth=SMOOTH,
        specs=json.dumps({k: v.to_dict() for k, v in specs.items()})))
    logger.info(f'wrote {out}')


if __name__ == '__main__':
    main()
