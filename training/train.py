"""Resumable, two-arm AlphaZero-style experiment. Run with --help."""
import argparse
import copy
from collections import Counter, deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import sys
import time

import torch

from status import StatusWriter
from adaptive import opponent_nodes, progression, promotion_check
from charts import loss_curves
from curriculum import (training_position, evaluation_target, evaluation_opening, summarize,
                        curriculum_state, curriculum_check, gate_opening)
from replay import retain
from device import select_device
from bridge import create_bridge
from core import ARCHITECTURES, model_from_config, choose, encode, key, search_batch, update

VERSION = 2
MODES = ('selfplay', 'negamax')
STOP = False


def repetition_limit(config, phase):
    """Optional early rollout adjudication; evaluation retains threefold."""
    return config.get('training_repetition_limit', 3) if phase == 'train' else 3


def stop_requested(_signum, _frame):
    global STOP
    STOP = True


def atomic_save(value, path):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


ANSI = re.compile(r'\x1b\[[0-9;]*m')
RESET = '\033[0m'
DIM = '\033[2m'
BOLD = '\033[1m'
CYAN = '\033[36m'
MAGENTA = '\033[35m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
RED = '\033[31m'


def visible_width(text):
    return len(ANSI.sub('', text))


def clip(text, width):
    """Truncate styled text to a visible column budget."""
    if visible_width(text) <= width:
        return text
    result, used, index = [], 0, 0
    while index < len(text):
        match = ANSI.match(text, index)
        if match:
            result.append(match.group())
            index = match.end()
            continue
        if used >= max(0, width - 1):
            break
        result.append(text[index])
        used += 1
        index += 1
    return ''.join(result) + '…' + RESET


def pad(text, width):
    return text + ' ' * max(0, width - visible_width(text))


def join_columns(left, right, width, gap=2):
    """Place right-aligned text when it fits, otherwise keep the left message."""
    room = width - visible_width(right) - gap
    if room <= 0:
        return clip(left, width)
    return pad(clip(left, room), room + gap) + right


def progress_bar(fraction, width, color=CYAN):
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(width * fraction))
    if 0 < fraction < 1 and filled == 0:
        filled = 1
    if fraction >= 1:
        filled = width
    return f'{color}{"█" * filled}{DIM}{"░" * (width - filled)}{RESET}'


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f'{seconds}s'
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f'{minutes}m {seconds:02d}s'
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f'{hours}h {minutes:02d}m'
    days, hours = divmod(hours, 24)
    return f'{days}d {hours}h'


def sparkline(values, width=10):
    values = [value for value in values if value is not None and math.isfinite(value)][-width:]
    if len(values) < 2:
        return ''
    low, high = min(values), max(values)
    if high == low:
        return '▄' * len(values)
    shades = '▁▂▃▄▅▆▇█'
    return ''.join(shades[min(7, int((value - low) / (high - low) * 7.999))] for value in values)


def change_cell(current, previous):
    """Loss change since the previous optimizer update; lower is better."""
    if current is None or previous is None:
        return f'{DIM}—{RESET}'
    change = current - previous
    if change == 0:
        return f'{DIM}· 0{RESET}'
    color, arrow = (GREEN, '▼') if change < 0 else (RED, '▲')
    return f'{color}{arrow}{abs(change):.3f}{RESET}'


class Dashboard:
    def __init__(self, plain=False, workers=1, status=None):
        self.status = status
        self.workers = workers
        self.tty = sys.stdout.isatty() and not plain
        self.last = 0
        self.start = time.monotonic()
        self.moves = 0
        self.last_label = None
        self.phase_key = None
        self.phase_start = self.start
        self.phase_completed = 0
        self.records_seen = 0
        self.monitoring = {mode: [] for mode in MODES}
        self.recent = {mode: deque(maxlen=128) for mode in MODES}
        self.iteration_seen = None
        self.iteration_start = self.start
        self.iteration_durations = deque(maxlen=6)

    def show(self, state, device, detail='', force=False):
        if self.status:
            self.status.write(state, device, self.moves, detail)
        now = time.monotonic()
        label = (state['mode'], state['phase'], state['iteration'], state['update_index'], len(state['records']))
        if not force and (now - self.last < 0.2 or (not self.tty and label == self.last_label)):
            return
        self.last, self.last_label = now, label
        terminal = shutil.get_terminal_size((86, 24))
        width = max(30, min(110, terminal.columns))
        inner = max(20, width - 4)
        cfg, phase = state['config'], state['phase']
        active = [g for g in state.get('active_games', [])
                  if g['state']['winner'] == -1 and g['ply'] < cfg['max_plies']
                  and g['history'].get(key(g['state']['position']), 0) < 3]
        target = cfg['updates'] if phase == 'update' else evaluation_target(state) if phase == 'eval' else cfg['games']
        completed = state['update_index'] if phase == 'update' else state['game_index']
        unit = 'updates' if phase == 'update' else 'games'
        phase_key = (state['mode'], phase, state['iteration'])
        if phase_key != self.phase_key:
            self.phase_key, self.phase_start, self.phase_completed = phase_key, now, completed
        phase_elapsed = now - self.phase_start
        advanced = completed - self.phase_completed
        rate = advanced / max(phase_elapsed, .01)
        eta = f"{(target - completed) / rate:.0f}s" if rate > 0 and completed < target else '—'
        fraction = 1 if phase == 'done' else min(1, completed / max(1, target))
        bar = '━' * int(24 * fraction) + '·' * (24 - int(24 * fraction))
        elapsed = now - self.start
        task = {'train': 'Generate training games', 'update': 'Optimize policy & value network',
                'eval': 'Evaluate against negamax', 'done': 'Experiment complete'}.get(phase, phase)
        tasks = Counter(g.get('task', 'standard') for g in active)
        task_text = ' | '.join(f'{name}: {count}' for name, count in sorted(tasks.items())) or 'none'
        plies = [g['ply'] for g in active]
        ply_text = f"{sum(plies)/len(plies):.1f} mean ({min(plies)}–{max(plies)})" if plies else '—'
        arm = state['arms'][state['mode']]
        lines = [
            'QUORIDOR  /  ALPHAZERO LAB',
            f"{state['mode'].upper()}   {phase.upper()}   iteration {state['iteration']} / {cfg['iterations']}",
            f"Current task: {task}",
            f"[{bar}] {completed}/{target} {unit} ({fraction:.0%})   device {device}" if phase != 'done' else 'All phases complete',
            f"Phase {phase_elapsed:.0f}s | {rate*60:.1f} {unit}/min | phase ETA ~{eta}",
            f"Active tasks: {task_text}",
            f"Active games {len(active)}/{cfg.get('parallel_games', 1)} | ply {ply_text} | workers {self.workers}",
            f"Replay {len(arm['replay']):,}/{cfg['replay_size']:,} | batch {cfg['batch_size']} | updates {state['update_index']}/{cfg['updates']}",
            f"Session {elapsed:.0f}s | {self.moves:,} moves | {self.moves / max(elapsed, .01):.1f} moves/s",
        ]
        if phase == 'eval':
            suites = Counter(g.get('suite', 'monitor') for g in active)
            lines.append('Evaluation suites: ' + (' | '.join(f'{k}: {v}' for k, v in sorted(suites.items())) or 'preparing'))
        if detail:
            lines.append(detail)
        # Plain logs include live task metrics without scanning experiment history.
        if not self.tty:
            print(' | '.join(lines[1:]), flush=True)
            return
        for record in state['records'][self.records_seen:]:
            if record['phase'] == 'train':
                self.recent[record['mode']].append(record)
            elif record.get('suite', 'monitor') == 'monitor':
                self.monitoring[record['mode']].append(record)
        self.records_seen = len(state['records'])

        if self.iteration_seen is None:
            self.iteration_seen, self.iteration_start = state['iteration'], now
        elif state['iteration'] != self.iteration_seen:
            self.iteration_durations.append(max(0.0, now - self.iteration_start))
            self.iteration_seen, self.iteration_start = state['iteration'], now
        iteration = state['iteration']
        total_iterations = max(1, cfg['iterations'])
        if self.iteration_durations and iteration < total_iterations:
            average = sum(self.iteration_durations) / len(self.iteration_durations)
            iteration_eta = format_duration(average * (total_iterations - iteration))
        else:
            iteration_eta = '—'

        def heading(title, note=''):
            label_text = f'{BOLD}{CYAN}▎ {title.upper()}{RESET}'
            if note:
                return join_columns(label_text, f'{DIM}{note}{RESET}', inner, gap=1)
            fill = max(0, inner - visible_width(label_text) - 1)
            return label_text + ' ' + f'{DIM}{"─" * fill}{RESET}'

        def cell(text, size):
            return pad(clip(text, size), size)

        def evaluation_text(mode_name):
            evaluations = self.monitoring[mode_name]
            if not evaluations:
                return f'{DIM}pending evaluation{RESET}'
            latest = max(row['iteration'] for row in evaluations)
            results = [row['score'] for row in evaluations if row['iteration'] == latest]
            wins, draws, losses = results.count(1.0), results.count(0.5), results.count(0.0)
            score = sum(results) / len(results)
            baseline = [row['score'] for row in evaluations if row['iteration'] == 0]
            delta = score - sum(baseline) / len(baseline) if baseline else 0
            color = GREEN if score >= .5 else RED
            return (f'{DIM}iter {latest}{RESET}  {wins}W {draws}C {losses}L  '
                    f'{color}{score:.0%}{RESET}  {DIM}start{RESET} {delta:+.0%}')

        arm_rows = []
        for mode_name in MODES:
            mode_arm = state['arms'][mode_name]
            history = [row for row in state.get('loss_history', []) if row['mode'] == mode_name]
            latest_loss = history[-1] if history else mode_arm['loss']
            previous = history[-2] if len(history) >= 2 else None
            marker = '●' if mode_name == 'selfplay' else '◆'
            if latest_loss:
                loss_text = f"{latest_loss['policy']:.3f} / {latest_loss['value']:.3f}"
                previous_policy = previous['policy'] if previous else None
                previous_value = previous['value'] if previous else None
                delta_text = (f"{change_cell(latest_loss['policy'], previous_policy)} / "
                              f"{change_cell(latest_loss['value'], previous_value)}")
            else:
                loss_text, delta_text = 'awaiting updates', f'{DIM}— / —{RESET}'
            policy_trend = sparkline([row['policy'] for row in history], 7)
            value_trend = sparkline([row['value'] for row in history], 7)
            trend_text = (f'p{policy_trend} v{value_trend}' if policy_trend and value_trend
                          else f'{DIM}—{RESET}')
            arm_rows.append((marker, mode_name, loss_text, delta_text, trend_text, evaluation_text(mode_name)))

        mode_color = CYAN if state['mode'] == 'selfplay' else MAGENTA
        bar_width = 20 if inner >= 76 else 10
        run_fraction = min(1.0, (max(0, iteration - 1) + fraction) / total_iterations)
        if inner >= 60:
            run_text = (f'{DIM}run{RESET} {progress_bar(run_fraction, bar_width)} {run_fraction:5.1%}  '
                        f'{DIM}iter ETA{RESET} {iteration_eta}')
            iteration_text = (f'{BOLD}{mode_color}{state["mode"].upper()}{RESET}  '
                              f'{DIM}iteration{RESET} {BOLD}{iteration}{RESET}/{total_iterations}')
        else:
            run_text = ''
            iteration_text = (f'{BOLD}{mode_color}{state["mode"].upper()}{RESET}  '
                              f'{BOLD}{iteration}{RESET}/{total_iterations}  '
                              f'{progress_bar(run_fraction, bar_width)} {run_fraction:.0%}')
        header_right = f'{DIM}device{RESET} {BOLD}{device}{RESET}'
        if inner >= 60:
            header_right += f'   {DIM}session{RESET} {format_duration(elapsed)}'
        core = [
            join_columns(f'{BOLD}{CYAN}QUORIDORANER{RESET} {DIM}/{RESET} {BOLD}ALPHAZERO LAB{RESET}',
                         header_right, inner),
            f'{CYAN}{"─" * inner}{RESET}',
            join_columns(iteration_text, run_text, inner) if run_text else iteration_text,
        ]
        pipeline = []
        for name in ('train', 'update', 'eval'):
            style = f'{BOLD}{mode_color}' if name == phase else DIM
            pipeline.append(f'{style}{name.upper() if name == phase else name}{RESET}')
        task_line = f'{BOLD}Current task: {task}{RESET}'
        core.append(join_columns(task_line, f' {DIM}▸{RESET} '.join(pipeline), inner)
                    if inner >= 84 else task_line)
        if phase == 'done':
            core.append(f'{progress_bar(1, bar_width)} {DIM}all phases complete{RESET}')
        else:
            detail_text = f'{phase_elapsed:.0f}s'
            if inner >= 72:
                detail_text += f'   {rate * 60:.1f} {unit}/min'
            core.append(f'{DIM}Phase{RESET} {progress_bar(fraction, bar_width)} '
                        f'{completed}/{target} {unit} ({fraction:.0%})   '
                        f'{detail_text}   {DIM}ETA{RESET} ~{eta}')
        core += [
            '',
            heading('live'),
            f'{CYAN}Active games:{RESET} {len(active)}/{cfg.get("parallel_games", 1)}  '
            f'{DIM}·{RESET}  ply {ply_text}  {DIM}·{RESET}  workers {self.workers}',
            f'{CYAN}Active tasks: {task_text}{RESET}',
            f'{CYAN}Replay {len(arm["replay"]):,}/{cfg["replay_size"]:,}{RESET}  {DIM}·{RESET}  '
            f'batch {cfg["batch_size"]}  {DIM}·{RESET}  updates {state["update_index"]}/{cfg["updates"]}',
            f'{CYAN}Session{RESET} {elapsed:.0f}s  {DIM}|{RESET}  {self.moves:,} moves  '
            f'{DIM}|{RESET}  {self.moves / max(elapsed, .01):.1f} moves/s',
        ]
        if phase == 'eval':
            suites = Counter(g.get('suite', 'monitor') for g in active)
            suite_text = ' | '.join(f'{name}: {count}' for name, count in sorted(suites.items())) or 'preparing'
            core.append(f'{CYAN}Evaluation suites:{RESET} {suite_text}')

        core += ['', heading('arms', 'loss is policy / value · lower is better')]
        if inner >= 102:
            core.append(f'{DIM}{"arm":<11}{"loss (p / v)":<18}{"Δ update":<18}{"trend":<18}'
                        f'evaluation{RESET}')
            for marker, mode_name, loss_text, delta_text, trend_text, eval_text in arm_rows:
                core.append(cell(f'{marker} {mode_name}', 11) + cell(loss_text, 18) +
                            cell(delta_text, 18) + cell(trend_text, 18) + eval_text)
        elif inner >= 72:
            core.append(f'{DIM}{"arm":<11}{"loss (p / v)":<18}evaluation{RESET}')
            for marker, mode_name, loss_text, _delta, _trend, eval_text in arm_rows:
                core.append(cell(f'{marker} {mode_name}', 11) + cell(loss_text, 18) + eval_text)
        else:
            for marker, mode_name, loss_text, _delta, trend_text, eval_text in arm_rows:
                core.append(cell(f'{marker} {mode_name}', 11) + loss_text)
                core.append(f'  {eval_text}  {trend_text}')

        recent = self.recent[state['mode']]
        quality = ['', heading('training window')]
        if recent:
            repeat = sum(row['termination'] == 'repetition' for row in recent) / len(recent)
            goals = sum(row['termination'] == 'goal' for row in recent) / len(recent)
            limit = sum(row['termination'] == 'maxPlies' for row in recent) / len(recent)
            mean_plies = sum(row['plies'] for row in recent) / len(recent)
            goal_color = GREEN if goals >= .8 else YELLOW if goals >= .5 else RED
            repeat_color = GREEN if repeat <= .05 else YELLOW if repeat <= .2 else RED
            limit_color = GREEN if limit <= .05 else YELLOW if limit <= .2 else RED
            quality += [
                f'{CYAN}Recent {len(recent)} training games:{RESET} '
                f'goals {goal_color}{goals:.0%}{RESET}  {DIM}|{RESET}  '
                f'repetition {repeat_color}{repeat:.0%}{RESET}  {DIM}|{RESET}  '
                f'move limit {limit_color}{limit:.0%}{RESET}',
                f'{CYAN}Recent game length{RESET} {mean_plies:.1f} plies  '
                f'{DIM}|  cutoffs are artificial, not draws{RESET}',
            ]
        else:
            quality.append(f'{DIM}Awaiting completed training games…{RESET}')

        settings = []
        if cfg.get('curriculum') == 'gated':
            progress = curriculum_state(state)
            settings.append(f'{CYAN}Curriculum{RESET} race-{progress["stage"]}  {DIM}|{RESET}  passing streak '
                            f'{progress["streak"]}/{cfg["curriculum_streak"]}  {DIM}|{RESET}  '
                            f'replay {cfg.get("replay_sampling")}')
        elif cfg.get('curriculum') == 'races':
            stage = min(3, max(0, iteration - 1) // max(1, cfg.get('curriculum_interval', 25)))
            settings.append(f'{CYAN}Curriculum{RESET} race-{stage}  {DIM}|{RESET}  '
                            f'interval {cfg.get("curriculum_interval", 25)} iterations')
        if cfg.get('adaptive_negamax'):
            opponent = state.get('adaptive', {})
            settings.append(f'{CYAN}Adaptive opponent{RESET} {opponent.get("nodes", "—")} nodes  {DIM}|{RESET}  '
                            f'promotion streak {opponent.get("streak", 0)}/{cfg.get("promotion_streak", 2)}')
        settings.append(f'{CYAN}Checkpoint{RESET} every {cfg.get("checkpoint_seconds", 15):g}s '
                        f'+ Ctrl-C (save & pause)')
        if detail:
            settings.append(f'{CYAN}Latest{RESET} {detail}')
        config_section = ['', heading('run settings')] + settings

        # Prefer the live telemetry; add history panels and settings while the
        # terminal has room, so short windows still show the important numbers.
        budget = max(8, terminal.lines - 2)
        info = list(core)
        chosen = []
        for group in (quality, config_section):
            if len(info) + len(group) + 6 <= budget:
                info += group
                chosen.append(group)
        remaining = budget - len(info)
        chart_height = min(5, max(0, remaining - 5))
        chart = loss_curves(state.get('loss_history', []), inner, chart_height) if chart_height >= 2 else []
        if chart and len(chart) + 2 <= remaining:
            updates = max((sum(1 for row in state.get('loss_history', []) if row['mode'] == name)
                           for name in MODES), default=0)
            info += ['', heading('loss curves', f'{updates:,} updates' if updates else '')] + chart
        else:
            for group in (quality, config_section):
                if group not in chosen and len(info) + len(group) <= budget:
                    info += group

        info = [line.replace('●', f'{CYAN}●{RESET}').replace('◆', f'{MAGENTA}◆{RESET}')
                    .replace('◎', f'{YELLOW}◎{RESET}') for line in info]
        info = [clip(line, inner) for line in info]
        border = f'{CYAN}{"─" * max(1, width - 2)}{RESET}'
        print('\033[2J\033[H' + border + '\n' + '\n'.join('  ' + line for line in info) + '\n' + border,
              flush=True)


def new_arm(initial, device, config=None):
    model = model_from_config(config).to(device)
    model.load_state_dict(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    return model, optimizer


def save(state, models, optimizers, rng, directory, device, bridge=None):
    if bridge is not None:
        state['trees'] = bridge.ask(op='snapshot')['trees']
    for mode in MODES:
        state['arms'][mode]['model'] = models[mode].state_dict()
        state['arms'][mode]['optimizer'] = optimizers[mode].state_dict()
    state['rng'] = rng.getstate()
    state['torch_rng'] = torch.get_rng_state()
    state['mps_rng'] = torch.mps.get_rng_state() if device == 'mps' else None
    state['cuda_rng'] = torch.cuda.get_rng_state() if device == 'cuda' else None
    atomic_save(state, directory / 'latest.pt')
    temp = directory / 'losses.json.tmp'
    temp.write_text(json.dumps(state.get('loss_history', []), indent=2) + '\n')
    os.replace(temp, directory / 'losses.json')
    # Human-readable log is derived from checkpoint data, so resume never duplicates records.
    temp = directory / 'metrics.json.tmp'
    temp.write_text(json.dumps(state['records'], indent=2) + '\n')
    os.replace(temp, directory / 'metrics.json')
    (directory / 'summary.json').write_text(json.dumps(summarize(state['records']), indent=2) + '\n')
    (directory / 'opponent.json').write_text(json.dumps(state.get('adaptive'), indent=2) + '\n')
    (directory / 'curriculum.json').write_text(json.dumps(state.get('curriculum_progress'), indent=2) + '\n')
    (directory / 'config.json').write_text(json.dumps(state['config'], indent=2) + '\n')


def fresh_game(bridge, state, index=None):
    index = state['game_index'] if index is None else index
    cfg = state['config']
    suite, task = 'monitor', 'standard'
    if state['phase'] == 'eval':
        suite = 'heldout' if index >= cfg['eval_games'] else 'monitor'
        pair_index = index - cfg['eval_games'] if suite == 'heldout' else index
        fixed_target = cfg['eval_games'] + (cfg.get('heldout_games', 0) if state['iteration'] == cfg['iterations'] else 0)
        gate_start = fixed_target + (cfg.get('promotion_games', 40)
            if cfg.get('adaptive_negamax') and state['mode'] == 'negamax' else 0)
        if index >= fixed_target and cfg.get('adaptive_negamax') and state['mode'] == 'negamax':
            suite, pair_index = 'promotion', index - fixed_target
        if cfg.get('curriculum') == 'gated' and index >= gate_start:
            suite = 'curriculum'
            stage = curriculum_state(state)['stage']
            task = f'race-{stage}'
            current = bridge.ask(position=gate_opening(cfg, state['iteration'], index - gate_start, stage))
            history = {key(current['position']): 1}
        else:
            current, history = evaluation_opening(bridge, cfg, pair_index, suite)
    else:
        stage = curriculum_state(state)['stage'] if cfg.get('curriculum') == 'gated' else None
        position, task = training_position(cfg, state['iteration'], index, stage)
        current = bridge.ask(position=position)
        history = {key(current['position']): 1}
    return {'state': current, 'history': history, 'ply': 0, 'samples': [], 'moves': [],
            'opponent_nodes': opponent_nodes(state, suite),
            'suite': suite, 'task': task, 'learner': index % 2, 'opening': current['position'], 'index': index,
            'id': f"{state['mode']}:{state['phase']}:{state['iteration']}:{index}"}


def finish_game(state, winner, reason, game=None):
    game = state['game'] if game is None else game
    score = 0.5 if winner == -1 else float(winner == game['learner'])
    record = {'mode': state['mode'], 'phase': state['phase'], 'iteration': state['iteration'],
              'game': game.get('index', state['game_index']), 'winner': winner, 'learner': game['learner'],
              'score': score, 'termination': reason, 'plies': game['ply'],
              'opening': game['opening'], 'moves': game['moves'],
              'suite': game.get('suite', 'monitor'), 'task': game.get('task', 'standard'),
              'opponent_nodes': game.get('opponent_nodes', state['config'].get('negamax_nodes', 2000)),
              'adjudicated': reason != 'goal'}
    state['records'].append(record)
    if state['phase'] == 'train':
        replay = state['arms'][state['mode']]['replay']
        for features, policy, player in game['samples']:
            outcome = 0.0 if winner == -1 else (1.0 if player == winner else -1.0)
            row = (features, policy, outcome)
            replay.append(row + (game.get('task', 'standard'),)
                          if state['config'].get('replay_sampling') == 'task' else row)
        retain(replay, state['config']['replay_size'], state['config'].get('replay_sampling') == 'task')
    state['game'] = None
    state['game_index'] += 1


def fill_games(state, bridge, target):
    """Completed count + active count is the next unstarted, unique index."""
    active = state['active_games']
    if active and state['config'].get('game_scheduler', 'cohort') == 'cohort':
        return
    next_index = state['game_index'] + len(active)
    stop = min(target, next_index + state['config']['parallel_games'] - len(active))
    active.extend(fresh_game(bridge, state, index) for index in range(next_index, stop))


def search_budgets(state, games):
    """Independent deterministic move budgets, unaffected by scheduling/RNG use."""
    cfg = state['config']
    full = cfg['simulations']
    if state['phase'] != 'train' or cfg.get('search_budget', 'fixed') == 'fixed':
        return {g['id']: full for g in games}
    result = {}
    for game in games:
        rng = random.Random(f"budget:{cfg['seed']}:{state['mode']}:{state['iteration']}:{game['index']}:{game['ply']}")
        result[game['id']] = full if rng.random() < cfg['full_search_probability'] else cfg['fast_simulations']
    return result


def advance(state):
    """Advance at transaction boundaries; iteration zero evaluates untrained weights."""
    cfg = state['config']
    if state['phase'] == 'eval':
        curriculum_check(state)
    if state['phase'] == 'eval' and state['mode'] == 'negamax':
        promotion_check(state)
    if state['phase'] == 'train':
        state['phase'] = 'update'
        state['update_index'] = 0
    elif state['phase'] == 'update':
        state['phase'] = 'eval'
        if state['iteration'] % cfg.get('eval_every', 1) and state['iteration'] != cfg['iterations']:
            advance(state)
    elif state['mode'] == 'selfplay':
        state['mode'] = 'negamax'
        state['phase'] = 'eval' if state['iteration'] == 0 else 'train'
    elif state['iteration'] >= cfg['iterations']:
        state['phase'] = 'done'
    else:
        state['iteration'] += 1
        state['mode'] = 'selfplay'
        state['phase'] = 'train'
    state['game_index'] = 0


def run(state, models, optimizers, rng, directory, device, plain):
    worker_count = min(state['config'].get('workers', 1), state['config']['parallel_games'],
                       max(state['config']['games'], state['config']['eval_games'] + state['config'].get('heldout_games', 0)))
    bridge = create_bridge(state['config'].get('engine', 'typescript'), workers=worker_count,
                           mps_inference=state['config'].get('mps_inference', 'bucketed'))
    inference_label = f"{device}/{state['config'].get('mps_inference', 'bucketed')}" if (
        device == 'mps' and state['config'].get('engine') == 'direct') else device
    print(f"Search engine: {state['config'].get('engine', 'typescript')}; workers: {worker_count}; "
          f"shared neural inference: {inference_label}.", flush=True)
    status = StatusWriter(directory)
    display = Dashboard(plain, worker_count, status)
    status.write(state, device, detail="Starting training engine", force=True)
    cfg = state['config']
    last_checkpoint = time.monotonic()
    started = time.perf_counter()
    previous_stats = dict(state.get('performance', {}))
    checkpoint_seconds = 0.0
    checkpoint_count = 0
    update_seconds = 0.0

    def checkpoint(force=False):
        nonlocal last_checkpoint, checkpoint_seconds, checkpoint_count
        status.write(state, device, display.moves)
        if not force and time.monotonic() - last_checkpoint < cfg['checkpoint_seconds']:
            return
        state['game'] = next(iter(state['active_games']), None)
        state['performance'] = {k: previous_stats.get(k, 0) + v for k, v in bridge.stats.items()}
        state['performance'].update(
            training_seconds=previous_stats.get('training_seconds', 0) + time.perf_counter() - started,
            checkpoint_seconds=previous_stats.get('checkpoint_seconds', 0) + checkpoint_seconds,
            checkpoints=previous_stats.get('checkpoints', 0) + checkpoint_count,
            update_seconds=previous_stats.get('update_seconds', 0) + update_seconds)
        before = time.perf_counter()
        save(state, models, optimizers, rng, directory, device, bridge)
        checkpoint_seconds += time.perf_counter() - before
        checkpoint_count += 1
        last_checkpoint = time.monotonic()

    try:
        if state.get('trees'):
            bridge.ask(op='restore', snapshot=state['trees'])
        while state['phase'] != 'done' and not STOP:
            mode = state['mode']
            model = models[mode]
            if state['phase'] == 'update':
                if state['update_index'] >= cfg['updates']:
                    advance(state)
                else:
                    arm = state['arms'][mode]
                    if not arm['replay']:
                        raise RuntimeError('No learning samples collected')
                    model.train()
                    before = time.perf_counter()
                    arm['loss'] = update(model, optimizers[mode], arm['replay'], cfg['batch_size'], rng, device,
                                         cfg.get('replay_sampling', 'uniform'))
                    update_seconds += time.perf_counter() - before
                    state['update_index'] += 1
                    state.setdefault('loss_history', []).append({'mode': mode, 'iteration': state['iteration'],
                        'update': state['update_index'], **arm['loss']})
                    display.show(state, device, f"Gradient update {state['update_index']}/{cfg['updates']}", force=display.tty)
                checkpoint()
                continue
            target = evaluation_target(state) if state['phase'] == 'eval' else cfg['games']
            if state['game_index'] >= target:
                advance(state)
                # Trees must never survive a network update or a different evaluation phase.
                bridge.ask(op='clear')
                checkpoint()
                continue
            fill_games(state, bridge, target)
            remaining = []
            for game in state['active_games']:
                position = game['state']['position']
                winner = game['state']['winner']
                if winner != -1 or game['history'][key(position)] >= repetition_limit(cfg, state['phase']) or game['ply'] >= cfg['max_plies']:
                    reason = 'goal' if winner != -1 else ('repetition' if game['history'][key(position)] >= repetition_limit(cfg, state['phase']) else 'maxPlies')
                    finish_game(state, winner, reason, game)
                    bridge.ask(op='drop', ids=[game['id']])
                    display.show(state, device, f'Game {game["index"] + 1} finished: {reason}', force=True)
                else:
                    remaining.append(game)
            state['active_games'] = remaining
            fill_games(state, bridge, target)
            state['game'] = next(iter(remaining), None)
            if not remaining:
                checkpoint()
                continue
            learning = state['phase'] == 'train'
            neural_games = [g for g in remaining if (learning and mode == 'selfplay')
                            or g['state']['position']['turn'] == g['learner']]
            policies = {}
            if neural_games:
                budgets = search_budgets(state, neural_games)
                model.eval()
                policies = search_batch(model, bridge, neural_games, cfg['max_plies'], budgets, rng, device,
                    explore=learning, progress=lambda n: display.show(state, device,
                        f"MCTS wave {n} · neural batch {len(neural_games)}"))
                if learning:
                    for name, value in [('training_search_moves', len(budgets)),
                                        ('training_requested_simulations', sum(budgets.values())),
                                        ('training_full_search_moves', sum(b == cfg['simulations'] for b in budgets.values()))]:
                        bridge.stats[name] = bridge.stats.get(name, 0) + value
            opponents = [g for g in remaining if g['id'] not in policies]
            opponent_moves = {}
            for nodes in sorted({g.get('opponent_nodes', cfg['negamax_nodes']) for g in opponents}):
                group = [g for g in opponents if g.get('opponent_nodes', cfg['negamax_nodes']) == nodes]
                replies = bridge.ask(op='negamax_batch', positions=[g['state']['position'] for g in group],
                                     nodes=nodes)['results']
                opponent_moves.update({g['id']: reply['move'] for g, reply in zip(group, replies)})
            moves = []
            for game in remaining:
                position = game['state']['position']
                if game['id'] in policies:
                    policy = policies[game['id']]
                    move = choose(policy, rng, sample=learning and game['ply'] < 20)
                    if learning:
                        game['samples'].append((encode(position, game['history'][key(position)],
                                                       (cfg['max_plies'] - game['ply']) / cfg['max_plies']),
                                                policy if budgets[game['id']] == cfg['simulations'] else None,
                                                position['turn']))
                else:
                    move = opponent_moves[game['id']]
                    if learning and cfg.get('negamax_value_turns', 'learner') == 'both':
                        game['samples'].append((encode(position, game['history'][key(position)],
                                                       (cfg['max_plies'] - game['ply']) / cfg['max_plies']),
                                                None, position['turn']))
                moves.append({'id': game['id'], 'position': position, 'move': move})
            states = bridge.ask(op='advance', moves=moves)['states']
            for game, move, next_state in zip(remaining, moves, states):
                game['state'] = next_state
                game['moves'].append(move['move'])
                game['ply'] += 1
                k = key(next_state['position'])
                game['history'][k] = game['history'].get(k, 0) + 1
                display.moves += 1
            checkpoint()
        checkpoint(force=True)
        display.show(state, device, 'Paused safely.' if STOP else 'Experiment complete.', force=True)
        for mode in MODES:
            atomic_save({'version': VERSION, 'model': models[mode].state_dict(), 'config': cfg,
                         'mode': mode, 'iteration': state['iteration']}, directory / f'{mode}.pt')
        performance = dict(state['performance'])
        performance['checkpoint_seconds'] = previous_stats.get('checkpoint_seconds', 0) + checkpoint_seconds
        performance['checkpoints'] = previous_stats.get('checkpoints', 0) + checkpoint_count
        (directory / 'performance.json').write_text(json.dumps(performance, indent=2) + '\n')
    except BaseException:
        status.write(state, device, display.moves, 'Training stopped with an error', 'failed', force=True)
        raise
    else:
        status.write(state, device, display.moves,
                     'Paused safely' if STOP else 'Experiment complete',
                     'paused' if STOP else 'complete', force=True)
    finally:
        bridge.close()
    print_summary(state)
    print(f"\n{'Paused' if STOP else 'Finished'}: {directory}\nResume: .venv/bin/python training/train.py --resume {directory}")


def migrate(state):
    """v1 checkpoints retain their game, replay, and RNG; new runs use batching."""
    if state['version'] not in (1, VERSION):
        raise ValueError('Unsupported checkpoint version')
    state['config'].setdefault('parallel_games', 1)
    state['config'].setdefault('workers', 1)
    state['config'].setdefault('architecture', 'small')
    state['config'].setdefault('mps_inference', 'bucketed')
    state['config'].setdefault('engine', 'typescript')
    state['config'].setdefault('checkpoint_seconds', 15.0)
    # Preserve the objective of older runs; new runs opt into both sides.
    state['config'].setdefault('negamax_value_turns', 'learner')
    state['config'].setdefault('replay_sampling', 'uniform')
    state['config'].setdefault('game_scheduler', 'cohort')
    state['config'].setdefault('search_budget', 'fixed')
    state['config'].setdefault('fast_simulations', 16)
    state['config'].setdefault('full_search_probability', 0.25)
    if 'active_games' not in state:
        game = state.get('game')
        if game:
            game['index'] = state['game_index']
            game['id'] = f"{state['mode']}:{state['phase']}:{state['iteration']}:{state['game_index']}"
        state['active_games'] = [game] if game else []
    state['version'] = VERSION



def print_summary(state):
    print('\nEvaluation vs negamax (monitor/heldout fixed; promotion adaptive; wins / cutoffs / losses; score):')
    for row in summarize(state['records']):
        if row['phase'] == 'eval':
            print(f"  {row['mode']:8s} {row['suite']:7s} iteration {row['iteration']:3d}: "
                  f"{row['wins']} / {row['cutoffs']} / {row['losses']}; {row['score']:.1%}")
    print('Cutoffs receive half-score for monitoring and zero value for training; they are not Quoridor draws.')



def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, help='New run directory (must not already contain a checkpoint)')
    parser.add_argument('--resume', type=Path, help='Run directory; all training settings are restored')
    parser.add_argument('--engine', choices=['typescript', 'native', 'native-negamax', 'direct'], help='Training rules/search backend (default typescript)')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--plain', action='store_true', help='Line-oriented output instead of live dashboard')
    parser.add_argument('--iterations', type=int, help='Total iterations; default 2, may extend a resumed run')
    parser.add_argument('--games', type=int, default=2, help='Training games per arm per iteration')
    parser.add_argument('--eval-games', type=int, default=20, help='Even number of monitoring games (color pairs)')
    parser.add_argument('--simulations', type=int, default=64)
    parser.add_argument('--architecture', choices=ARCHITECTURES, default='small',
                        help='Network and board representation variant; saved on resume')
    parser.add_argument('--training-repetition-limit', type=int, choices=[2, 3], default=3,
                        help='Training rollout cutoff at second/third occurrence; evaluation and search retain threefold')
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32',
                        help='CUDA forward precision; BF16 requires supported GPU; saved on resume')
    parser.add_argument('--search-budget', choices=['fixed', 'randomized'], default='fixed')
    parser.add_argument('--fast-simulations', type=int, default=16, help='Shallow training search budget; value targets only')
    parser.add_argument('--full-search-probability', type=float, default=0.25)
    parser.add_argument('--eval-every', type=int, default=10, help='Evaluate at baseline, every N iterations, and final iteration')
    parser.add_argument('--heldout-games', type=int, default=40, help='Additional final-only games on independent paired openings; 0 disables')
    parser.add_argument('--walls', type=int, default=10, help='Target wall inventory per player (0..10); also used in evaluation')
    parser.add_argument('--curriculum', choices=['races', 'gated', 'none'], default='races')
    parser.add_argument('--curriculum-interval', type=int, default=25, help='Iterations per stage: 0, 2, 5, then target walls')
    parser.add_argument('--curriculum-games', type=int, default=40, help='Paired games per arm per gated evaluation')
    parser.add_argument('--curriculum-win-rate', type=float, default=0.4, help='Goal-win fraction needed to pass current difficulty')
    parser.add_argument('--curriculum-streak', type=int, default=2, help='Consecutive passing evaluations before advancing')
    parser.add_argument('--negamax-nodes', type=int, default=2000)
    parser.add_argument('--negamax-value-turns', choices=['both', 'learner'], default='both',
                        help='Value targets on both players or learner only; policy targets remain learner-only')
    parser.add_argument('--adaptive-negamax', action='store_true', help='Promote training opponent using separate full-game evaluations')
    parser.add_argument('--negamax-start-nodes', type=int, default=32)
    parser.add_argument('--promotion-win-rate', type=float, default=0.6)
    parser.add_argument('--promotion-games', type=int, default=40, help='Even number of color-balanced games per evaluation')
    parser.add_argument('--promotion-streak', type=int, default=2, help='Consecutive successful evaluation suites before doubling nodes')
    parser.add_argument('--max-plies', type=int, default=100)
    parser.add_argument('--updates', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--replay-size', type=int, default=10000)
    parser.add_argument('--replay-sampling', choices=['uniform', 'task'], default='uniform',
                        help='Uniform samples or equal retention/minibatch shares per task')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--parallel-games', type=int, default=8, help='Concurrent games for neural inference batching')
    parser.add_argument('--game-scheduler', choices=['cohort', 'rolling'], default='rolling',
                        help='Refill completed game slots (rolling) or wait for the whole cohort')
    parser.add_argument('--mps-inference', choices=['eager', 'bucketed', 'hybrid'],
                        help='Direct-engine MPS inference: stable GPU batches or architecture-specific hybrid CPU cutoff (default bucketed)')
    parser.add_argument('--workers', type=int, help='CPU search workers: processes for bridge, threads for direct (default 1); can change on resume')
    parser.add_argument('--checkpoint-seconds', type=float, help='Periodic checkpoint interval (default 15 seconds); always save on Ctrl-C')
    args = parser.parse_args()
    if args.resume and args.out:
        parser.error('--resume and --out are mutually exclusive')
    supplied = {item.split('=')[0] for item in sys.argv[1:] if item.startswith('--')}
    if args.resume:
        changed = supplied - {'--resume', '--device', '--plain', '--iterations', '--checkpoint-seconds', '--engine', '--workers', '--mps-inference'}
        if changed:
            parser.error('Resume restores settings; do not override: ' + ', '.join(changed))
    if args.fast_simulations < 1 or not 0 < args.full_search_probability <= 1:
        parser.error('Fast simulations must be positive and full-search probability in (0, 1]')
    if args.search_budget == 'randomized' and args.fast_simulations >= args.simulations:
        parser.error('Randomized search requires --fast-simulations < --simulations')
    for name in ('iterations', 'games', 'eval_games', 'simulations', 'negamax_nodes', 'max_plies', 'updates', 'batch_size', 'replay_size', 'parallel_games', 'workers', 'eval_every', 'curriculum_interval'):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    if args.negamax_start_nodes < 1 or args.promotion_games < 2 or args.promotion_games % 2 or args.promotion_streak < 1:
        parser.error('Starting nodes and streak must be positive; promotion games must be positive and even')
    if args.curriculum_games < 2 or args.curriculum_games % 2 or args.curriculum_streak < 1:
        parser.error('Curriculum games must be positive and even; streak must be positive')
    if not 0 < args.curriculum_win_rate <= 1:
        parser.error('--curriculum-win-rate must be in (0, 1]')
    if args.replay_sampling == 'task' and args.replay_size < 5:
        parser.error('Task-balanced replay requires --replay-size >= 5')
    if not 0 < args.promotion_win_rate <= 1:
        parser.error('--promotion-win-rate must be in (0, 1]')
    if args.adaptive_negamax and args.negamax_start_nodes > args.negamax_nodes:
        parser.error('--negamax-start-nodes cannot exceed --negamax-nodes')
    if args.checkpoint_seconds is not None and (not 0 < args.checkpoint_seconds < float('inf')):
        parser.error('--checkpoint-seconds must be finite and positive')
    if not 0 <= args.walls <= 10:
        parser.error('--walls must be between 0 and 10')
    if args.heldout_games < 0 or args.heldout_games % 2:
        parser.error('--heldout-games must be nonnegative and even')
    if args.eval_games % 2 or args.games % 2:
        parser.error('--games and --eval-games must be even to balance colors')
    return args


def main():
    args = arguments()
    torch.set_num_threads(1)
    directory = args.resume or args.out or Path('training/runs') / datetime.now().strftime('%Y%m%d-%H%M%S')
    directory = directory.resolve()
    if args.resume and not (directory / 'latest.pt').is_file():
        raise SystemExit(f'No checkpoint in {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    # Advisory lock is released by the OS even on crashes; never run two writers on one run.
    import fcntl
    with (directory / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f'Another trainer is using {directory}')
        if not args.resume and (directory / 'latest.pt').exists():
            raise SystemExit('Run already exists; use --resume or a different --out directory')
        rng = random.Random(args.seed)
        if args.resume:
            state = torch.load(directory / 'latest.pt', map_location='cpu', weights_only=True)
            migrate(state)
            if args.mps_inference is not None:
                state['config']['mps_inference'] = args.mps_inference
            if args.workers is not None:
                state['config']['workers'] = args.workers
            if args.engine is not None:
                state['config']['engine'] = args.engine
            if args.checkpoint_seconds is not None:
                state['config']['checkpoint_seconds'] = args.checkpoint_seconds
            if args.iterations is not None:
                if args.iterations < state['config']['iterations']:
                    raise SystemExit('--iterations may extend the saved budget, but cannot reduce it')
                state['config']['iterations'] = args.iterations
            if state['phase'] == 'done':
                if state['iteration'] >= state['config']['iterations']:
                    for mode in MODES:
                        atomic_save({'version': VERSION, 'model': state['arms'][mode]['model'],
                                     'config': state['config'], 'mode': mode, 'iteration': state['iteration']},
                                    directory / f'{mode}.pt')
                    print_summary(state)
                    print(f'Run already complete: {directory}. Use --iterations N to extend it.')
                    return
                state.update(iteration=state['iteration'] + 1, mode='selfplay', phase='train', game_index=0)
        else:
            args.varied_eval = True
            args.engine = args.engine or 'typescript'
            args.workers = args.workers or 1
            args.mps_inference = args.mps_inference or 'bucketed'
            args.iterations = args.iterations or 2
            args.checkpoint_seconds = args.checkpoint_seconds or 15.0
            torch.manual_seed(args.seed)
            cfg = {k: v for k, v in vars(args).items() if k not in ('resume', 'out', 'device', 'plain')}
            initial = model_from_config(cfg).state_dict()
            state = {'version': VERSION, 'config': cfg, 'created': datetime.now(timezone.utc).isoformat(),
                     'mode': 'selfplay', 'phase': 'eval', 'iteration': 0, 'game_index': 0,
                     'update_index': 0, 'game': None, 'active_games': [], 'records': [],
                     'arms': {m: {'model': copy.deepcopy(initial), 'optimizer': None, 'replay': [], 'loss': None} for m in MODES}}
            (directory / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
        if state['config'].get('adaptive_negamax'):
            progression(state)
        device, measurements = select_device(args.device, min(state['config']['parallel_games'], state['config']['games']),
                                             state.get('selected_device') if args.resume else None,
                                             model_config=state['config'])
        if device == 'cuda' and state['config'].get('precision') == 'bf16' and not torch.cuda.is_bf16_supported():
            raise SystemExit('This GPU does not support BF16; create an FP32 run instead.')
        state['selected_device'] = device
        if measurements:
            (directory / 'device-benchmark.json').write_text(json.dumps(measurements, indent=2) + '\n')
        models, optimizers = {}, {}
        for mode in MODES:
            models[mode], optimizers[mode] = new_arm(state['arms'][mode]['model'], device, state['config'])
            if state['arms'][mode]['optimizer']:
                optimizers[mode].load_state_dict(state['arms'][mode]['optimizer'])
        if args.resume:
            rng.setstate(state['rng'])
            torch.set_rng_state(state['torch_rng'])
            if device == 'mps' and state.get('mps_rng') is not None:
                torch.mps.set_rng_state(state['mps_rng'])
            if device == 'cuda' and state.get('cuda_rng') is not None:
                torch.cuda.set_rng_state(state['cuda_rng'])
        signal.signal(signal.SIGINT, stop_requested)
        signal.signal(signal.SIGTERM, stop_requested)
        save(state, models, optimizers, rng, directory, device)
        run(state, models, optimizers, rng, directory, device, args.plain)


if __name__ == '__main__':
    main()
