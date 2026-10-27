"""Small, atomic telemetry snapshots, independent of expensive checkpoints."""
from collections import Counter
import json
import os
import time

from curriculum import evaluation_target


class StatusWriter:
    def __init__(self, directory):
        self.path = directory / 'status.json'
        self.started = time.monotonic()
        self.last = 0

    def write(self, state, device, moves=0, detail='', lifecycle='running', force=False):
        now = time.monotonic()
        if not force and now - self.last < 2:
            return
        cfg, phase = state['config'], state['phase']
        games = state.get('active_games', [])
        elapsed = now - self.started
        target = (cfg['updates'] if phase == 'update' else
                  evaluation_target(state) if phase == 'eval' else cfg['games'])
        completed = state['update_index'] if phase == 'update' else state['game_index']
        snapshot = dict(
            updated=time.time(), lifecycle=lifecycle, mode=state['mode'], phase=phase,
            iteration=state['iteration'], iterations=cfg['iterations'], detail=detail,
            completed=completed, target=target, unit='updates' if phase == 'update' else 'games',
            device=device, session_seconds=elapsed, moves=moves,
            moves_per_second=moves / max(elapsed, .001), active_games=len(games),
            tasks=dict(Counter(g.get('task', 'standard') for g in games)),
            suites=dict(Counter(g.get('suite', 'monitor') for g in games)),
            mean_ply=sum(g['ply'] for g in games) / len(games) if games else None,
            replay={m: len(a['replay']) for m, a in state['arms'].items()},
            losses={m: a['loss'] for m, a in state['arms'].items()},
            curriculum=state.get('curriculum_progress'), opponent=state.get('adaptive'))
        temp = self.path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(snapshot, allow_nan=False) + '\n')
        os.replace(temp, self.path)
        self.last = now
