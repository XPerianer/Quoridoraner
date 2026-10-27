import json
from pathlib import Path
import tempfile
import unittest

from status import StatusWriter


class StatusTests(unittest.TestCase):
    def test_phase_progress_and_terminal_states(self):
        state = dict(config=dict(updates=8, games=20, iterations=5, eval_games=4),
                     phase='train', mode='selfplay', iteration=1, game_index=3, update_index=2,
                     active_games=[dict(task='race-0', suite='monitor', ply=12)],
                     arms={'selfplay': dict(replay=[1, 2], loss=None)})
        with tempfile.TemporaryDirectory() as directory:
            writer = StatusWriter(Path(directory))
            def snapshot(**kwargs):
                writer.write(state, 'cpu', moves=10, force=True, **kwargs)
                return json.loads(writer.path.read_text())
            row = snapshot()
            self.assertEqual((row['completed'], row['target'], row['unit']), (3, 20, 'games'))
            self.assertEqual(row['tasks'], {'race-0': 1})
            self.assertEqual(row['mean_ply'], 12)
            self.assertEqual(row['replay']['selfplay'], 2)
            state['phase'] = 'update'
            state['active_games'] = []
            row = snapshot(lifecycle='paused')
            self.assertEqual((row['completed'], row['target'], row['unit']), (2, 8, 'updates'))
            self.assertEqual(row['lifecycle'], 'paused')
            self.assertIsNone(row['mean_ply'])
            state['phase'] = 'eval'
            self.assertEqual(snapshot()['target'], 4)
            state['phase'] = 'done'
            self.assertEqual(snapshot(lifecycle='complete')['lifecycle'], 'complete')
            self.assertFalse(writer.path.with_suffix('.json.tmp').exists())
