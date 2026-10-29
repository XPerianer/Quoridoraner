from contextlib import redirect_stdout
import io
import os
import unittest
from unittest.mock import patch

from train import ANSI, Dashboard


class TerminalDashboardTests(unittest.TestCase):
    def test_tasks_and_optimizer_progress(self):
        cfg = dict(games=256, updates=8, iterations=2000, max_plies=100,
                   replay_size=10000, batch_size=256, parallel_games=128)
        state = dict(config=cfg, mode='selfplay', phase='update', iteration=204,
                     update_index=3, game_index=256, records=[], active_games=[],
                     arms={m: dict(replay=[], loss=None) for m in ('selfplay', 'negamax')})
        display = Dashboard(plain=True)
        output = io.StringIO()
        with redirect_stdout(output):
            display.show(state, 'mps', force=True)
        text = ANSI.sub('', output.getvalue())
        self.assertIn('Current task: Optimize policy & value network', text)
        self.assertIn('3/8 updates (38%)', text)
        self.assertNotIn('256/256', text)
        self.assertIn('Replay 0/10,000', text)
        display.tty = True
        state['phase'] = 'train'
        state['game_index'] = 1
        state['active_games'] = [dict(task='race-1', ply=4, history={},
            state=dict(winner=-1, position=dict(pawns=[76, 4], wallsLeft=[2, 2],
                                               hWalls=[], vWalls=[], turn=0)))]
        output = io.StringIO()
        with patch('train.shutil.get_terminal_size', return_value=os.terminal_size((110, 40))), redirect_stdout(output):
            display.show(state, 'mps', force=True)
        text = ANSI.sub('', output.getvalue())
        self.assertIn('Active tasks: race-1: 1', text)
        self.assertIn('ply 4.0 mean (4–4)', text)
        self.assertIn('LOSS CURVES', text)
        self.assertIn('iteration 204/2000', text)
        self.assertIn('iter ETA', text)
        self.assertIn('loss (p / v)', text)

    def test_arm_trends_and_narrow_layout(self):
        cfg = dict(games=256, updates=32, iterations=2000, max_plies=100,
                   replay_size=10000, batch_size=256, parallel_games=128,
                   curriculum='gated', curriculum_streak=2, checkpoint_seconds=60)
        records = [dict(mode='selfplay', phase='train', iteration=77, game=i, winner=0,
                        learner=0, score=1.0, termination='goal', plies=2, suite='monitor',
                        task='race-2') for i in range(128)]
        records += [dict(mode='selfplay', phase='eval', iteration=70, game=i, winner=1,
                         learner=0, score=0.0, termination='goal', plies=2, suite='monitor',
                         task='standard') for i in range(20)]
        history = [dict(mode=mode, iteration=1, update=i, policy=1.4 - i * .001, value=.02)
                   for mode in ('selfplay', 'negamax') for i in range(1, 10)]
        state = dict(config=cfg, mode='selfplay', phase='train', iteration=77,
                     update_index=9, game_index=157, records=records, loss_history=history,
                     active_games=[dict(task='race-1', ply=4, suite='monitor', history={},
                         state=dict(winner=-1, position=dict(pawns=[76, 4], wallsLeft=[2, 2],
                                                            hWalls=[], vWalls=[], turn=0)))],
                     arms={m: dict(replay=[1] * 10000, loss=dict(policy=1.3, value=.02))
                           for m in ('selfplay', 'negamax')})
        display = Dashboard(plain=False)
        display.tty = True
        output = io.StringIO()
        with patch('train.shutil.get_terminal_size', return_value=os.terminal_size((110, 40))), redirect_stdout(output):
            display.show(state, 'mps', force=True)
        text = ANSI.sub('', output.getvalue())
        self.assertIn('loss (p / v)', text)
        self.assertIn('▼', text)
        self.assertIn('p█', text)
        self.assertIn('v▄', text)
        self.assertIn('iter 70', text)
        self.assertIn('Curriculum race-0', text)
        output = io.StringIO()
        with patch('train.shutil.get_terminal_size', return_value=os.terminal_size((52, 24))), redirect_stdout(output):
            display.show(state, 'mps', force=True)
        self.assertIn('selfplay', ANSI.sub('', output.getvalue()))
