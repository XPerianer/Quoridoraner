import random
import unittest

from bridge import create_bridge
from core import key
from curriculum import (evaluation_opening, evaluation_target, training_position, summarize,
                        curriculum_state, curriculum_check, gate_opening)
from train import advance, fresh_game


class CurriculumTests(unittest.TestCase):
    def test_gated_starts_never_advance_with_time(self):
        cfg = dict(seed=1, walls=10, curriculum='gated')
        for stage in range(4):
            tasks = set()
            for i in range(128):
                p, task = training_position(cfg, 10000, i, stage)
                tasks.add(task)
                if stage < 3:
                    self.assertNotEqual(task, 'standard')
                    self.assertLessEqual(p['wallsLeft'][0], (0, 2, 5)[stage])
            self.assertIn(f'race-{stage}', tasks)
            self.assertEqual('standard' in tasks, stage == 3)

    def test_gate_promotes_only_complete_consecutive_goal_win_suites(self):
        cfg = dict(curriculum='gated', curriculum_games=4, curriculum_win_rate=.5, curriculum_streak=2)
        state = dict(config=cfg, mode='negamax', records=[], iteration=0)
        def suite(iteration, scores):
            state['iteration'] = iteration
            stage = curriculum_state(state)['stage']
            state['records'] += [dict(mode='negamax', phase='eval', suite='curriculum', iteration=iteration,
                                     task=f'race-{stage}', score=s, termination='repetition' if s == .5 else 'goal')
                                 for s in scores]
            curriculum_check(state)
        suite(0, [1] * 4)
        self.assertEqual(curriculum_state(state)['streak'], 0)
        suite(10, [1, 1, 0, 0])
        curriculum_check(state)  # Do not count a resumed evaluation twice.
        self.assertEqual(curriculum_state(state)['streak'], 1)
        suite(20, [.5] * 4)
        self.assertEqual(curriculum_state(state)['streak'], 0)
        suite(30, [1, 1, 0, 0])
        suite(40, [1, 1, 0, 0])
        self.assertEqual(curriculum_state(state)['stage'], 1)
        suite(50, [1, 1])  # Partial suites cannot promote.
        self.assertEqual(curriculum_state(state)['streak'], 0)
        state['mode'] = 'selfplay'
        self.assertEqual(curriculum_state(state)['stage'], 0)

    def test_gate_openings_are_paired_legal_and_separate(self):
        cfg = dict(seed=1, walls=10, curriculum='gated')
        for stage in range(4):
            for i in range(0, 40, 2):
                p = gate_opening(cfg, 10, i, stage)
                self.assertEqual(p, gate_opening(cfg, 10, i + 1, stage))
                self.assertEqual(self.bridge.ask(position=p)['winner'], -1)
                self.assertEqual(p['wallsLeft'], [(0, 2, 5, 10)[stage]] * 2)
            self.assertNotEqual(gate_opening(cfg, 10, 0, stage), gate_opening(cfg, 20, 0, stage))

    def test_gate_suite_coexists_with_final_heldout_and_opponent_promotion(self):
        cfg = dict(seed=1, walls=10, curriculum='gated', curriculum_games=4,
                   iterations=1, eval_games=2, heldout_games=2, adaptive_negamax=True,
                   promotion_games=2, negamax_start_nodes=32, negamax_nodes=2000)
        state = dict(config=cfg, iteration=1, mode='negamax', phase='eval')
        self.assertEqual(evaluation_target(state), 10)
        games = [fresh_game(self.bridge, state, i) for i in range(10)]
        self.assertEqual([g['suite'] for g in games],
                         ['monitor'] * 2 + ['heldout'] * 2 + ['promotion'] * 2 + ['curriculum'] * 4)
        self.assertEqual(games[-1]['opponent_nodes'], 2000)
        self.assertEqual(games[-1]['task'], 'race-0')

    @classmethod
    def setUpClass(cls):
        cls.bridge = create_bridge('direct', workers=1)

    @classmethod
    def tearDownClass(cls):
        cls.bridge.close()

    def test_curriculum_stages_are_legal_and_keep_target_games(self):
        config = dict(seed=1, walls=10, curriculum='races', curriculum_interval=25)
        for iteration, cap in [(1, 0), (26, 2), (51, 5), (76, 10)]:
            budgets = set()
            for index in range(128):
                position, task = training_position(config, iteration, index)
                self.assertEqual((position, task), training_position(config, iteration, index ^ 1))
                state = self.bridge.ask(position=position)
                self.assertEqual(state['winner'], -1)
                self.assertTrue(state['legal'])
                if task == 'standard':
                    self.assertEqual(position['pawns'], [76, 4])
                    self.assertEqual(position['wallsLeft'], [10, 10])
                else:
                    budgets.add(position['wallsLeft'][0])
                    self.assertLessEqual(position['wallsLeft'][0], cap)
            self.assertEqual(max(budgets), cap)
        config['walls'] = 2
        for index in range(128):
            position, _ = training_position(config, 100, index)
            self.assertLessEqual(max(position['wallsLeft']), 2)

    def test_paired_eval_is_varied_independent_and_uses_target_walls(self):
        cfg = dict(seed=1, walls=10, varied_eval=True)
        suites = {}
        for suite in ('monitor', 'heldout'):
            positions = set()
            for index in range(0, 40, 2):
                current, history = evaluation_opening(self.bridge, cfg, index, suite)
                self.assertEqual((current, history), evaluation_opening(self.bridge, cfg, index + 1, suite))
                self.assertEqual(current['winner'], -1)
                self.assertLess(max(history.values()), 3)
                p = current['position']
                self.assertEqual(sum(p['wallsLeft']) + len(p['hWalls']) + len(p['vWalls']), 20)
                positions.add(key(p))
            self.assertGreater(len(positions), 10)
            suites[suite] = positions
        self.assertFalse(suites['monitor'] & suites['heldout'])

    def test_legacy_openings_are_unchanged(self):
        for index in range(4):
            current = self.bridge.ask()
            history = {key(current['position']): 1}
            rng = random.Random(1 + 10000 + index // 2)
            for _ in range(4):
                move = rng.choice([m for m in current['legal'] if m < 81])
                current = self.bridge.ask(position=current['position'], move=move)
                k = key(current['position'])
                history[k] = history.get(k, 0) + 1
            self.assertEqual((current, history), evaluation_opening(self.bridge, {'seed': 1}, index))

    def test_eval_schedule_and_final_heldout(self):
        cfg = dict(iterations=3, eval_every=2, eval_games=20, heldout_games=40)
        state = dict(config=cfg, iteration=0, mode='selfplay', phase='eval')
        phases = []
        while state['phase'] != 'done':
            phases.append((state['mode'], state['phase'], state['iteration']))
            if state['phase'] == 'eval':
                self.assertEqual(evaluation_target(state), 60 if state['iteration'] == 3 else 20)
            advance(state)
        for mode in ('selfplay', 'negamax'):
            self.assertEqual([it for m, p, it in phases if m == mode and p == 'eval'], [0, 2, 3])
            self.assertEqual([it for m, p, it in phases if m == mode and p == 'update'], [1, 2, 3])

    def test_final_game_metadata_and_cutoff_reporting(self):
        state = dict(config=dict(seed=1, iterations=3, eval_games=20, heldout_games=40,
                                 walls=10, varied_eval=True), iteration=3, mode='selfplay', phase='eval')
        game = fresh_game(self.bridge, state, 20)
        self.assertEqual(game['suite'], 'heldout')
        records = [dict(mode='selfplay', phase='train', iteration=1, score=s,
                        termination=t, plies=10) for s, t in [(1., 'goal'), (.5, 'repetition'), (.5, 'maxPlies')]]
        row = summarize(records)[0]
        self.assertEqual(row['cutoffs'], 2)
        self.assertEqual(row['wins'], 1)
        self.assertEqual(row['repetition_rate'], 1/3)
        self.assertEqual(row['goal_rate'], 1/3)


if __name__ == '__main__':
    unittest.main()
