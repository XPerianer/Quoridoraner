import unittest
from train import repetition_limit

class RepetitionLimitTests(unittest.TestCase):
    def test_legacy_defaults_and_evaluation_control(self):
        self.assertEqual(repetition_limit({}, 'train'), 3)
        self.assertEqual(repetition_limit({'training_repetition_limit': 2}, 'train'), 2)
        self.assertEqual(repetition_limit({'training_repetition_limit': 2}, 'eval'), 3)
