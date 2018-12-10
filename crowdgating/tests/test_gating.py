"""Test gating.

Assume tutorial of size 5 and screening of size 10.

"""
import os
import unittest
import json
from crowdgating import gating

PARAMS = {
    'n_tutorial': 5,
    'n_screening': 10,
    'desired_accuracy': 0.8,
    'n_gold_sliding': 10,
    'batch_size': 20,
    'desired_accuracy': 0.8,
    'gold_per_batch': 5,
    'exponential_backoff': True,
}

TUTORIAL_INCOMPLETE_1 = [
    1,
    1,
    0,
]

TUTORIAL_INCOMPLETE_2 = [
    1,
    1,
    1,
    1,
    0,
]


TUTORIAL_COMPLETE = [
    1,
    1,
    1,
    1,
    1,
]

SCREENING_INCOMPLETE = [
    0,
    1,
]

SCREENING_COMPLETE_FAIL = [
    0,
    1,
    1,
    1,
    1,
    0,
    1,
    1,
    0,
    1,
]

SCREENING_COMPLETE_PASS = [
    0,
    1,
    1,
    1,
    1,
    0,
    1,
    1,
    1,
    1,
]



class TutorialTest(unittest.TestCase):

    def setUp(self):
        self.params = {}
        self.params.update(PARAMS)

    def test_incomplete_1(self):
        action = gating.next_action(
            tutorial=TUTORIAL_INCOMPLETE_1,
            screening=None,
            work=None,
            **self.params
        )
        self.assertDictEqual(action, {'tutorial': 2})

    def test_incomplete_2(self):
        action = gating.next_action(
            tutorial=TUTORIAL_INCOMPLETE_2,
            screening=None,
            work=None,
            **self.params
        )
        self.assertDictEqual(action, {'tutorial': 4})

class ScreeningTest(unittest.TestCase):

    def setUp(self):
        self.params = {}
        self.params.update(PARAMS)

    def test_incomplete(self):
        action = gating.next_action(
            tutorial=TUTORIAL_COMPLETE,
            screening=SCREENING_INCOMPLETE,
            work=None,
            **self.params
        )
        self.assertDictEqual(action, {'screening': 2})

    def test_pass(self):
        action = gating.next_action(
            tutorial=TUTORIAL_COMPLETE,
            screening=SCREENING_COMPLETE_PASS,
            work=None,
            **self.params
        )
        self.assertIn('test', action)

    def test_fail(self):
        action = gating.next_action(
            tutorial=TUTORIAL_COMPLETE,
            screening=SCREENING_COMPLETE_FAIL,
            work=None,
            **self.params
        )
        self.assertIsNone(action)


class WorkTest(unittest.TestCase):
    def setUp(self):
        import random
        self.batch_1 = [True] * 5 + [False] * 15
        random.seed(0)
        random.shuffle(self.batch_1)

        self.batch_2 = [True] * 5 + [False] * 15
        random.seed(1)
        random.shuffle(self.batch_2)

        self.params = {}
        self.params.update(PARAMS)

    def test_first_batch(self):
        for i in range(20):
            action = gating.next_action(
                tutorial=TUTORIAL_COMPLETE,
                screening=SCREENING_COMPLETE_PASS,
                work=[1 if x else None for x in self.batch_1[:i]],
                seed=0,
                **self.params
            )
            self.assertEqual(action['test'], self.batch_1[i])

    def test_second_batch(self):
        for i in range(20):
            action = gating.next_action(
                tutorial=TUTORIAL_COMPLETE,
                screening=SCREENING_COMPLETE_PASS,
                work=[
                    1 if x else None
                    for x in self.batch_1 + self.batch_2[:i]
                ],
                seed=0,
                **self.params
            )
            self.assertEqual(action['test'], self.batch_2[i])

    def test_third_batch(self):
        for i in range(4):
            action = gating.next_action(
                tutorial=TUTORIAL_COMPLETE,
                screening=SCREENING_COMPLETE_PASS,
                work=[
                    1 if x else None
                    for x in self.batch_1 + self.batch_2
                ] + [0] * i,
                seed=0,
                **self.params
            )
            if i == 3:
                self.assertIsNone(action)
            else:
                self.assertFalse(action['test'])
