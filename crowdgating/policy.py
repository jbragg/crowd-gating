"""policy.py

Requirements: $PATH must include pomdpsol-appl for 'appl' policies and
pomdpsol-aitoolbox for 'aitoolbox' policies.

"""
from __future__ import division
import collections
import os
import time
import copy
import random
import math
import subprocess
import numpy as np
from .pomdp import POMDPPolicy, POMDPModel
from . import util
from .util import ensure_dir
from . import work_learn_problem as wlp
from . import param

ZMDP_ALIAS = os.environ.get('ZMDP_ALIAS', 'pomdpsol-zmdp')

class Policy:
    """Policy class

    Assumes policy files for appl policies live in relative folder 'policies'

    """
    def __init__(self, policy_type, n_worker_classes, params_gt,
                 **kwargs):
        #print 'Reinitializing policy'
        default_discount = 0.99
        self.policy = policy_type
        self.epsilon = kwargs.get('epsilon', None)
        self.explore_actions = kwargs.get('explore_actions', None)
        self.explore_policy = kwargs.get('explore_policy', None)
        self.thompson = bool(kwargs.get('thompson', False))
        self.hyperparams = kwargs.get('hyperparams', None)
        self.desired_accuracy = params_gt.get('desired_accuracy', None)

        if self.rl_p():
            name = kwargs['hyperparams']
            cls = getattr(param, name)
            self.model = POMDPModel(
                n_worker_classes, params=params_gt,
                hyperparams=cls(params_gt, n_worker_classes),
                estimate_all=True)
            if self.explore_policy is not None:
                self.explore_policy = Policy(
                    policy_type=self.explore_policy['type'],
                    n_worker_classes=n_worker_classes,
                    params_gt=params_gt,
                    **self.explore_policy)
        else:
            self.model = POMDPModel(n_worker_classes, params=params_gt)

        if self.policy in ('appl', 'zmdp'):
            self.discount = kwargs.get('discount', default_discount)
            self.timeout = kwargs.get('timeout', None)
        elif self.policy == 'aitoolbox':
            self.discount = kwargs.get('discount', default_discount)
            self.horizon = kwargs['horizon']
        elif self.policy == 'test_and_boot':
            self.teach_type = kwargs.get('teach_type', None)
            self.n_teach = kwargs.get('n_teach', 0)
            self.n_blocks = kwargs.get('n_blocks', None)
            if self.n_blocks != 0:
                self.n_test = kwargs['n_test']
                self.n_work = kwargs['n_work']
                self.accuracy = kwargs['accuracy']
                n_test_actions = len(
                    [a for a in self.model.actions if a.is_quiz()])
                self.accuracy_window = kwargs.get('accuracy_window', None)
                if self.accuracy_window is None:
                    self.accuracy_window = self.n_test * n_test_actions
            self.final_action = kwargs.get('final_action', 'work')
        elif self.policy != 'work_only':
            raise NotImplementedError

        self.params_estimated = dict()
        self.hparams_estimated = dict()
        self.estimate_times = dict()
        self.resolve_times = []
        self.external_policy = None
        self.use_explore_policy = False

    def rl_p(self):
        """Policy does reinforcement learning."""
        return self.epsilon is not None or self.thompson

    def get_epsilon_probability(self, worker, t, budget_frac):
        """Return probability specified by the given exploration function.

        Exploration function is a function of the worker (w or worker)
        the current timestep (t), and the fraction of the exploration
        budget (f or budget_frac).

        WARNING: Evaluates the expression in self.epsilon without security
        checks.

        """
        # Put some useful variable abbreviations in the namespace.
        w = worker
        f = budget_frac
        e = math.e
        if isinstance(self.epsilon, basestring):
            return eval(self.epsilon)
        else:
            return self.epsilon

    def prep_worker(self, model_filepath, policy_filepath, history,
                    resolve_p=False,
                    resolve_random_restarts=1,
                    previous_workers=None, explore=None):
        """Reestimate and resolve as needed.

        Args:
            history (.history.History): History of workers.
                IMPORTANT: Do not call .history.History.new_worker() before
                running this function or the worker count will be incorrect.
            resolve_p: Resolve or not.
            resolve_random_restarts (int): Number of random restarts to use
                when re-estimating model.
            previous_workers (Optional[int]): Number of previous workers.
                Defaults to one less than number of workers in history object.

        """
        t = 0
        self.use_explore_policy = False

        estimate_p = self.rl_p() and resolve_p
        model = self.model
        if estimate_p:
            start = time.clock()
            model.estimate(history=history,
                           last_params=(len(self.params_estimated) > 0),
                           random_restarts=resolve_random_restarts)
            if self.thompson:
                model.thompson_sample()
            self.estimate_times[worker] = time.clock() - start
            self.params_estimated[worker] = copy.deepcopy(
                model.get_params_est())
            self.hparams_estimated[worker] = copy.deepcopy(model.hparams)
        if resolve_p:
            utime1, stime1, cutime1, cstime1, _ = os.times()
            self.external_policy = self.run_solver(
                model_filepath=model_filepath, policy_filepath=policy_filepath)
            utime2, stime2, cutime2, cstime2, _ = os.times()
            # All solvers are run as subprocesses, so count elapsed
            # child process time.
            self.resolve_times.append(cutime2 - cutime1 + \
                                      cstime2 - cstime1)

    def get_best_action(self, history, belief=None):
        """Get best action according to policy.

        If policy requires an external_policy, assumes it already exists.

        self.n_blocks should be None unless teaching actions disabled.

        Accuracy for test_and_boot policy is averaged across question
        types.

        Args:
            history (History object):   Defined in history.py.

        Returns: Action index.

        """
        valid_actions = self.get_valid_actions(history)
        model = self.model
        a_ask = model.actions.index(wlp.Action('ask'))
        a_boot = model.actions.index(wlp.Action('boot'))
        worker = history.n_workers() - 1
        current_AO = history.history[-1]
        if len(current_AO) == 0:
            current_actions = []
            current_observations = []
        else:
            current_actions, current_observations, _ = zip(*current_AO)
        n_actions = len(current_actions)

        # Get POMDP policy.
        rewards = self.external_policy.get_action_rewards(belief)
        valid_actions_with_rewards = set(valid_actions).intersection(
            set(rewards))
        if len(valid_actions_with_rewards) == 0:
            raise Exception('No valid actions in policy')
        max_reward = max(rewards.values())
        valid_rewards = dict((a, rewards[a]) for a in valid_actions_with_rewards)
        max_valid_reward = max(valid_rewards.values())
        if max_reward > max_valid_reward:
            raise Exception('Warning: best reward not available')
        # Take random best action.
        best_valid_action = random.choice(
            [a for a in valid_rewards if
             valid_rewards[a] == max_valid_reward])
        return best_valid_action

    def run_solver(self, model_filepath, policy_filepath):
        """Run POMDP solver.

        Args:
            model_filepath (str):       Path for input to POMDP solver.
            policy_filepath (str):      Path for computed policy.

        Returns:
            policy (POMDPPolicy)

        """
        model = self.model
        if self.policy == 'appl':
            with open(model_filepath, 'w') as f:
                model.write_pomdp(f, discount=self.discount)
            args = ['pomdpsol-appl',
                    model_filepath,
                    '-o', policy_filepath]
            if self.timeout is not None:
                args += ['--timeout', str(self.timeout)]
            _ = subprocess.check_output(args)
            return POMDPPolicy(policy_filepath,
                               file_format='policyx')
        elif self.policy == 'aitoolbox':
            with open(model_filepath, 'w') as f:
                model.write_txt(f)
            args = ['pomdpsol-aitoolbox',
                    '--input', model_filepath,
                    '--output', policy_filepath,
                    '--discount', str(self.discount),
                    '--horizon', str(self.horizon),
                    '--n_states', str(len(model.states)),
                    '--n_actions', str(len(model.actions)),
                    '--n_observations', str(len(model.observations))]
            _ = subprocess.check_output(args)
            return POMDPPolicy(policy_filepath,
                               file_format='aitoolbox',
                               n_states=len(model.states))
        elif self.policy == 'zmdp':
            with open(model_filepath, 'w') as f:
                model.write_pomdp(f, discount=self.discount)
            args = [ZMDP_ALIAS,
                    'solve', model_filepath,
                    '-o', policy_filepath]
            if self.timeout is not None:
                args += ['-t', str(self.timeout)]
            _ = subprocess.check_output(args)
            return POMDPPolicy(policy_filepath,
                               file_format='zmdp',
                               n_states=len(model.states))


    def get_valid_actions(self, history):
        """Return valid action indices based on the history."""
        current_AO = history.history[-1]
        if len(current_AO) == 0:
            current_actions = []
            current_observations = []
        else:
            current_actions, current_observations, _ = zip(*current_AO)

        try:
            last_action = self.model.actions[current_actions[-1]]
        except IndexError:
            last_action = None
        return [i for i, a in enumerate(self.model.actions) if
                a.valid_after(last_action)]
