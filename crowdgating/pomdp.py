"""pomdp.py"""

from __future__ import division
import copy
import logging
import random
import xml.etree.ElementTree as ET
import numpy as np
from numpy import log
from scipy.misc import logsumexp
import scipy.stats as ss
from . import util
from . import work_learn_problem as wlp

from . import zmdp_util


def param_to_string(p):
    """Convert a param in tuple form to a string.

    >>> param_to_string(('p_learn', None))
    'p_learn'
    >>> param_to_string(('p_guess', 2))
    'p_guess_w2'
    >>> param_to_string((('p_s', 2), None))
    'p_s2'
    >>> param_to_string((('p_s', 2), 1))
    'p_s2_w1'

    """
    if not isinstance(p, tuple):
        return p

    if isinstance(p[0], tuple):
        name = '{}{}'.format(*p[0])
    else:
        name = p[0]
    return name if p[1] is None else '{}_w{}'.format(name, p[1])


class POMDPModel:
    """POMDP model"""

    def __init__(self, n_worker_classes, params, hyperparams=None,
                 estimate_all=False):
        """Initialize.

        Args:
            n_worker_classes:       Number of worker classes.
            params (dict):          Dictionary of parameters. Must contain
                                    keys for all parameters to be estimated.
                                    When a parameter needs a worker class
                                    specified, use None to indicate
                                    that the parameter be shared across
                                    worker classes. Do not use both None
                                    and worker classes for a given parameter.
            hyperparams (object):   HyperParams instance.
            estimate_all (bool):    Ignore parameter values that we can
                                    estimate.

        """
        self.n_skills = len(params['p_r'])
        self.n_question_types = len(params['p_1'])
        self.n_worker_classes = n_worker_classes
        self.actions = wlp.actions_all(n_skills=self.n_skills,
                                       n_question_types=self.n_question_types,
                                       tell=params.get('tell', False),
                                       exp=params.get('exp', False))
        self.states = wlp.states_all(
            n_skills=self.n_skills,
            n_worker_classes=self.n_worker_classes,
            n_question_types=self.n_question_types)
        self.observations = wlp.observations(
            n_question_types=self.n_question_types)
        # TODO: Change this to a list of estimated params, and
        # make params (p, ind) format.
        self.params_fixed = [
            'exp', 'tell', 'cost', 'cost_exp', 'cost_tell',
            'p_r', 'p_1', 'utility_type',
            'penalty_fp', 'penalty_fn', 'reward_tp', 'reward_tn',
            'dataset', 'desired_accuracy']
        if estimate_all:
            self.params = dict(
                (k, params[k] if k in self.params_fixed else None) for
                k in params)
        else:
            self.params = params
        self.hparams = None

        self.hyperparams = hyperparams

    def get_params_est(self):
        """Return subset of parameters that are estimated"""
        return dict((k, self.params[k]) for k in self.params if
                    k not in self.params_fixed)

    def get_param_version(self, s, k):
        """Get param version to use for worker class in this state.

        If state is terminal, and key is defined for multiple worker classes,
        arbitrarily select the param version for the first worker class.

        Args:
            s:  State index.
            k:  Key.

        """
        st = self.states[s]
        worker_class = 0 if st.term else st.worker_class
        if k in self.params:
            return k
        elif (k, None) in self.params:
            return (k, None)
        else:
            return (k, worker_class)

    def get_names(self):
        """Get mapping of state/action/observation indices to names"""
        rows = []
        for i, a in enumerate(self.actions):
            rows.append({'i': i,
                         'type': 'action',
                         's': str(a),
                         'uses_gold': a.uses_gold()})
        for i, s in enumerate(self.states):
            rows.append({'i': i,
                         'type': 'state',
                         's': str(s),
                         'uses_gold': None})
        for i, o in enumerate(self.observations):
            rows.append({'i': i,
                         'type': 'observation',
                         's': str(o),
                         'uses_gold': None})
        return rows

    def write_txt(self, fo):
        """Write model to file as needed for AI-Toolbox."""
        for s, _ in enumerate(self.states):
            for a, _ in enumerate(self.actions):
                for s1, _ in enumerate(self.states):
                    fo.write('{}\t{}\t'.format(self.get_transition(s, a, s1),
                                               sum(self.get_reward(s, a, s1)[0])))
            fo.write('\n')
        for s, _ in enumerate(self.states):
            for a, _ in enumerate(self.actions):
                for o, _ in enumerate(self.observations):
                    fo.write('{}\t'.format(self.get_observation(s, a, o)))
            fo.write('\n')

    def write_pomdp(self, fo, discount):
        """Write a Cassandra-style POMDP spec with the given discount"""
        if discount >= 1.0:
            raise Exception('Discount must be less than 1.0')

        # Write header
        fo.write('discount: {}\n'.format(discount))
        fo.write('values: reward\n')
        fo.write('states: {}\n'.format(' '.join(str(s) for s in self.states)))
        fo.write('actions: {}\n'.format(' '.join(str(a) for
                                                 a in self.actions)))
        fo.write('observations: {}\n'.format(' '.join(self.observations)))

        fo.write('start: {}\n'.format(' '.join(
            str(x) for x in self.get_start_belief())))

        fo.write('\n\n### Transitions\n')
        for s, st in enumerate(self.states):
            for a, act in enumerate(self.actions):
                prob_sum = 0
                for s1, st1 in enumerate(self.states):
                    prob = self.get_transition(s, a, s1)
                    fo.write('T: {} : {} : {} {}\n'.format(
                        act, st, st1, prob))
                    prob_sum += prob
                if not np.isclose(1.0, prob_sum):
                    raise Exception("Transitions sum to {} for s:{}, a:{}".format(prob_sum, st, act))
                fo.write('\n')

        fo.write('\n\n### Observations\n')
        for s, st in enumerate(self.states):
            for a, act in enumerate(self.actions):
                prob_sum = 0
                for o, obs in enumerate(self.observations):
                    prob = self.get_observation(s, a, o)
                    #print 's:{}, a:{}, o:{} :: {}'.format(st, act, obs, prob)
                    fo.write('O: {} : {} : {} {}\n'.format(
                        act, st, obs, prob))
                    prob_sum += prob
                if not np.isclose(1.0, prob_sum):
                    raise Exception("Observations sum to {} for s:{}, a:{}".format(prob_sum, st, act))
                fo.write('\n')

        fo.write('\n\n### Rewards\n')
        for s, st in enumerate(self.states):
            for a, act in enumerate(self.actions):
                for s1, st1 in enumerate(self.states):
                    fo.write('R: {} : {} : {} : * {}\n'.format(
                        act, st, st1, sum(self.get_reward(s, a, s1)[0])))
                fo.write('\n')

    def get_start_belief(self, params=None):
        if params is None:
            params = self.params
        return [self.get_start_probability(s, params) for
                s in range(len(self.states))]

    def get_start_probability(self, s, params=None, exponents=False):
        """Get start probability, or exponents for parameters.

        Args:
            s:          State index
            params:
            exponents:  Return dictionary with parameter exponents instead
                        of raw probability

        """
        if params is None and not exponents:
            params = self.params
        st = self.states[s]
        if st.term or st.is_quiz():
            return dict() if exponents else 0
        else:
            exponents_dict = dict()
            exponents_dict['p_worker'] = [
                1 if i == st.worker_class else 0 for
                i in range(self.n_worker_classes)]
            for i, v in enumerate(st.skills):
                k = self.get_param_version(s, ('p_s', i))
                exponents_dict[k] = [1, 0] if v else [0, 1]
            if exponents:
                return exponents_dict
            else:
                return np.prod([np.prod(np.power(params[p],
                                                 exponents_dict[p])) for
                                p in exponents_dict])

    def get_transition(self, s, a, s1, params=None, exponents=False):
        """Get transition probability, or derivative

        Args:
            s:          State index (starting)
            a:          Action index
            s2:         State index (ending)
            params:
            exponents:  Return dictionary with parameter exponents instead
                        of raw probability

        """
        if params is None and not exponents:
            params = self.params

        st = self.states[s]
        act = self.actions[a]
        st1 = self.states[s1]

        p_leave = self.get_param_version(s, 'p_leave')

        # Once in terminal state, stay in terminal state.
        if st.term and st1.term:
            return dict() if exponents else 1
        elif st.term:
            return dict() if exponents else 0

        if act.name == 'boot':
            # Booting takes to a new start state. (Changed 8/26/2015)
            # NOTE: No longer takes to terminal state.
            return self.get_start_probability(s1, params=params,
                                              exponents=exponents)
        elif (not st.term and not st1.term and
              st.worker_class != st1.worker_class):
            # Otherwise, can't switch worker classes.
            return dict() if exponents else 0
        elif act.name == 'ask':
            if st1.term:
                return {p_leave: [1, 0]} if exponents else params[p_leave][0]
            elif not st1.quiz_val == act.quiz_val or st.n_skills_learned(st1):
                return dict() if exponents else 0
            else:
                loseable_skills_lost = [
                    i for i in range(self.n_skills) if
                    st.has_skill(i) and not st1.has_skill(i)]
                loseable_skills_not_lost = [
                    i for i in range(self.n_skills) if
                    st.has_skill(i) and st1.has_skill(i)]
                if exponents:
                    return_val = {p_leave: [0, 1]}
                else:
                    return_val = params[p_leave][1]
                for sk in loseable_skills_lost:
                    p_lose = self.get_param_version(s, ('p_lose', sk))
                    if exponents:
                        return_val[p_lose] = [1, 0]
                    else:
                        return_val *= params[p_lose][0]
                for sk in loseable_skills_not_lost:
                    p_lose = self.get_param_version(s, ('p_lose', sk))
                    if exponents:
                        return_val[p_lose] = [0, 1]
                    else:
                        return_val *= params[p_lose][1]
                return return_val
        elif act.name == 'exp' and st.is_quiz() or act.name == 'tell':
            # Could learn skill (no chance of losing taught skill).
            # Might lose each other skill independently.
            # For multiple question types, assume teaching actions teach
            # all rules.
            if self.n_question_types > 1:
                skills_taught = range(self.n_skills)
                skills_not_taught = []
            else:
                quiz_val = act.quiz_val or st.quiz_val
                skills_taught = [quiz_val]
                skills_not_taught = [
                    x for x in range(self.n_skills) if x != quiz_val]
            if st1.term:
                return {p_leave: [1, 0]} if exponents else params[p_leave][0]
            elif (st1.is_quiz() or (
                    any(st.has_skill(x) and not st1.has_skill(x) for
                        x in skills_taught) or
                    any(x not in skills_taught for
                        x in st.skills_learned(st1)))):
                return dict() if exponents else 0
            else:
                p_learn_str = 'p_learn_{}'.format(act.name)
                learnable_skills_learned = [
                    i for i in skills_taught if
                    not st.has_skill(i) and st1.has_skill(i)]
                learnable_skills_not_learned = [
                    i for i in skills_taught if
                    not st.has_skill(i) and not st1.has_skill(i)]
                loseable_skills_lost = [
                    i for i in skills_not_taught if
                    st.has_skill(i) and not st1.has_skill(i)]
                loseable_skills_not_lost = [
                    i for i in skills_not_taught if
                    st.has_skill(i) and st1.has_skill(i)]
                if exponents:
                    return_val = {p_leave: [0, 1]}
                else:
                    return_val = params[p_leave][1]
                for sk in learnable_skills_learned:
                    p_learn = self.get_param_version(s, (p_learn_str, sk))
                    if exponents:
                        return_val[p_learn] = [1, 0]
                    else:
                        return_val *= params[p_learn][0]
                for sk in learnable_skills_not_learned:
                    p_learn = self.get_param_version(s, (p_learn_str, sk))
                    if exponents:
                        return_val[p_learn] = [0, 1]
                    else:
                        return_val *= params[p_learn][1]
                for sk in loseable_skills_lost:
                    p_lose = self.get_param_version(s, ('p_lose', sk))
                    if exponents:
                        return_val[p_lose] = [1, 0]
                    else:
                        return_val *= params[p_lose][0]
                for sk in loseable_skills_not_lost:
                    p_lose = self.get_param_version(s, ('p_lose', sk))
                    if exponents:
                        return_val[p_lose] = [0, 1]
                    else:
                        return_val *= params[p_lose][1]
                return return_val
        else:
            if s == s1:
                return dict() if exponents else 1
            else:
                return dict() if exponents else 0

    def get_reward(self, s, a, s1, params=None, sample=False):
        """Get cost, expected reward, and ameta data.

        Args:
            s:          State index (starting)
            a:          Action index
            s2:         State index (ending)
            params:
            sample:     Sample reward rather than expected reward.

        Returns:
            ((cost, reward), meta)

        """
        if params is None:
            params = self.params
        p_r = params['p_r']
        p_slip_question_types = []
        p_guess_question_types = []
        for i in range(self.n_question_types):
            p_slip_question_types.append(
                params[self.get_param_version(s1, ('p_slip', i))][0])
            p_guess_question_types.append(
                params[self.get_param_version(s1, ('p_guess', i))][0])
        p_1 = params['p_1']
        cost = params['cost']
        utility_type = params['utility_type']
        penalty_fp = params['penalty_fp'] if 'penalty_fp' in params else None
        penalty_fn = params['penalty_fn'] if 'penalty_fn' in params else None
        reward_tp = params['reward_tp'] if 'reward_tp' in params else None
        reward_tn = params['reward_tn'] if 'reward_tn' in params else None

        st = self.states[s]
        act = self.actions[a]
        st1 = self.states[s1]

        if not st.is_valid_action(act):
            return ((wlp.NINF, 0), None)
        elif st.term or st1.term:
            return ((0, 0), None)
        elif act.name == 'exp':
            cost_exp = params['cost_exp']
            return ((cost_exp, 0), None)
        elif act.name == 'tell':
            cost_tell = params['cost_tell']
            return ((cost_tell, 0), None)
        elif act.is_quiz():
            return ((cost, 0), None)
        elif act.name == 'ask':
            reward, meta = st1.rewards_ask(
                p_r=p_r,
                p_slip=p_slip_question_types,
                p_guess=p_guess_question_types,
                priors=p_1,
                utility_type=utility_type,
                penalty_fp=penalty_fp,
                penalty_fn=penalty_fn,
                reward_tp=reward_tp,
                reward_tn=reward_tn,
                sample=sample)
            return ((cost, reward), meta)
        elif act.name == 'boot':
            return ((0, 0), None)
        else:
            raise Exception('Unexpected action when defining rewards')

    def get_observation(self, s, a, o, params=None, exponents=False):
        """Get observation probability, or derivative

        Args:
            s:          State index (ending)
            a:          Action index
            o:          Observation string
            params:

        """
        if params is None and not exponents:
            params = self.params

        act = self.actions[a]
        st = self.states[s]
        obs = self.observations[o]

        p_slip_keys = []
        p_guess_keys = []
        for i in range(self.n_question_types):
            p_slip_keys.append(
                self.get_param_version(s, ('p_slip', i)))
            p_guess_keys.append(
                self.get_param_version(s, ('p_guess', i)))

        if st.term or act.name == 'boot':
            # Always know when we enter terminal state or boot.
            if obs == 'term':
                return dict() if exponents else 1
            else:
                return dict() if exponents else 0
        elif act.is_quiz() and obs in ['term', 'null']:
            return dict() if exponents else 0
        elif act.is_quiz():
            # Assume teaching actions ask questions that require only a
            # single skill.
            if self.n_question_types == 1 or self.n_skills == 1:
                p_r_gold_question_types = [[int(i == st.quiz_val) for i in range(self.n_skills)]] * self.n_question_types
            else:
                p_r_gold_question_types = np.eye(self.n_question_types)
            return_val = dict() if exponents else 1
            for p_r_gold, obs_char, p_slip, p_guess in zip(
                    p_r_gold_question_types, obs, p_slip_keys, p_guess_keys):
                has_skills = st.p_has_skills(p_r_gold) == 1
                if has_skills and obs_char == 'r':
                    # TODO: Try to add to existing value in case
                    # multiple question types use same p_slip / p_guess in
                    # the future.
                    if exponents:
                        return_val[p_slip] = [0, 1]
                    else:
                        return_val *= params[p_slip][1]
                elif has_skills:
                    if exponents:
                        return_val[p_slip] = [1, 0]
                    else:
                        return_val *= params[p_slip][0]
                elif obs_char == 'r':
                    if exponents:
                        return_val[p_guess] = [1, 0]
                    else:
                        return_val *= params[p_guess][0]
                else:
                    if exponents:
                        return_val[p_guess] = [0, 1]
                    else:
                        return_val *= params[p_guess][1]
            return return_val
        else:
            if obs == 'null':
                return dict() if exponents else 1
            else:
                return dict() if exponents else 0

    def make_tables(self, params):
        """Create model tables from parameters

        Returns:
            p_t (|S|.|A|.|S| array):        Transition probabilties
            p_o (|S|.|A|.|O| array):        Observation probabilities
            p_i (|S| array):        i       Initial belief
            rewards (|S|.|A|.|S| array):    Rewards
        """
        S = len(self.states)
        A = len(self.actions)
        O = len(self.observations)

        p_t = np.zeros((S, A, S))
        p_o = np.zeros((S, A, O))
        rewards = np.zeros((S, A, S))
        for s in range(S):
            for a in range(A):
                for s1 in range(S):
                    p_t[s][a][s1] = self.get_transition(s, a, s1, params)
                    rewards[s][a][s1] = sum(
                        self.get_reward(s, a, s1, params)[0])

                for o in range(O):
                    p_o[s][a][o] = self.get_observation(s, a, o, params)

        # Initial beliefs
        p_i = self.get_start_belief(params)

        return p_t, p_o, p_i, rewards

    def sample_SOR(self, state_num, action_num):
        '''
        Sample a next state, observation, and reward.

        state_num       int
        action_num      int
        Returns:
            s_prime:    New state.
            o_prime:    Observation.
            r:          (cost, reward) tuple.
            meta:       Meta data (labels).


        '''
        p_s_prime = [self.get_transition(state_num, action_num, s_num) for
                     s_num in range(len(self.states))]
        s_prime = np.random.choice(range(len(self.states)), p=p_s_prime)
        p_o_prime = [
            self.get_observation(s_prime, action_num, observation_num) for
            observation_num in range(len(self.observations))]
        o_prime = np.random.choice(range(len(self.observations)), p=p_o_prime)
        r, meta = self.get_reward(state_num, action_num, s_prime, sample=True)
        return s_prime, o_prime, r, meta

    def update_belief(self, prev_belief, action_num, observation_num):
        '''
        POMDPModel doesn't store beliefs, so this takes
        and returns a belief vector.

        prev_belief     numpy array
        action_num      int
        observation_num int
        return          numpy array
        '''
        b_new_nonnormalized = []
        for s_prime in range(len(self.states)):
            p_o_prime = self.get_observation(
                s_prime, action_num, observation_num)
            summation = 0.0
            for s in range(len(self.states)):
                p_s_prime = self.get_transition(s, action_num, s_prime)
                b_s = float(prev_belief[s])
                summation = summation + p_s_prime * b_s
            b_new_nonnormalized.append(p_o_prime * summation)

        # normalize
        b_new = []
        total = sum(b_new_nonnormalized)
        for b_s in b_new_nonnormalized:
            b_new.append(b_s / total)
        return np.array(b_new)

    def expected_sufficient_statistics(self, log_marginals,
                                       log_pairwise_marginals, history):
        """Make tables with expected sufficient statistics

        Args:
            log_marginals:          list of unnormalized log marginals
                                    (np.arrays of (|T+1| x |S|))
            log_pairwise_marginals: list of unnormalized log marginal pairs
                                    (np.arrays of (|T| x |S| x |S|))

        Returns:
            ess_t:  Expected sufficient statistics for transitions.
            ess_o:  Expected sufficient statistics for observations.
            ess_i:  Expected sufficient statistics for initial probabilities.
        """
        S = len(self.states)
        A = len(self.actions)
        O = len(self.observations)
        ess_t = np.zeros((S, A, S))
        ess_o = np.zeros((S, A, O))
        ess_i = np.zeros((S))
        for worker, m in enumerate(log_marginals):
            m_norm = np.exp(m - logsumexp(m, axis=1, keepdims=True))
            T = history.n_t(worker)
            for t in range(T):
                for s in range(S):
                    a, o, _ = history.history[worker][t]
                    ess_o[s][a][o] += m_norm[t + 1][s]
            ess_i += m_norm[0, :]

        for worker, pm in enumerate(log_pairwise_marginals):
            pm_norm = np.exp(pm - logsumexp(pm, axis=(1, 2), keepdims=True))
            T = history.n_t(worker)
            for t in range(T):
                a, o, _ = history.history[worker][t]
                for s in range(S):
                    for s1 in range(S):
                        ess_t[s][a][s1] += pm_norm[t][s][s1]
        return ess_t, ess_o, ess_i

    def get_unnormalized_marginals(self, params, history):
        """Estimate unnormalized marginals from provided model parameters
        
        Args:
            params:     
            history:    History object

        Returns:
            tuple(log_marginals, log_pairwise_marginals, log_likelihood):
                log_marginals:          list of unnormalized log marginals
                                        (np.arrays of (|T+1| x |S|))
                log_pairwise_marginals: list of unnormalized log marginal pairs
                                        (np.arrays of (|T| x |S| x |S|))
                log_likelihood:         Log-likelihood 
        
        """
        S = len(self.states)
        ll = 0
        log_marginals = []
        log_pairwise_marginals = []
        for worker_AO in history.history:
            T = len(worker_AO)
            if T == 0:
                continue

            # Forward-backward init.
            alpha = np.zeros((T + 1, S))
            beta = np.zeros((T + 1, S))
            for s in range(S):
                p_i = self.get_start_probability(s, params)
                alpha[0][s] = log(p_i)
                beta[T][s] = log(1.0)

            # Forward.
            for t in range(T):
                a, o, _ = worker_AO[t]
                for s1 in range(S):
                    v = []
                    for s0 in range(S):
                        p_t = self.get_transition(s0, a, s1, params)
                        p_o = self.get_observation(s1, a, o, params)
                        v.append(alpha[t][s0] + log(p_t) + log(p_o))
                    alpha[t + 1][s1] = logsumexp(v)

            # Backward.
            for t in reversed(range(T)):
                a, o, _ = worker_AO[t]
                for s0 in range(S):
                    v = []
                    for s1 in range(S):
                        p_t = self.get_transition(s0, a, s1, params)
                        p_o = self.get_observation(s1, a, o, params)
                        v.append(beta[t + 1][s1] + log(p_t) + log(p_o))
                    beta[t][s0] = logsumexp(v)

            log_marginals.append(alpha + beta)

            # Make pairwise marginals
            pm = np.zeros((T, S, S))
            for t in range(T):
                a, o, _ = worker_AO[t]
                for s in range(S):
                    for s1 in range(S):
                        p_t = self.get_transition(s, a, s1, params)
                        p_o = self.get_observation(s1, a, o, params)
                        pm[t][s][s1] = alpha[t][s] + log(p_t) + log(p_o) + \
                            beta[t + 1][s1]  # BUG: should this be s1 or s
            log_pairwise_marginals.append(pm)

            # Update likelihood
            ll += logsumexp(alpha[T, :])

        return log_marginals, log_pairwise_marginals, ll

    def estimate_E(self, history, params):
        """Get expected sufficient statistics"""
        logging.debug('Estimating E step')
        log_marginals, log_pairwise_marginals, ll = \
            self.get_unnormalized_marginals(params, history)
        ess_t, ess_o, ess_i = self.expected_sufficient_statistics(
            log_marginals, log_pairwise_marginals, history)

        # Add param likelihood.
        for p in params:
            ll += log(ss.dirichlet.pdf(params[p], self.hyperparams.p[p]))

        return ess_t, ess_o, ess_i, ll

    def estimate_M(self, ess_t, ess_o, ess_i):
        """Perform M step for EM."""
        logging.debug('Estimating M step')
        S = len(self.states)
        A = len(self.actions)
        O = len(self.observations)

        params = copy.deepcopy(self.hyperparams.p)
        for s in range(S):
            exponents = self.get_start_probability(s, exponents=True)
            for p in exponents:
                for i, v in enumerate(exponents[p]):
                    params[p][i] += ess_i[s] * v
            for a in range(A):
                for s1 in range(S):
                    exponents = self.get_transition(s, a, s1, exponents=True)
                    for p in exponents:
                        for i, v in enumerate(exponents[p]):
                            params[p][i] += ess_t[s][a][s1] * v
                for o in range(O):
                    exponents = self.get_observation(s, a, o, exponents=True)
                    for p in exponents:
                        for i, v in enumerate(exponents[p]):
                            params[p][i] += ess_o[s][a][o] * v
        map_estimate = dict((p, util.dirichlet_mode(params[p])) for
                            p in params)
        return map_estimate, params

    def estimate_once(self, history, random_init, ll_max_improv):
        """Run EM starting from a single initialization."""
        if random_init:
            params = dict()
            for p in self.params:
                if p not in self.params_fixed:
                    params[p] = np.random.dirichlet(self.hyperparams.p[p])
        else:
            params = dict((k, copy.copy(self.params[k])) for
                          k in self.params if k not in self.params_fixed)

        ess_t, ess_o, ess_i, ll = self.estimate_E(history, params)
        ll_improv = float('inf')
        t = 0
        #print 'EM step {}: {} ({})'.format(t, ll, ll_improv)
        #print params
        while (np.isnan(ll_improv) or ll_improv > ll_max_improv):
            t += 1
            params, hparams = self.estimate_M(ess_t, ess_o, ess_i)
            ess_t, ess_o, ess_i, ll_new = self.estimate_E(history, params)
            ll_improv = abs((ll_new - ll) / ll)
            ll = ll_new
            #print 'EM step {}: {} ({})'.format(t, ll, ll_improv)
            #print params

        return params, hparams, ll

    def estimate(self, history, last_params=True, random_restarts=1,
                 ll_max_improv=0.001):
        """Estimate parameters from history.

        Args:
            history: History object.
            last_params: Initialize from last parameter values.
            random_restarts: Number of random initializations to perform.
            ll_max_improv: Threshold of % log-likelihood improvement.

        Returns:
            ll_best: Final log likelihood
            params_best: Final parameters

        """
        params_best = None
        hparams_best = None
        ll_best = float('-inf')

        # Run EM.
        if last_params:
            params, hparams, ll = self.estimate_once(
                history, random_init=False, ll_max_improv=ll_max_improv)
            if ll > ll_best:
                params_best = params
                hparams_best = hparams
                ll_best = ll
        for i in range(random_restarts):
            params, hparams, ll = self.estimate_once(
                history, random_init=True, ll_max_improv=ll_max_improv)
            if ll > ll_best:
                params_best = params
                hparams_best = hparams
                ll_best = ll

        self.params.update(params_best)
        self.hparams = hparams_best
        return ll_best, params_best

    def thompson_sample(self):
        """Reset self.params by sampling from self.hparams"""
        d = self.hparams
        for p in d:
            self.params[p] = np.random.dirichlet(d[p])


class POMDPPolicy:
    '''
    Based on mbforbes/py-pomdp on github.

    Read a policy file

    Attributes:
        action_nums    The full list of action (numbers) from the alpha
                       vectors. In other words, this saves the action
                       number from each alpha vector and nothing else,
                       but in the order of the alpha vectors.

        pMatrix        The policy matrix, constructed from all of the
                       alpha vectors.
    '''

    def __init__(self, filename, file_format='policyx', n_states=None):
        self.file_format = file_format
        if file_format == 'policyx':
            tree = ET.parse(filename)
            root = tree.getroot()
            avec = list(root)[0]
            alphas = list(avec)
            self.action_nums = []
            val_arrs = []
            for alpha in alphas:
                self.action_nums.append(int(alpha.attrib['action']))
                vals = []
                for val in alpha.text.split():
                    vals.append(float(val))
                val_arrs.append(vals)
            if len(val_arrs) == 0:
                raise Exception('APPL policy contained no alpha vectors')
            self.pMatrix = np.array(val_arrs)
        elif file_format == 'aitoolbox':
            # Retrieve max horizon alpha vectors.
            # TODO: Allow retrieval of horizons other than max.
            horizons = [[]]
            with open(filename, 'r') as f:
                for line in f:
                    if line.startswith('@'):
                        horizons.append([])
                    else:
                        horizons[-1].append(line)
            horizons = [lst for lst in horizons if len(lst) > 0]
            if len(horizons) == 0:
                raise Exception('AIToolbox policy contained no alpha vectors')
            lines_max_horizon = horizons[-1]
            alphas = [[float(v) for v in line.split()[:n_states]] for
                      line in lines_max_horizon]
            self.pMatrix = np.array(alphas)
            self.action_nums = [int(line.split()[n_states]) for
                                line in lines_max_horizon]
        elif file_format == 'zmdp':
            actions, alphas = zmdp_util.read_zmdp_policy(filename, n_states)
            self.action_nums = actions
            self.pMatrix = np.array(alphas)
        else:
            raise NotImplementedError

    def zmdp_filter(self, belief, alpha):
        """Return true iff this alpha vector applies to this belief"""
        return not any(b > 0 and a is None for b, a in zip(belief, alpha))

    def zmdp_convert(self, alpha):
        """Return new array with Nones replaced with 0's"""
        return [a if a is not None else 0 for a in alpha]

    def get_best_action(self, belief):
        '''
        Returns tuple:
            (best-action-num, expected-reward-for-this-action).
        '''
        """
        res = self.pMatrix.dot(belief)
        highest_expected_reward = res.max()
        best_action = self.action_nums[res.argmax()]
        return (best_action, highest_expected_reward)
        """
        raise NotImplementedError  # Untested.
        res = self.get_action_rewards(belief)
        max_reward = max(res.itervalues())
        best_action = random.choice([a for a in res if res[a] == max_reward])
        return (best_action, max_reward)

    def get_action_rewards(self, belief):
        '''
        Returns dictionary:
            action-num: max expected-reward.
        '''
        if self.file_format == 'zmdp':
            alpha_indices_relevant = [
                i for i, alpha in enumerate(self.pMatrix) if
                self.zmdp_filter(belief, alpha)]
            alphas = []
            actions = []
            for i in alpha_indices_relevant:
                alphas.append(self.zmdp_convert(self.pMatrix[i, :]))
                actions.append(self.action_nums[i])
            alphas = np.array(alphas)
        else:
            alphas = self.pMatrix
            actions = self.action_nums
        res = alphas.dot(belief)
        d = dict()
        for a, r in zip(actions, res):
            if a not in d:
                d[a] = r
            else:
                d[a] = max(d[a], r)
        return d


def main_estimate(tup):
    """Helper function for main."""
    i, history, model, model_name, bic_penalty = tup
    import numpy as np
    import random
    np.random.seed(i)
    random.seed(i)
    ll, _ = model.estimate(history, last_params=False, random_restarts=1,
                           ll_max_improv=0.001)
    params = model.params
    return {'ll': ll,
            'params': params,
            'model_name': model_name,
            'bic_penalty': bic_penalty}


def main():
    """Run passive simulator and estimate parameters.

    Tries all combinations of parameter sharing for given number of worker
    classes.

    """
    import argparse
    import itertools
    import multiprocessing
    import json
    import os
    from .history import History
    from .exp import add_config_argparse_group
    from . import simulator
    from . import param
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help='Experiment name')
    parser.add_argument('--config_json', type=argparse.FileType('r'))
    parser.add_argument('--restarts', type=int, default=50)
    parser.add_argument('--convert_work_to_quiz', action='store_true')
    parser.add_argument('--processes', type=int)
    add_config_argparse_group(parser)

    parser.add_argument(
        '--hyperparams', type=str, default='HyperParams',
        choices=param.HYPERPARAMS, help='Hyperparams class name, in param.py')
    args = parser.parse_args()
    args_vars = vars(args)
    if args.config_json is not None:
        config = json.load(args.config_json)
    else:
        config = dict()

    config_params = [
        'p_worker', 'exp', 'tell', 'cost', 'cost_exp', 'cost_tell',
        'p_lose', 'p_leave',
        'p_slip', 'p_slip_std', 'p_guess', 'p_r', 'p_1', 'p_s',
        'utility_type', 'dataset']
    if 'exp' not in config:
        config['exp'] = args.exp
    if 'tell' not in config:
        config['tell'] = args.tell
    if config['exp']:
        config_params.append('p_learn_exp')
    if config['tell']:
        config_params.append('p_learn_tell')
    if args.utility_type in ['pen', 'pen_diff', 'pen_nonboolean']:
        config_params += ['penalty_fp', 'penalty_fn', 'reward_tp', 'reward_tn']

    for k in config_params:
        if k not in config:
            config[k] = args_vars[k]

    params = param.Params.from_cmd(config)

    n_worker_classes = params.n_classes
    passive_simulator = simulator.LiveSimulator(params, repeat=False,
        random_workers=False, random_actions=False,
        convert_work_to_quiz=args.convert_work_to_quiz)
    history = History()
    while passive_simulator.worker_available():
        passive_simulator.new_worker()
        history.new_worker()
        while passive_simulator.worker_hired():
            a, _, o, _, _ = passive_simulator.sample_SOR(a=None)
            history.record(a, o)

    hyperparams_cls = getattr(param, args.hyperparams)
    model_names = []
    models_all = []
    bic_penalties = []
    result_dir = os.path.join(os.path.dirname(__file__), 'model_selection')
    util.ensure_dir(result_dir)
    param_types_to_vary = ['p_s', 'p_lose', 'p_slip', 'p_guess']
    if config['exp']:
        param_types_to_vary.append('p_learn_exp')
    if config['tell']:
        param_types_to_vary.append('p_learn_tell')
    for params_shared in itertools.product([True, False], repeat=len(param_types_to_vary)):
        model_name = '{}_classes'.format(params.n_classes)
        for param_type, shared in zip(param_types_to_vary, params_shared):
            if shared:
                params.set_shared(param_type)
                model_name += '-{}_shared'.format(param_type)
            else:
                params.set_not_shared(param_type)
        model_names.append(model_name)
        params_dict = params.get_param_dict(sample=False)
        model = POMDPModel(
            n_worker_classes=n_worker_classes, params=params_dict,
            hyperparams=hyperparams_cls(params_dict, n_worker_classes),
            estimate_all=True)
        models_all.append(model)


        # Calculate BIC.
        # See https://arxiv.org/pdf/1301.7374.pdf for details.
        start_complexity = params.get_model_complexity_start()
        transition_complexity = params.get_model_complexity_transition()
        n_sequences = history.n_workers()
        n_transitions = sum(history.n_t(worker) for worker in range(n_sequences))
        bic_penalties.append(0.5 * (log(n_sequences) * start_complexity +
                                    log(n_transitions) * transition_complexity))



    pool = multiprocessing.Pool(initializer=util.init_worker, processes=args.processes)
    import functools as ft
    f = ft.partial(util.run_functor, main_estimate)
    res = pool.map(f, itertools.chain(
        *[zip(range(args.restarts),
                       itertools.repeat(history),
                       itertools.repeat(model),
                       itertools.repeat(model_name),
                       itertools.repeat(bic_penalty)) for model, model_name, bic_penalty in zip(models_all, model_names, bic_penalties)]))
    import pandas as pd
    df = pd.DataFrame(res)
    df['bic_score'] = df['ll'] - df['bic_penalty']
    df['params_json'] = df['params'].map(lambda p: json.dumps(param.Params(p).to_cmd()))
    df.sort_values(by='bic_score', ascending=False).to_csv(
        os.path.join(result_dir, '{}.csv'.format(args.name)), index=False)

if __name__ == '__main__':
    main()
