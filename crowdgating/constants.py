from . import work_learn_problem as wlp

_actions = wlp.actions_all(
    n_skills=1,
    n_question_types=1,
    tell=False,
    exp=False,
)
_observations = wlp.observations(
    n_question_types=1,
)

WORK = _actions.index(wlp.Action('ask'))
TEST = _actions.index(wlp.Action('ask', 0))
BOOT = _actions.index(wlp.Action('boot'))

O_TERM = _observations.index('term')
O_NULL = _observations.index('null')
O_RIGHT = _observations.index('r')
O_WRONG = _observations.index('w')


DEFAULT_CONFIG = {
    'zmdp_discount': 0.99,
    'zmdp_timeout': 600,
    'utility_type': 'pen',
    "cost": -0.000001,
    "p_guess": [
        0.5,
    ],
    "p_s": [
        1.0,
    ],
    'p_leave': [
        0.1,
    ],
    "p_slip": [
        0.1,
        0.4,
    ],
    'p_lose': [
        0.05,
        0.1,
    ],
    "p_1": [0.5],
    "p_worker": [
        0.8,
        0.2,
    ],
    "p_r": [
        1.0,
    ],
}

DEFAULT_GATING_PARAMS = {
    'n_gold_sliding': 10,
    'batch_size': 20,
    'desired_accuracy': 0.8,
    'gold_per_batch': 5,
    'exponential_backoff': True,
    'n_tutorial': 0,
    'n_screening': 0,
}
