import os
from . import constants
from . import param
from . import util
from .policy import Policy
from .history import History

def get_config(desired_accuracy):
    # TODO: Make this more configurable.
    config = dict()
    config.update(constants.DEFAULT_CONFIG)

    config['reward_tp'] = 1
    config['reward_tn'] = 1
    p = util.get_penalty(desired_accuracy, reward=1)
    config['penalty_fp'] = p
    config['penalty_fn'] = p
    config['desired_accuracy'] = desired_accuracy
    return config


def get_action(desired_accuracy, work_history, resolve=False):
    """Get action from policy."""
    config = get_config(desired_accuracy)

    # Convert history.
    history = History()
    history.new_worker()
    for v in work_history:
        if v is None:
            history.record(
                action=constants.WORK,
                observation=constants.O_NULL,
            )
        elif not v:
            history.record(
                action=constants.TEST,
                observation=constants.O_WRONG,
            )
        else:
            history.record(
                action=constants.TEST,
                observation=constants.O_RIGHT,
            )

    params = param.Params.from_cmd(config)

    pol = Policy(
        policy_type='zmdp',
        n_worker_classes=params.n_classes,
        params_gt=params.get_param_dict(sample=False),
    )
    pomdp_dirpath = os.path.join(
        os.path.dirname(__file__), 'models',
    )
    policy_dirpath = os.path.join(
        os.path.dirname(__file__), 'policies',
    )
    for d in [pomdp_dirpath, policy_dirpath]:
        util.ensure_dir(d)
    pomdp_fpath = os.path.join(
        pomdp_dirpath, 'sample.pomdp',
    )
    policy_fpath = os.path.join(
        policy_dirpath, 'sample.policy',
    )
    pol.prep_worker(
        model_filepath=pomdp_fpath,
        policy_filepath=policy_fpath,
        history=history,
        resolve_p=resolve,
    )
    belief = pol.model.get_start_belief()
    for a, o, _ in history.history[-1]:
        pol.model.update_belief(belief, a, o)
    a = pol.get_best_action(
        history=history,
        belief=belief,
    )
    if a == constants.TEST:
        return {'test': True}
    elif a == constants.WORK:
        return {'test': False}
    elif a == constants.BOOT:
        return None
    return Exception('Unknown action')
