from . import gating
from .constants import DEFAULT_GATING_PARAMS

class Gate:
    def __init__(
            self,
            desired_accuracy=DEFAULT_GATING_PARAMS['desired_accuracy'],
            n_gold_sliding=DEFAULT_GATING_PARAMS['n_gold_sliding'],
            batch_size=DEFAULT_GATING_PARAMS['batch_size'],
            gold_per_batch=DEFAULT_GATING_PARAMS['gold_per_batch'],
            n_tutorial=DEFAULT_GATING_PARAMS['n_tutorial'],
            n_screening=DEFAULT_GATING_PARAMS['n_screening'],
            exponential_backoff=DEFAULT_GATING_PARAMS['exponential_backoff'],
            seed=None,
            test_policy=None,
    ):
        self.n_tutorial = n_tutorial
        self.n_screening = n_screening
        self.desired_accuracy = desired_accuracy
        self.n_gold_sliding = n_gold_sliding
        self.batch_size = batch_size
        self.gold_per_batch = gold_per_batch
        self.exponential_backoff = exponential_backoff
        self.seed = seed
        self.test_policy = test_policy

    def next(self, history, seed=None):
        tutorial = history.get('tutorial') or []
        screening = history.get('screening') or []
        work = history.get('work') or []
        gating_recommendation = gating.next_action(
            tutorial=tutorial,
            screening=screening,
            work=work,
            n_tutorial=self.n_tutorial,
            n_screening=self.n_screening,
            desired_accuracy=self.desired_accuracy,
            n_gold_sliding=self.n_gold_sliding,
            batch_size=self.batch_size,
            gold_per_batch=self.gold_per_batch,
            exponential_backoff=self.exponential_backoff,
            seed=seed if seed is None else self.seed,
        )
        if self.test_policy:
            raise NotImplementedError
        if (
                not self.test_policy
                or gating_recommendation and (
                    'tutorial' in gating_recommendation
                    or 'screening' in gating_recommendation
                ) or not gating._passes_screening(
                    screening=screening,
                    accuracy=self.desired_accuracy,
                )
        ):
            return gating_recommendation

        from . import guru
        return guru.get_action(
            desired_accuracy=self.desired_accuracy,
            work_history=work,
            resolve=True,  # TODO: Don't always resolve.
        )
