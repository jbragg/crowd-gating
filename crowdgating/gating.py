from __future__ import division
import math
import random
import doctest


def _is_power_two(n):
    return math.ceil(math.log(n, 2)) == math.floor(math.log(n, 2))


def next_tutorial_action(tutorial, size):
    """
    >>> next_tutorial_action([True], 2)
    1
    >>> next_tutorial_action([1, 1, 0], 4)
    2
    >>> next_tutorial_action([], 4)
    0
    >>> next_tutorial_action([1], 2)
    1
    >>> next_tutorial_action([1], 1)
    -1
    >>> next_tutorial_action([0], 1)
    0
    >>> next_tutorial_action([1, 0], 2)
    1
    """
    if tutorial and not tutorial[-1]:
        return len(tutorial) - 1
    elif len(tutorial) < size:
        return len(tutorial)
    return -1


def next_screening_action(screening, size):
    if len(screening) >= size:
        return None
    return len(screening)


def _passes_screening(screening, accuracy):
    return not screening or sum(screening) / len(screening) >= accuracy


def _passes_gold(work, n_gold_sliding, accuracy):
    """
    >>> _passes_gold([None, False, None, True, None, False], 2, 0.5)
    True
    >>> _passes_gold([None, False, None, True, None, False], 3, 0.5)
    False
    """
    min_right = math.ceil(accuracy * n_gold_sliding)
    tests = [x for x in work if x is not None][-n_gold_sliding:]
    return len(tests) < n_gold_sliding or sum(tests) >= min_right


def _is_test_batch(batch_index, exponential_backoff):
    if not exponential_backoff:
        return True
    return _is_power_two(batch_index + 1)


def should_test(
        screening, work, desired_accuracy, n_gold_sliding, batch_size,
        gold_per_batch, exponential_backoff, seed=None,
):
    if (
            not _passes_screening(
                screening=screening,
                accuracy=desired_accuracy,
            )
            or not _passes_gold(
                work=work,
                n_gold_sliding=n_gold_sliding,
                accuracy=desired_accuracy,
            )
    ):
        return None
    batch_index = int(len(work) / batch_size)

    test_batch = True
    if exponential_backoff:
        test_batch = _is_test_batch(
            batch_index=batch_index,
            exponential_backoff=exponential_backoff,
        )

    if not test_batch:
        return False

    gold = [True if i < gold_per_batch else False for i in range(batch_size)]
    if seed is not None:
        seed += batch_index
    random.seed(seed)
    random.shuffle(gold)

    n_current_batch = len(work) % batch_size
    return gold[n_current_batch]


def next_action(
        tutorial, screening, work, n_tutorial, n_screening, desired_accuracy,
        n_gold_sliding, batch_size, gold_per_batch, exponential_backoff,
        seed=None,
):
    tutorial_action = next_tutorial_action(
        tutorial=tutorial or [],
        size=n_tutorial,
    )
    if tutorial_action >= 0:
        return {'tutorial': tutorial_action}
    screening_action = next_screening_action(
        screening=screening or [],
        size=n_screening,
    )
    if screening_action is not None:
        return {'screening': screening_action}
    is_test_work_action = should_test(
        screening=screening or [],
        work=work or [],
        desired_accuracy=desired_accuracy,
        n_gold_sliding=n_gold_sliding,
        batch_size=batch_size,
        gold_per_batch=gold_per_batch,
        exponential_backoff=exponential_backoff,
        seed=seed,
    )
    if is_test_work_action is not None:
        return {'test': is_test_work_action}
    return None

if __name__ == "__main__":
    doctest.testmod()
