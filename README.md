# crowdgating

Provides an interface for deciding how to teach, screen, and test workers,
using Gated Instructions and POMDP-based adaptive testing algorithms.

## Relevant papers

### Gated instructions:
[Effective Crowd Annotation for Relation Extraction](https://www.cs.washington.edu/ai/pubs/liu-naacl16.pdf).
Angli Liu, Stephen Soderland, Jonathan Bragg, Christopher H. Lin, Xiao Ling, Daniel S. Weld.
NAACL 2016.

### Guru adaptive testing (not fully integrated here yet):
[Optimal Testing for Crowd Workers](https://www.cs.washington.edu/ai/pubs/bragg-aamas16.pdf).
Jonathan Bragg, Mausam, Daniel S. Weld.
AAMAS 2016.

## Installation

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Parameter definitions
- `n_tutorial` (int): Number of tutorial questions. Defaults to `0` except in web app.
- `n_screening` (int): Number of screening questions. Defaults to `0` except in web app.
- `desired_accuracy` (float): Desired accuracy (between `0` and `1`). Used as a gating threshold for both the screening questions and the inserted gold questions. Defaults to `0.8`.
- `n_gold_sliding` (int): Number of last inserted gold to use for removing workers. Defaults to `10`.
- `batch_size` (int): Size of a batch of work questions (not tutorial or screening). Defaults to `20`. 
- `gold_per_batch` (int): Number of gold questions to insert randomly into a batch. Defaults to `5`.
- `exponential_backoff` (bool): If `true`, ask gold questions only in exponentially decreasing batches (1, 2, 4, 8, ...). If `false`, ask gold questions in every batch. Defaults to `true`.
- `seed` (int): Random seed to use for gold question insertion. Should never change for a worker.

## Data format

A history should be formatted as follows.

Here is an illustrative example, where a worker
- has answered the two tutorial questions correctly,
- has answered all 3 screening, but got the last one wrong,
- and has worked on questions with unknown answers for the first 2 actual work questions, then answered a gold test question correctly, worked on another question with an unknown answer, and finally answered a gold test question incorrectly.

JSON for command line:
```json
{
    'tutorial': [true, true],
    'screening': [true, true, false],
    'work': [null, null, true, null, false]
}
```

Python:
``python`
{
    'tutorial': [True, True],
    'screening': [True, True, False],
    'work': [None, None, True, None, False]
}
```






## Sample python usage

```python
from crowdgating import Gate
args = {
    'n_tutorial': 5,
    'n_screening': 5,
   ...
}
history = {
    'tutorial': [True, True],
    'screening': [True, True, False],
    'work': [None, None, True, None, False],
}
gate = Gate(**args)
print(gate.next(history))
>>> {'test': False}

```

## Sample command line usage

```
> source venv/bin/activate
> python3 -m crowdgating.main crowdgating/examples/history1.json
```

## Sample web application

### Usage (hosted, not local URLs)

<https://crowdgating.appspot.com/?>
You may use the following GET params
- `n_tutorial` (int): Number of tutorial questions. Defaults to `5`.
- `n_screening` (int): Number of screening questions. Defaults to `5`.
- `desired_accuracy` (float): Desired accuracy (between `0` and `1`). Used as a gating threshold for both the screening questions and the inserted gold questions. Defaults to `0.8`.
- `n_gold_sliding` (int): Number of last inserted gold to use for removing workers. Defaults to `10`.
- `batch_size` (int): Size of a batch of work questions (not tutorial or screening). Defaults to `20`. 
- `gold_per_batch` (int): Number of gold questions to insert randomly into a batch. Defaults to `5`.
- `exponential_backoff` (int): If equal to `1`, ask gold questions only in exponentially decreasing batches (1, 2, 4, 8, ...). If equal to `0`, ask gold questions in every batch. Defaults to `1`.
- `seed` (int): Random seed to use for gold question insertion. Should never change for a worker.

These params are saved unless you make a GET request with at least one of these parameters again, e.g.:
<https://crowdgating.appspot.com/?n_tutorial=5>

If you click the `Reset` button, you will become a new worker but keep the params.

### Installation

Install the [GCloud CLI](https://cloud.google.com/sdk/).

```
python3 -m venv venv
source venv/bin/activate
pip install -t lib -r requirements.txt
```

To run in development, use `dev_appserver.py .`.

### Deploying your own copy

To deploy, use `gcloud app deploy`.

## Testing

```
> python -m unittest discover crowdgating/tests/
> python -m crowdgating.gating
```
