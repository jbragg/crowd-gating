import os
import copy
from flask import Flask
from flask import render_template
from flask import request
from flask import session
from flask import redirect
from flask import url_for
from flask import flash
from flask import jsonify
from crowdgating import Gate

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

RIGHT = 'right'
WRONG = 'wrong'


DEFAULTS = {
    'n_tutorial': 5,
    'n_screening': 5,
}
NEW_WORKER_MESSAGE = 'This is a new worker.'

class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

def get_args(args):
    d = copy.copy(DEFAULTS)
    int_keys = ['n_tutorial', 'n_screening', 'n_gold_sliding', 'batch_size', 'gold_per_batch', 'seed', 'exponential_backoff']
    float_keys = ['desired_accuracy']
    for key in int_keys:
        if key in args:
            try:
                d[key] = int(args[key])
            except:
                raise InvalidUsage('Invalid {} parameter'.format(key))
    for key in float_keys:
        if key in args:
            try:
                d[key] = float(args[key])
            except:
                raise InvalidUsage('Invalid {} parameter'.format(key))
    return d

@app.route('/', methods=['GET', 'POST'])
def question():
    if request.args or not session.get('args'):
        session['args'] = get_args(request.args)
        reset(**request.args)
    args = session.get('args')
    gate = Gate(**args)

    history = session.get('history', {})
    session['history'] = history
    next_action = gate.next(history)
    session['next'] = next_action

    if not next_action:
        return bye()

    feedback=False
    if request.method == 'POST':
        if request.form.get('answer') is None:
            return render_template('question.html', feedback='You must provide an answer')
        is_right = request.form.get('answer') == RIGHT

        if 'tutorial' in next_action:
            history['tutorial'] = history.get('tutorial', [])
            if not history['tutorial'] or next_action['tutorial'] >= len(history['tutorial']):
                history['tutorial'].append(is_right)
            else:
                history['tutorial'][-1] = is_right
            if not is_right:
                feedback='Try again. Wrong answer is wrong because [REASON]'
        if 'screening' in next_action:
            history['screening'] = history.get('screening', [])
            history['screening'].append(is_right)
        if 'test' in next_action:
            history['work'] = history.get('work', [])
            history['work'].append(is_right if next_action['test'] else None)
        next_action = gate.next(history)

    if not next_action:
        return bye()

    if 'tutorial' in next_action and not history.get('tutorial'):
        flash('Start of tutorial (should tell worker about this)')
    elif 'screening' in next_action and not history.get('screening'):
        flash('Start of screening (should tell worker about this)')
    elif 'test' in next_action and not history.get('work'):
        flash('Start of main task (should tell worker about this)')

    return render_template('question.html', feedback=feedback)

def bye(**kwargs):
    messages = ['The last worker was asked (politely!) to leave.']
    return reset(extra_messages=messages, **kwargs)

@app.route('/reset')
def reset(extra_messages=None, **kwargs):
    args = session.get('args')
    session.clear()
    if args:
        session['args'] = args
    if extra_messages:
        for message in extra_messages:
            flash(message)
    flash(NEW_WORKER_MESSAGE)
    return redirect(url_for('question', **kwargs))

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
