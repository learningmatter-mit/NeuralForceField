from datetime import datetime
from sigopt import Connection


def create_expt(name, param_regime, objective, client_token, budget='default'):
    conn = Connection(
        client_token=client_token)

    # usually 10-20 x number of parameters
    if budget == 'default':
        budget = 15 * len(param_regime)

    experiment = conn.experiments().create(
        name=name,
        metrics=[dict(name="objective", objective=objective)],
        parameters=param_regime,
        observation_budget=budget
    )

    return conn, experiment


def make_param_dic(set_params,
                   assignments):
    param_dic = {}
    all_keys = list(assignments.keys())
    all_keys += list(set_params.keys())

    for key in all_keys:
        if key in assignments.keys():
            param_dic.update({key: assignments[key]})
        else:
            param_dic.update({key: set_params[key]})
    return param_dic


def loop_begin_msg(suggestion):
    time = datetime.now()
    assignments = str(suggestion.assignments)
    msg = ("Date: {0}\n"
           "Assignments: {1}\n".format(time, assignments))
    return msg


def loop_end_msg(value, metric_name, model_folder):
    model_num = model_folder.split("_")[-1]
    msg = "{} performance of model {}: {}".format(
        metric_name, model_num, value)

    return msg


def conclude_round(conn,
                   experiment,
                   suggestion,
                   value=None,
                   failed=False):
    if failed:
        conn.experiments(experiment.id).observations().create(
            suggestion=suggestion.id,
            failed=failed
        )
    else:
        conn.experiments(experiment.id).observations().create(
            suggestion=suggestion.id,
            value=value
        )  
    experiment = conn.experiments(experiment.id).fetch()
    return experiment


def log(project_name, msg, log_file):
    text = "{:>30}".format(msg)
    print(text)
    with open(log_file, 'a') as f:
        f.write(text)
