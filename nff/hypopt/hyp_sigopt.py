from datetime import datetime
from sigopt import Connection
import os

from nff.hypopt.hyp_io import (make_model_folder,
                               save_info,
                               get_splits)
from nff.hypopt.hyp_data import get_data_dic
from nff.hypopt.hyp_params import make_class_model
from nff.hypopt.hyp_eval import evaluate_model
from nff.hypopt.hyp_train import make_trainer

from nff.data import Dataset


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


def add_features(param_dic):
    """
    Example:
        param_dic = {
            "extra_features": [{"name": "morgan", "length": 1048},
                              {"name": "rdkit_2d"}],
            "rdkit_2d_length": 120

        }
    "rdkit_2d_length" is specified outside of "extra_features" here
    because it's a sigopt-learnable parameter, and so it has
    to be specified as its own entity.

    """
    extra_feats = param_dic.get("extra_features")
    if extra_feats is None:
        return

    for dic in extra_feats:
        if "name" in dic and "length" in dic:
            continue
        name = dic["name"]
        length = param_dic.get("{}_length".format(name))
        if length is None:
            raise Exception(("Must specify length of "
                             "{} feature".format(name)))
        dic["length"] = length


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
    add_features(param_dic)
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


def init_class_loop(conn,
                    experiment,
                    save_dir,
                    project_name,
                    set_params,
                    base_train,
                    base_val,
                    base_test,
                    target_name,
                    metrics,
                    model_type):

    suggestion = conn.experiments(experiment.id).suggestions().create()
    model_folder = make_model_folder(save_dir=save_dir,
                                     project_name=project_name,
                                     model_id=suggestion.id)
    param_dic = make_param_dic(set_params=set_params,
                               assignments=suggestion.assignments)
    data_dic = get_data_dic(base_train=base_train,
                            base_val=base_val,
                            base_test=base_test,
                            params=param_dic)

    model = make_class_model(model_type=model_type,
                             param_dic=param_dic)
    metric_dics = [{"target": target_name, "metric": name}
                   for name in metrics]

    return (suggestion,
            model_folder,
            param_dic,
            data_dic,
            model,
            metric_dics)


def begin_log(model_folder, project_name, suggestion):
    log_file = os.path.join(model_folder, "..", "log.txt")
    msg = loop_begin_msg(suggestion=suggestion)
    log(project_name=project_name,
        msg=msg,
        log_file=log_file)


def end_log(value, eval_metric, model_folder, project_name):
    log_file = os.path.join(model_folder, "..", "log.txt")
    msg = loop_end_msg(value=value,
                       metric_name=eval_metric,
                       model_folder=model_folder)
    log(project_name=project_name, msg=msg,
        log_file=log_file)


def get_init_func(model_kind):
    dic = {"classification": init_class_loop}
    return dic[model_kind]


def run_loop(project_name,
             save_dir,
             param_regime,
             val_size,
             test_size,
             objective,
             client_token,
             dataset_path,
             target_name,
             eval_metric,
             monitor_metrics,
             set_params,
             loss_name,
             num_epochs,
             device,
             budget,
             eval_on,
             model_type,
             model_kind,
             loss_coef,
             **kwargs):

    conn, experiment = create_expt(name=project_name,
                                   param_regime=param_regime,
                                   objective=objective,
                                   client_token=client_token,
                                   budget=budget)

    dataset = Dataset.from_file(dataset_path)

    base_train, base_val, base_test = get_splits(dataset=dataset,
                                                 val_size=val_size,
                                                 test_size=test_size,
                                                 save_dir=save_dir,
                                                 project_name=project_name)

    while (experiment.progress.observation_count <
           experiment.observation_budget):

        init_func = get_init_func(model_kind)
        (suggestion, model_folder, param_dic,
         data_dic,  model,  metric_dics) = init_func(conn=conn,
                                                     experiment=experiment,
                                                     save_dir=save_dir,
                                                     project_name=project_name,
                                                     set_params=set_params,
                                                     base_train=base_train,
                                                     base_val=base_val,
                                                     base_test=base_test,
                                                     target_name=target_name,
                                                     metrics=monitor_metrics,
                                                     model_type=model_type)

        T = make_trainer(model=model,
                         train_loader=data_dic["train"]["loader"],
                         val_loader=data_dic["val"]["loader"],
                         model_folder=model_folder,
                         loss_name=loss_name,
                         loss_coef=loss_coef,
                         metric_dics=metric_dics,
                         max_epochs=num_epochs)

        begin_log(model_folder=model_folder, project_name=project_name,
                  suggestion=suggestion)
        try:
            T.train(device=device, n_epochs=num_epochs)
        except Exception as e:
            print(e)
            experiment = conclude_round(conn=conn,
                                        experiment=experiment,
                                        suggestion=suggestion,
                                        failed=True)
            continue

        best_model = T.get_best_model()

        value = evaluate_model(model=best_model,
                               target_name=target_name,
                               metric_name=eval_metric,
                               loader=data_dic[eval_on]["loader"])
        experiment = conclude_round(conn=conn,
                                    experiment=experiment,
                                    suggestion=suggestion,
                                    value=value)

        save_info(param_regime=param_regime,
                  assignments=suggestion.assignments,
                  value=value,
                  metric_name=eval_metric,
                  model_folder=model_folder,
                  sugg_id=suggestion.id,
                  expt_id=experiment.id,
                  set_params=set_params)

        end_log(value=value,
                eval_metric=eval_metric,
                model_folder=model_folder,
                project_name=project_name)
