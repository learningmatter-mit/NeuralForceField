from datetime import datetime
from sigopt import Connection
import os
import json

from nff.hypopt.io import (make_model_folder,
                           save_info,
                           get_splits)
from nff.hypopt.data import get_data_dic
from nff.hypopt.params import (make_wc_model,
                               make_cp3d_model,
                               make_cp2d_model)
from nff.hypopt.eval import evaluate_model
from nff.hypopt.train import make_trainer

from nff.data import Dataset


def create_expt(name,
                param_regime,
                objective,
                client_token,
                metric_name,
                budget='default',
                expt_id=None):
    conn = Connection(client_token=client_token)

    # usually 10-20 x number of parameters
    if budget == 'default':
        budget = 15 * len(param_regime)

    if expt_id is not None:
        experiment = conn.experiments().fetch(
            id=expt_id).data[0]
    else:
        experiment = conn.experiments().create(
            name=name,
            metrics=[dict(name=metric_name, objective=objective)],
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


def get_log_file(model_folder):
    log_file = os.path.join(model_folder, "../log.txt")
    return log_file


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

    model_builder = get_model_builder(model_type)
    model = model_builder(param_dic=param_dic)
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


def get_model_builder(model_type):
    dic = {"WeightedConformers": make_wc_model,
           "ChemProp3D": make_cp3d_model,
           "ChemProp2D": make_cp2d_model}
    return dic[model_type]


def get_expt_ids(project_name, save_dir):

    main_dir = os.path.join(save_dir, "sigopt", project_name)
    folders = os.listdir(main_dir)
    model_folders = [folder for folder in folders
                     if folder.startswith("model_")]
    expt_ids = []

    for folder in model_folders:
        job_info_file = os.path.join(main_dir,
                                     folder, "job_info.json")
        if not os.path.isfile(job_info_file):
            continue
        with open(job_info_file, "r") as f:
            job_info = json.load(f)
        expt_id = job_info["experiment_id"]
        expt_ids.append(expt_id)

    expt_ids = list(set(expt_ids))

    return expt_ids


def get_best_params(project_name,
                    save_dir,
                    client_token,
                    set_params,
                    objective):

    expt_ids = get_expt_ids(project_name=project_name,
                            save_dir=save_dir)
    conn = Connection(client_token=client_token)

    comparison_dic = {"minimize": lambda old, new: new < old,
                      "maximize": lambda old, new: new > old}
    best_val = None

    for expt_id in expt_ids:
        all_best_assignments = conn.experiments(expt_id
                                                ).best_assignments().fetch()
        new_val = all_best_assignments.data[0].value
        if best_val is None:
            best_val = new_val
        else:
            better = comparison_dic[objective](best_val, new_val)
            if better:
                best_val = new_val

        if best_val == new_val:
            assignments = dict(all_best_assignments.data[0].assignments)
            param_dic = make_param_dic(set_params=set_params,
                                       assignments=assignments)
            assgn_id = all_best_assignments.data[0].id

    return param_dic, assgn_id


def report_scores(eval_metric,
                  model_folder,
                  project_name,
                  score_dic):

    log_file = get_log_file(model_folder)
    names = ["train", "val", "test"]
    for name in names:
        score = score_dic[name]
        msg = "%s score on %s set is %.3f" % (eval_metric,
                                              name,
                                              score)
        log(project_name=project_name,
            msg=msg,
            log_file=log_file)

    stats_file = os.path.join(model_folder, "stats.json")
    with open(stats_file, "w") as f:
        json.dump(score_dic, f, indent=4, sort_keys=True)


def get_scores(eval_metric,
               best_model,
               model_type,
               target_name,
               data_dic,
               param_dic):
    score_dic = {"metric": eval_metric}
    for name in ["train", "val", "test"]:
        score = evaluate_model(model=best_model,
                               model_type=model_type,
                               target_name=target_name,
                               metric_name=eval_metric,
                               loader=data_dic[name]["loader"],
                               param_dic=param_dic)
        score_dic.update({name: score})
    return score_dic


def get_and_report_scores(eval_metric,
                          best_model,
                          model_folder,
                          project_name,
                          model_type,
                          target_name,
                          data_dic,
                          param_dic):
    score_dic = get_scores(eval_metric=eval_metric,
                           best_model=best_model,
                           model_type=model_type,
                           target_name=target_name,
                           data_dic=data_dic,
                           param_dic=param_dic)
    report_scores(eval_metric=eval_metric,
                  model_folder=model_folder,
                  project_name=project_name,
                  score_dic=score_dic)


def retrain_best(project_name,
                 save_dir,
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
                 model_type,
                 loss_coef,
                 **kwargs):

    # get data
    dataset = Dataset.from_file(dataset_path)
    base_train, base_val, base_test = get_splits(dataset=dataset,
                                                 val_size=val_size,
                                                 test_size=test_size,
                                                 save_dir=save_dir,
                                                 project_name=project_name)
    # bet param_dic for best assignments
    param_dic, assgn_id = get_best_params(project_name=project_name,
                                          save_dir=save_dir,
                                          client_token=client_token,
                                          set_params=set_params,
                                          objective=objective)

    # make a new folder for retraining
    model_id = "assgn_{}_retrain".format(assgn_id)
    model_folder = make_model_folder(save_dir=save_dir,
                                     project_name=project_name,
                                     model_id=model_id)

    # get the data, build the model and make the trainer
    data_dic = get_data_dic(base_train=base_train,
                            base_val=base_val,
                            base_test=base_test,
                            params=param_dic)

    model_builder = get_model_builder(model_type)
    model = model_builder(param_dic=param_dic)
    metric_dics = [{"target": target_name, "metric": name}
                   for name in monitor_metrics]

    T = make_trainer(model=model,
                     model_type=model_type,
                     train_loader=data_dic["train"]["loader"],
                     val_loader=data_dic["val"]["loader"],
                     model_folder=model_folder,
                     loss_name=loss_name,
                     loss_coef=loss_coef,
                     metric_dics=metric_dics,
                     max_epochs=num_epochs,
                     param_dic=param_dic)

    # train
    T.train(device=device, n_epochs=num_epochs)
    best_model = T.get_best_model()

    # get and report scores
    get_and_report_scores(eval_metric=eval_metric,
                          best_model=best_model,
                          model_folder=model_folder,
                          project_name=project_name,
                          model_type=model_type,
                          target_name=target_name,
                          data_dic=data_dic,
                          param_dic=param_dic)


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
             expt_id=None,
             **kwargs):

    conn, experiment = create_expt(name=project_name,
                                   param_regime=param_regime,
                                   objective=objective,
                                   client_token=client_token,
                                   budget=budget,
                                   metric_name=eval_metric,
                                   expt_id=expt_id)

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

        print(model)
        T = make_trainer(model=model,
                         model_type=model_type,
                         train_loader=data_dic["train"]["loader"],
                         val_loader=data_dic["val"]["loader"],
                         model_folder=model_folder,
                         loss_name=loss_name,
                         loss_coef=loss_coef,
                         metric_dics=metric_dics,
                         max_epochs=num_epochs,
                         param_dic=param_dic)

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
                               model_type=model_type,
                               target_name=target_name,
                               metric_name=eval_metric,
                               loader=data_dic[eval_on]["loader"],
                               param_dic=param_dic)
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
