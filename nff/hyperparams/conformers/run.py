import sys
sys.path.insert(0, "/home/saxelrod/Repo/projects/covid_nff/NeuralForceField")

import argparse
import json
import os

from nff.data import Dataset
from nff.hyperparams.utils import (create_expt,
                                   make_model_folder,
                                   get_splits,
                                   evaluate_model,
                                   save_info,
                                   get_data_dic,
                                   make_class_model,
                                   make_param_dic,
                                   make_trainer,
                                   log,
                                   loop_begin_msg,
                                   loop_end_msg,
                                   conclude_round)

import pdb

def init_loop(conn,
              experiment,
              save_dir,
              project_name,
              set_params,
              base_train,
              base_val,
              base_test,
              target_name,
              metrics):

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

    model = make_class_model(model_type='WeightedConformers',
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


def main(project_name,
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

        (suggestion, model_folder, param_dic,
         data_dic,  model,  metric_dics) = init_loop(conn=conn,
                                                     experiment=experiment,
                                                     save_dir=save_dir,
                                                     project_name=project_name,
                                                     set_params=set_params,
                                                     base_train=base_train,
                                                     base_val=base_val,
                                                     base_test=base_test,
                                                     target_name=target_name,
                                                     metrics=monitor_metrics)

        T = make_trainer(model=model,
                         train_loader=data_dic["train"]["loader"],
                         val_loader=data_dic["val"]["loader"],
                         model_folder=model_folder,
                         loss_name=loss_name,
                         loss_coef={target_name: 1},
                         metric_dics=metric_dics,
                         max_epochs=num_epochs)

        begin_log(model_folder=model_folder, project_name=project_name,
                  suggestion=suggestion)

        try:
            T.train(device=device, n_epochs=num_epochs)
        except Exception as e:
            print(e)
            pdb.post_mortem()
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('info_file',
                        type=str,
                        help=('path of the json file with information '
                              'about the hyperparameter optimization.'))

    arguments = parser.parse_args()
    with open(arguments.info_file, "r") as f:
        info = json.load(f)
    main(**info)
