import sys

import numpy as np
import tensorflow as tf
from graphcompletionlib.config import get_config
from graphcompletionlib.trainer import Trainer
from graphcompletionlib.utils import prepare_dirs, save_config, \
    save_results, evaluate_result
from main.common.logger import log

config = None

def main(_):
    log("In gcrn_main_gcnn")

    prepare_dirs(config)
    prepare_config_date(config, config.ds_ind)
    # Random seed settings
    rng = np.random.RandomState(config.random_seed)
    tf.set_random_seed(config.random_seed)

    # Model training
    log("Initilializing trainer...")
    trainer = Trainer(config, rng)
    save_config(config.model_dir, config)
    log("Trainer initialized!")

    config.is_train = True

    config.load_path = config.model_dir
    if config.is_train:
        log("Training...")
        trainer.train(save=True)
        log("Training done!")
        log("Testing...")
        result_dict = trainer.test()
        log("Testing done!")
    else:
        if not config.load_path:
            raise Exception(
                "[!] You should specify `load_path` to "
                "load a pretrained model")
        log("Testing...")
        result_dict = trainer.test()
        log("Testing done!")
    save_results(config.result_dir, result_dict)
    accept_rate = evaluate_result(result_dict, method='KS-test', alpha=0.1)
    kl_div = evaluate_result(result_dict, method='KL')
    wasser_dis = evaluate_result(result_dict, method='wasser')
    sig_test = evaluate_result(result_dict, method='sig_test')
    print("The accept rate of KS test is ", accept_rate)
    print("The final KL div is ", kl_div)
    print("The wasser distance is ", wasser_dis)
    print("The AR of Sign Test is ", sig_test)


def run_algo(server_name):
    global config

    print("Running graph completion algorithm")
    config, unparsed = get_config()

    # SW502 config
    # If prediction is on, no training will be performed, and only test batches will be made
    # in order to test the current model with generated data.
    config.is_predicting = False
    if config.is_predicting:
        config.is_train = False
    # SW502 config end

    config.mode = 'prediction'
    config.target = 'hist'
    config.classif_loss = 'kl'
    config.hist_range = list(range(0, 41, 5))

    config.data_rm = 0.5
    config.ds_ind = 0 # Index of dates

    input=f"--server_name={server_name}"

    # optimal params for kl 1e-3
    config.server_name = server_name
    config.conv = 'gcnn'
    config.filter = 'chebyshev5'
    config.is_coarsen = True
    config.is_train = True
    config.stop_early = True
    config.sub_folder = False
    config.stop_win_size = 10
    config.learning_rate = 4e-5
    config.dropout = 0.3
    config.regularization = 1e-4
    config.decay_rate = 0.999
    config.num_kernels = [32, 16]
    config.conv_size = [8, 16]
    config.pool_size = [4, 2]
    config.normalized = True

    tf.app.run(main=main, argv=[input] + unparsed)

def prepare_config_date(config, ind):
    """
    prepare the config date

    """
    if config.server_name == 'chengdu':
        config.win_size = 3
        start_months = [8, 8, 8, 8, 8]
        start_dates = [2, 8, 12, 18, 22]
        end_months = [8, 8, 8, 8, 8]
        end_dates = [8, 12, 18, 22, 31]
    elif config.server_name == 'sw502':
        config.win_size = 3
        start_months = [10]
        start_dates = [31]
        end_months = [11]
        end_dates = [1]
    else:
        raise Exception(
            "[!] Unkown server name: {}".format(config.server_name))

    config.s_month = start_months[ind]
    config.s_date = start_dates[ind]
    config.e_month = end_months[ind]
    config.e_date = end_dates[ind]

    print(f"Start date: {config.s_date}, start month: {config.s_month}")
    print(f"End date: {config.e_date}, end month: {config.e_month}")
