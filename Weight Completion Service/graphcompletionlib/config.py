#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Parse input command to hyper-parameters


import argparse

parser = argparse.ArgumentParser()
arg_list = []

def str2bool(v):
    return v.lower() in ('true', '1')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_list.append(arg)
    return arg


# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--server_name', type=str, default='server_kdd',
                      choices=['server_kdd', 'chengdu'], help='')
data_arg.add_argument('--conv', type=str, default='gcnn',
                      choices=['gcnn', 'cnn'], help='')
data_arg.add_argument('--ds_ind', type=int, default=0,
                      choices=[0, 1, 2, 3, 4], help='')
data_arg.add_argument('--normalized', type=bool, default=True, help='')
data_arg.add_argument('--mode', type=str, default='estimation',
                      choices=['estimation', 'prediction'], help='')
data_arg.add_argument('--target', type=str, default='hist',
                      choices=['hist', 'avg'], help='')
data_arg.add_argument('--sample_rate', type=int, default=15, help='')
data_arg.add_argument('--data_rm', type=float, default=0.5,
                      choices=[0.5, 0.6, 0.7, 0.8], help='')
data_arg.add_argument('--hist_range', type=list, default=list(range(0, 41, 5)),
                      choices=[list(range(0, 41, 10)), list(range(0, 41, 5)), list(range(0, 41, 2))], help='')
data_arg.add_argument('--coarsening_level', type=int, default=4, help='')
data_arg.add_argument('--is_coarsen', type=bool, default=True, help='')

# Training/Test param
train_arg = add_argument_group('Training')
train_arg.add_argument('--num_epochs', type=int, default=200, help='')
train_arg.add_argument('--stop_win_size', type=int, default=5, help='')
train_arg.add_argument('--stop_early', type=bool, default=False, help='')
train_arg.add_argument('--sub_folder', type=bool, default=False, help='')
train_arg.add_argument('--batch_size', type=int, default=20, help='')
train_arg.add_argument('--random_seed', type=int, default=123, help='')
train_arg.add_argument('--max_step', type=int, default=1000000, help='')
train_arg.add_argument('--is_train', type=str2bool, default=True, help='')
train_arg.add_argument('--classif_loss', type=str,
                       default='kl_div', choices=['kl_div', 'l2'], help='')
train_arg.add_argument('--learning_rate', type=float, default=1e-4, help='')
train_arg.add_argument('--decay_step', type=int, default=10, help='')
train_arg.add_argument('--decay_rate', type=float, default=0.99, help='')
train_arg.add_argument('--max_grad_norm', type=float, default=-1, help='')
train_arg.add_argument('--optimizer', type=str,
                       default='adam', choices=['adam_wgan', 'adam', 'sgd', 'rmsprop'], help='')
train_arg.add_argument('--checkpoint_secs', type=int, default=300, help='')
train_arg.add_argument('--dropout', type=float, default=0.4, help='')
train_arg.add_argument('--regularization', type=float, default=1e-4, help='')


# Model args
model_arg = add_argument_group('Model')
model_arg.add_argument('--model_type', type=str, default='hist',
                        choices=['hist', 'avg'], help='')
model_arg.add_argument('--filter', type=str, default='chebyshev5',
                        choices=['chebyshev5', 'conv1', 'conv2',
                                 'lanczos', 'learn_heat'], help='')
model_arg.add_argument('--brelu', type=str, default='b1relu',
                        choices=['b1relu', 'b2relu'], help='')
model_arg.add_argument('--pool', type=str, default='mpool1',
                        choices=['mpool1', 'mpool2'], help='')
# LSM model args
model_arg.add_argument('--T', type=int, default=1, help='')
# default value for HW dataset,
model_arg.add_argument('--k', type=int, default=10, help='')
model_arg.add_argument('--lamda', type=float, default=pow(2, 3), help='')


model_arg.add_argument('--gamma', type=float, default=2e-5, help='')
model_arg.add_argument('--eval_frequency', type=int, default=20, help='')
model_arg.add_argument('--converge_loss', type=float, default=0.01, help='')

# Hyperparams for graph
graph_arg = add_argument_group('Graph')
graph_arg.add_argument('--feat_in', type=int, default=1,
                       choices=[1, 4, 8], help='')
graph_arg.add_argument('--feat_out', type=int, default=1,
                       choices=[1, 4, 8], help='')
graph_arg.add_argument('--num_kernels', type=list, default=[32, 16])
graph_arg.add_argument('--conv_size', type=list, default=[8, 8])
graph_arg.add_argument('--pool_size', type=list, default=[4, 2])
graph_arg.add_argument('--FC_size', type=list, default=[])

# Miscellaneous (summary write, model reload)
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--log_step', type=int, default=20, help='')
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--load_path', type=str, default="")
misc_arg.add_argument('--gpu_memory_fraction', type=float, default=1.0)
misc_arg.add_argument('--output_dir', type=str, default='output')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

