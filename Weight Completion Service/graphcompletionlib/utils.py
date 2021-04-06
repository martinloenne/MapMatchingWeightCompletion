import tensorflow as tf
import numpy as np
import math
import pickle
import os
import sys
import json
from datetime import datetime
import tensorflow.contrib.slim as slim
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
import configparser
from sqlalchemy import create_engine
import graphcompletionlib.dataset as ds
import graphcompletionlib.coarsening as coarsening
from pyemd import emd

from main.common.logger import log

CONF_DIR = os.path.join('.', 'graphcompletionlib\conf')
EPSILON = 1e-6

def connect_sql_server(server, dir):
    print("Connecting to sql server...")
    print("Server name param: " + server)
    print("Dir name param: " + dir)




    """Connect to a sql server with configure file

    This function is used to connect to a sql server using sqlalchemy package by
    specifying a configure file and the a configure name in this file.

    :param server (string): the name of the sql server in configure file
    :param dir (string): the directory of configure file

    :return: the connected sql engine.
    """
    db_conf_file = os.path.join(dir, "dbconf.conf")
    db_conf = configparser.ConfigParser()
    db_conf.read(db_conf_file)
    connect_str = db_conf.get(server, "conn_str")
    engine = create_engine(connect_str)

    return engine


def specify_node(link_id, row, col_name, added, node_id, dict_edge_node):
    nodes = []
    nodes.append(link_id)
    if row[col_name] is not None:
        in_tops = row[col_name].split(',')
        for in_top in in_tops:
            node = in_top + added
            nodes.append(node)
    cur_node_id = node_id
    node_exist = False
    for node in nodes:
        if node in dict_edge_node.keys():
            node_exist = True
            cur_node_id = dict_edge_node[node]
            break
    for node in nodes:
        dict_edge_node[node] = cur_node_id
    if node_exist:
        cur_node_id = node_id
    else:
        cur_node_id += 1

    return dict_edge_node, cur_node_id


def save_config(model_dir, config):
    '''
    save config params in a form of param.json in model directory
    '''

    param_path = os.path.join(model_dir, "params.json")

    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def prepare_dirs(config, test=False):
    config.model_name = "{}_{}_{}_{}_{}_{}_rm{}".format(
        config.mode, config.server_name,
        config.target, config.conv, config.filter,
        config.ds_ind, config.data_rm)

    sub_folder = 'lr_{0:.0e}_reg_{1:.0e}_dp_{2:.0e}_decay_{3:.0e}' \
                 '_{4}_{5}_{6}_{7}_{8}_{9}/'.format(
        config.learning_rate, config.regularization,
        config.dropout, config.decay_rate,
        config.num_kernels[0], config.num_kernels[1],
        config.conv_size[0], config.conv_size[1],
        config.pool_size[0], config.pool_size[1])

    config.model_dir = os.path.join(
        config.log_dir, config.mode,
        config.server_name, config.target,
        config.conv, config.filter,
        '{}_rm{}'.format(config.ds_ind, config.data_rm))

    if config.sub_folder:
        config.model_dir = os.path.join(
            config.model_dir, sub_folder)

    for path in [config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
            print("Model Directory '%s' created" % path)

    if test:
        config.result_dir = os.path.join(
            config.output_dir,
            'test',
            # config.classif_loss,
            config.model_name)
    else:
        config.result_dir = os.path.join(
            config.output_dir,
            config.model_name)

    for path in [config.result_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
            print("Output Directory '%s' created" % path)







def prepare_model_param(config):
    config.FC_size = []


def pklLoad(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def pklSave(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def save_results(result_dir, obj):
    fname = os.path.join(result_dir, 'result_dict.pickle')
    pklSave(fname, obj)


def read_results(result_dir):
    fname = os.path.join(result_dir, 'result_dict.pickle')
    return pklLoad(fname)

def cal_kl_div(y_pred, y_true, base=2, eps=1e-6):
    log_op = np.log2(y_pred+eps) - np.log2(y_true+eps)
    mul_op = np.multiply(y_pred, log_op)
    sum_hist = np.sum(mul_op, axis=1)
    multi_factor = np.log2(base)
    sum_hist = sum_hist/multi_factor

    return sum_hist

def cal_cos_dist(y_pred, y_true):
    mul = np.multiply(y_pred, y_true)
    sum_mul = np.sum(mul, axis=-1)
    square_pred = y_pred**2
    square_true = y_true**2
    sum_pred2 = np.sum(square_pred, axis=-1)
    sum_true2 = np.sum(square_true, axis=-1)
    root_mul = np.sqrt(sum_pred2) * np.sqrt(sum_true2)
    cos_matrix = sum_mul / root_mul

    return cos_matrix

def cal_common_area(y_pred, y_true):
    inter_section = np.minimum(y_pred, y_true)
    total_section = np.sum(inter_section, axis=1)

    return total_section

def cal_bc(y_pred, y_true):
    mul = np.multiply(y_pred, y_true)
    mul_root = np.sqrt(mul)
    bc_matrix = np.sum(mul_root, axis=-1)

    return bc_matrix

def weighted_kl_div(y_true, y_pred,
                    epsilon=EPSILON, metric='kl', base=2):
    if metric == 'kl':
        y_true = y_true
        y_pred = y_pred
        sum_hist = cal_kl_div(y_pred, y_true, base, eps=epsilon)
    elif metric == 'kl_sim':
        y_true = y_true
        y_pred = y_pred
        sum_hist = cal_kl_div(y_pred, y_true, base=base, eps=epsilon)
        max_bound = np.log2(1/epsilon) / np.log2(base)
        sum_hist = 1 - sum_hist/max_bound
    elif metric == 'emd':
        sum_hist = weighted_emd(y_true, y_pred)
    elif metric == 'cos':
        sum_hist = cal_cos_dist(y_pred, y_true)
    elif metric == 'inter':
        sum_hist = cal_common_area(y_pred, y_true)
    elif metric == 'bc':
        sum_hist = cal_bc(y_pred, y_true)
    elif metric == 'ce':
        sum_hist = weighted_cross_entropy(y_true, y_pred, epsilon)
    else:
        sum_hist = cal_kl_div(y_pred, y_true, eps=epsilon)

    sum_hist = sum_hist[np.isnan(sum_hist) == False]
    print("The total test number is ", sum_hist.shape[0])
    avg_kl_div = np.nanmean(sum_hist)
    std_kl_div = np.std(sum_hist)

    return avg_kl_div, std_kl_div


def weighted_kl_div_true(y_true, y_pred, epsilon=1e-3):
    log_op = np.log(y_true + epsilon) - np.log(y_pred + epsilon)
    mul_op = np.multiply(y_true, log_op)
    sum_hist = np.sum(mul_op, axis=1)
    avg_kl_div = np.nanmean(sum_hist)

    return avg_kl_div


def weighted_cross_entropy(y_true, y_pred, epsilon=1e-3):
    log_op = np.log(y_pred+epsilon)
    mul_op = np.multiply(y_true, log_op)
    sum_hist = np.sum(mul_op, axis=1)

    return sum_hist


def weighted_mape(y_true, y_pred, weight, y_scaler):
    predict = y_scaler.inverse_transform(y_pred)
    real = y_scaler.inverse_transform(y_true)

    avg_mape = np.sum(np.abs((real - predict) / real)
                      * weight) / np.sum(weight) * 100

    return avg_mape


def get_avg_predictions(y_pred, y_scaler):
    predict = y_scaler.inverse_transform(y_pred)

    return predict


def wasser_dist(y_true, y_pred, p=1):
    num_bins = y_true.shape[1]
    cdf_pred = y_pred.copy()
    cdf_true = y_true.copy()
    for i in range(num_bins):
        cdf_pred[:, i] = np.sum(y_pred[:, :i + 1], axis=1)
        cdf_true[:, i] = np.sum(y_true[:, :i + 1], axis=1)
    abs_minus = np.abs(cdf_pred - cdf_true)
    abs_minus_pow = abs_minus ** p
    sum_abs_minus_pow = np.sum(abs_minus_pow, axis=1)
    root_sum_abs_minus_pow = sum_abs_minus_pow ** 1 / p
    p_wasser_dist = np.nanmean(root_sum_abs_minus_pow)

    return p_wasser_dist


def euclidean_dist(y_true, y_pred, p=2):
    abs_minus = np.abs(y_true - y_pred)
    abs_minus_pow = abs_minus ** p
    sum_abs_minus_pow = np.sum(abs_minus_pow, axis=1)
    root_sum_abs_minus_pow = sum_abs_minus_pow ** 1 / p
    p_wasser_dist = np.nanmean(root_sum_abs_minus_pow)

    return p_wasser_dist


def weighted_emd(y_true, y_pred):
    n_bucket = y_true.shape[1]
    nb_result = y_true.shape[0]
    d_matrix = np.zeros((n_bucket, n_bucket))
    for i in range(n_bucket):
        d_matrix[i, i:n_bucket] = np.arange(1, n_bucket+1)[:n_bucket - i]
    d_matrix = np.maximum(d_matrix, d_matrix.T)
    emds = []
    for j in range(nb_result):
        hist_true = y_true[j, :].astype(np.float64)
        hist_pred = y_pred[j, :].astype(np.float64)
        emd_j = emd(hist_pred, hist_true, d_matrix)
        emds.append(emd_j)
    emds = np.array(emds)

    return emds


def eval_ks_test(y_true, y_pred, count_y, alpha=0.05):
    num_bins = y_true.shape[1]

    cdf_pred = y_pred.copy()
    cdf_true = y_true.copy()
    upper_ones = np.triu(np.ones((num_bins, num_bins)), 0)
    cdf_pred = np.matmul(cdf_pred, upper_ones)
    cdf_true = np.matmul(cdf_true, upper_ones)
    max_dis = np.max(np.abs(cdf_pred - cdf_true), axis=1)

    # 1 step cross
    cdf_pred_1f = cdf_pred[:, :-1]
    cdf_true_1b = cdf_true[:, 1:]
    max_1f_1b = np.max(np.abs(cdf_pred_1f - cdf_true_1b), axis=1)

    cdf_pred_1b = cdf_pred[:, 1:]
    cdf_true_1f = cdf_true[:, :-1]
    max_1b_1f = np.max(np.abs(cdf_pred_1b - cdf_true_1f), axis=1)

    max_dis = np.maximum(max_dis, max_1f_1b)
    max_dis = np.maximum(max_dis, max_1b_1f)

    scale = np.sqrt(2 / count_y) * np.sqrt(-0.5 * np.log(alpha / 2))

    accept = np.sum(max_dis < scale)
    accept_rate = accept / y_true.shape[0]

    return accept_rate

def sig_test_zvalue(vel_list, pmf_array):
    num_bins = len(pmf_array)
    bucket_size = 40 / num_bins
    mean_bucket = np.arange(0, 40, bucket_size)
    mean_bucket = mean_bucket + bucket_size / 2
    pred_mean = np.sum(np.array(pmf_array) * mean_bucket)
    true_mean = np.mean(vel_list)
    true_std = np.std(vel_list)
    true_num = len(vel_list)
    z_value = np.abs(pred_mean - true_mean) / (true_std / np.sqrt(true_num))
    # print("True Mean: ", true_mean)
    # print("Pred Mean: ", pred_mean)
    return z_value


def sig_test_zvalue_array(y_true, pmf_array, vel_list):
    num_bins = len(pmf_array)
    bucket_size = 40 / num_bins
    mean_bucket = np.arange(0, 40, bucket_size)
    mean_bucket = mean_bucket + bucket_size / 2
    pred_mean = np.sum(np.array(pmf_array) * mean_bucket)
    true_mean = np.sum(np.array(y_true) * mean_bucket)
    true_std = np.std(vel_list)
    true_num = len(vel_list)
    z_value = np.abs(pred_mean - true_mean) / (true_std / np.sqrt(true_num))
    print("True Mean: ", true_mean)
    print("Pred Mean: ", pred_mean)
    return z_value


def eval_sig_test_vel_list(vel_list, y_pred, alpha=0.05):
    accept_num = 0
    for i in range(y_pred.shape[0]):
        pred_pmf = y_pred[i]
        vel_list_i = vel_list[i]
        max_dis = sig_test_zvalue(vel_list_i, pred_pmf)
        # scale = np.sqrt((len(vel_list_i) + 40) / (len(vel_list_i) * 40)) * np.sqrt(-0.5 * np.log(alpha / 2))
        scale = 1.282
        if max_dis < scale:
            accept_num += 1

    accept_rate = accept_num / y_pred.shape[0]

    return accept_rate


def eval_sig_test(y_true, y_pred, vel_list, alpha=0.05):
    accept_num = 0
    for i in range(y_pred.shape[0]):
        pred_pmf = y_pred[i]
        true_pmf = y_true[i]
        vel_list_i = vel_list[i]
        max_dis = sig_test_zvalue_array(true_pmf, pred_pmf, vel_list_i)
        # scale = np.sqrt((len(vel_list_i) + 40) / (len(vel_list_i) * 40)) * np.sqrt(-0.5 * np.log(alpha / 2))
        scale = 1.282
        if max_dis < scale:
            accept_num += 1

    accept_rate = accept_num / y_pred.shape[0]

    return accept_rate


def cal_max_wasser_dis(pmf_array, vel_list):
    round_vel = np.round(np.array(vel_list), decimals=0).astype(int)
    round_vel = round_vel[round_vel <= 40]
    sorted_array = np.sort(round_vel)
    true_cdf = np.zeros(41)
    sum_i = 0
    for i in range(41):
        if i in round_vel:
            record_i = np.sum(round_vel == i)
            sum_i += record_i / len(sorted_array)
        else:
            sum_i += 0
        true_cdf[i] = sum_i

    pred_cdf = np.zeros(41)
    unit = 40 / len(pmf_array)
    sum_j = 0
    for i in range(40):
        bucket_i = pmf_array[int(i / unit)]
        sum_j += bucket_i * 1 / unit
        pred_cdf[i] = sum_j
    pred_cdf[-1] = 1.0

    # print("True cdf: ", true_cdf)
    # print("Pred cdf: ", pred_cdf)
    wasser = np.max(np.abs(true_cdf - pred_cdf))

    return wasser


def cal_max_wasser_dis_interpolation(pmf_array, vel_list):
    round_vel = np.round(vel_list, decimals=0).astype(int)
    round_vel = round_vel[round_vel <= 40]
    sorted_array = np.sort(round_vel)
    true_cdf = np.zeros(41)
    sum_i = 0
    for i in range(41):
        if i in round_vel:
            record_i = np.sum(round_vel == i)
            sum_i += record_i / len(sorted_array)
        else:
            sum_i += 0
        true_cdf[i] = sum_i

    true_cdf2 = np.zeros(41)
    sum_i2 = 0
    number_record = len(sorted_array)
    prob_unit = 1. / number_record
    last_num = 0
    sorted_unique, count = np.unique(sorted_array, return_counts=True)
    for i, value in enumerate(sorted_unique):
        num_value = count[i]
        prob_i = prob_unit * num_value
        if i == -1:
            sum_i2 += prob_i
            true_cdf2[value] = sum_i2
            last_num = value
        else:
            for j in range(last_num + 1, value):
                true_cdf2[j] += \
                    prob_i / (value - last_num) * (j - last_num) + sum_i2
            sum_i2 += prob_i
            true_cdf2[value] = sum_i2
            last_num = value
    max_value = np.max(sorted_unique)
    true_cdf2[max_value:] = 1.

    pred_cdf = np.zeros(41)
    unit = 40 / len(pmf_array)
    sum_j = 0
    for i in range(40):
        bucket_i = pmf_array[int(i / unit)]
        sum_j += bucket_i * 1 / unit
        pred_cdf[i] = sum_j
    pred_cdf[-1] = 1.0

    # print("True cdf: ", true_cdf2)
    # print("Pred cdf: ", pred_cdf)
    wasser = np.max(np.abs(true_cdf2 - pred_cdf))

    return wasser


def eval_ks_test_vel_list(vel_list, y_pred, alpha=0.05):
    accept_num = 0
    for i in range(y_pred.shape[0]):
        pred_pmf = y_pred[i]
        vel_list_i = vel_list[i]
        max_dis = cal_max_wasser_dis(pred_pmf, vel_list_i)
        scale = np.sqrt((len(vel_list_i) + 40) / (len(vel_list_i) * 40)) * np.sqrt(-0.5 * np.log(alpha / 2))
        # scale = np.sqrt(2 / 40) * np.sqrt(-0.5 * np.log(alpha / 2))
        # print("max_dis  |   scale")
        # print("{}   |   {}".format(max_dis, scale))
        if max_dis < scale:
            accept_num += 1

    accept_rate = accept_num / y_pred.shape[0]

    return accept_rate


def eval_wasser_vel_list(vel_list, y_pred):
    wasser_dist_sum = 0
    for i in range(y_pred.shape[0]):
        pred_pmf = y_pred[i]
        vel_list_i = vel_list[i]
        max_dis = cal_max_wasser_dis(pred_pmf, vel_list_i)
        wasser_dist_sum += max_dis

    avg_wasser_dist = wasser_dist_sum / y_pred.shape[0]

    return avg_wasser_dist


def real_output(result_dict):
    final_gt = result_dict['ground_truth']
    final_pred = result_dict['prediction']
    final_weight = result_dict['weight']
    final_count = result_dict['count']
    final_vel_list = result_dict['vel_list']

    print("final weight shape", final_weight.shape)
    print("Final vel list shape", final_vel_list.shape)
    selected_pos = final_weight == 1
    y_true = final_gt[selected_pos]
    y_pred = final_pred[selected_pos]
    count_y = final_count[selected_pos]
    vel_list = final_vel_list[selected_pos]

    return y_true, y_pred, count_y, vel_list


def real_output_gt_pred(result_dict, thres=5):
    final_gt = result_dict['ground_truth']
    final_pred = result_dict['prediction']
    final_weight = result_dict['weight']
    final_count = result_dict['count']

    selected_pos = final_weight == 1
    y_true = final_gt[selected_pos]
    y_pred = final_pred[selected_pos]
    y_count = final_count[selected_pos]
    y_true = y_true[y_count > thres]
    y_pred = y_pred[y_count > thres]
    return y_true, y_pred


def real_output_gt_pd_ha(result_dict, HA):
    final_gt = result_dict['ground_truth']
    final_pred = result_dict['prediction']
    final_weight = result_dict['weight']
    final_vel_list = result_dict['vel_list']

    HA = np.expand_dims(HA, axis=0)
    final_HA = np.tile(HA, (final_weight.shape[0], 1, 1))

    selected_pos = final_weight == 1
    vel_list = final_vel_list[selected_pos]
    y_pred = final_pred[selected_pos]
    select_HA = final_HA[selected_pos]

    return vel_list, y_pred, select_HA

def real_output_gt_pd(result_dict):
    final_gt = result_dict['ground_truth']
    final_pred = result_dict['prediction']
    final_weight = result_dict['weight']
    final_vel_list = result_dict['vel_list']

    selected_pos = final_weight == 1
    vel_list = final_vel_list[selected_pos]
    y_pred = final_pred[selected_pos]

    return vel_list, y_pred


def evaluate_array(y_true, y_pred, count_y,
                   vel_list=None, method='KL',
                   alpha=0.05, count_thres=0):
    select_pos = count_y > count_thres
    print("Total number in evaluating is ", np.sum(select_pos))
    y_true = y_true[select_pos]
    y_pred = y_pred[select_pos]
    if vel_list is not None:
        vel_list = vel_list[select_pos]

    if method == 'KL':
        loss_value, loss_std = weighted_kl_div(y_true, y_pred)
    elif method == 'KS-test':
        loss_value = eval_ks_test_vel_list(vel_list, y_pred, alpha)
    elif method == 'emd':
        loss_value = weighted_emd(y_true, y_pred)
    elif method == 'wasser':
        loss_value = eval_wasser_vel_list(vel_list, y_pred)
    elif method == 'euclidean':
        loss_value = euclidean_dist(y_true, y_pred, p=2)
    elif method == 'sig_test':
        loss_value = eval_sig_test_vel_list(vel_list, y_pred)
        # loss_value = eval_sig_test(y_true, y_pred, vel_list)
    else:
        loss_value = None
        print("Please specify a valid metric...")

    return loss_value


def evaluate_result(result_dict, method='KL', alpha=0.05):
    y_true, y_pred, count_y, vel_list = real_output(result_dict)

    return evaluate_array(y_true, y_pred, count_y, vel_list, method, alpha)


def convert_to_one_hot(a, max_val=None):
    N = a.size
    data = np.ones(N, dtype=int)
    sparse_out = coo_matrix(
        (data, (np.arange(N), a.ravel())), shape=(N, max_val))
    return np.array(sparse_out.todense())


def mean_gt(batch_y):
    mean_y = np.mean(batch_y, axis=0)
    mean_y = softmax(mean_y, n_axis=-1, exp=False)

    return mean_y


def fill_mean(source, mean, zero_fill=True):
    num_record = source.shape
    tile_shape = [1] * len(num_record)
    tile_shape[0] = num_record[0]
    tile_shape = tuple(tile_shape)
    tile_mean = np.tile(mean, tile_shape)

    sum_source = np.sum(source, axis=-1)
    if zero_fill:
        selected_pos = sum_source < 0.5
    else:
        selected_pos = sum_source > 0.9
    source[selected_pos] = tile_mean[selected_pos]

    return source


def softmax(x, n_axis=-1, exp=True):
    if exp:
        # take the sum along the specified axis
        x = np.exp(x)
    else:
        # in case there's negative value in the output
        # x_min = np.expand_dims(np.min(x, axis=n_axis), n_axis)
        # x = (x - x_min)
        x[x < 0] = 0.

    ax_sum = np.expand_dims(np.sum(x, axis=n_axis), n_axis)

    return x / ax_sum


class BatchLoader(object):
    def __init__(self, sname, mode, target, sample_rate, win_size,
                 hist_range, s_month, s_date, e_month, e_date,
                 data_rm, batch_size=-1, coarsening_level=4,
                 conv_mode='gcnn', is_coarsen=True, is_predicting = False):


        base_dir = os.path.join('.\graphcompletionlib', 'data', sname)
        print(base_dir)
        if target == 'avg':
            data_dir = os.path.join(base_dir, '{}_{}'.format(sample_rate, win_size), mode, target,
                                    '{}_{}-{}_{}'.format(s_date,
                                                         s_month, e_date, e_month),
                                    'rm{}'.format(data_rm))
        else:
            data_dir = os.path.join(base_dir, '{}_{}'.format(sample_rate, win_size), mode, target,
                                    '{}_{}_{}'.format(
                                        hist_range[0], hist_range[-1] + 1, hist_range[1] - hist_range[0]),
                                    '{}_{}-{}_{}'.format(s_date,
                                                         s_month, e_date, e_month),
                                    'rm{}'.format(data_rm))

        dict_normal_fname = os.path.join(data_dir, 'dict_normal.pickle')
        train_data_dict_fname = os.path.join(
            data_dir, 'train_data_dict.pickle')
        validate_data_dict_fname = os.path.join(
            data_dir, 'validate_data_dict.pickle')
        Adj_fname = os.path.join(base_dir, 'edge_adj.pickle')



        if not os.path.exists(dict_normal_fname) or \
                not os.path.exists(train_data_dict_fname) or \
                not os.path.exists(validate_data_dict_fname) or \
                not os.path.exists(Adj_fname):
            print("Data does not exist! Creating datasets...")
            self.data_generator(base_dir, data_dir, sname, mode, target,
                                sample_rate, win_size, hist_range, s_month,
                                s_date, e_month, e_date, data_rm, is_predicting)

        print("Data exists! Loading datasets...")
        adj = pklLoad(Adj_fname)
        dict_normal = pklLoad(dict_normal_fname)
        train_data_dict = pklLoad(train_data_dict_fname)
        validate_data_dict = pklLoad(validate_data_dict_fname)
        log("Data loaded")

        if target == 'avg':
            self.y_scaler = dict_normal['velocity']
        else:
            self.y_scaler = None


        train_data = train_data_dict['velocity_x']
        train_labels = train_data_dict['velocity_y']
        train_label_weight = train_data_dict['weight_y']
        train_counts = train_data_dict['count_y']
        train_vel_lists = train_data_dict['vel_list']
        cat_train = train_data_dict['cat']
        con_train = train_data_dict['con']

        log(f"Train data loaded, sample count: {len(train_data)} ")

        test_data = validate_data_dict['velocity_x']
        test_labels = validate_data_dict['velocity_y']
        test_labels_weight = validate_data_dict['weight_y']
        test_counts = validate_data_dict['count_y']
        test_vel_lists = validate_data_dict['vel_list']
        cat_test = validate_data_dict['cat']
        con_test = validate_data_dict['con']

        log(f"Test data loaded, sample count: {len(test_data)} ")
        # self.mean_y = mean_gt(train_data)
        # self.mean_y[np.isnan(self.mean_y)] = 0.0
        # train_data = fill_mean(train_data, self.mean_y)
        # train_labels = fill_mean(train_labels, self.mean_y)
        # train_label_weight = np.ones(train_label_weight.shape)
        #
        # test_data = fill_mean(test_data, self.mean_y)

        if conv_mode == 'gcnn':
            perm_file = os.path.join(base_dir, 'adj_perm.pickle')
            graph_file = os.path.join(base_dir, 'perm_graphs.pickle')
            if os.path.exists(perm_file) and os.path.exists(graph_file):
                self.perm = pklLoad(perm_file)
                self.graphs = pklLoad(graph_file)
            else:
                self.graphs, self.perm = coarsening.coarsen(
                    adj, levels=coarsening_level, self_connections=False)
                pklSave(perm_file, self.perm)
                pklSave(graph_file, self.graphs)
            if is_coarsen:
                train_data = coarsening.perm_data_hist(train_data, self.perm)
                test_data = coarsening.perm_data_hist(test_data, self.perm)

        self.all_batches = []
        self.sizes = []


        log(f"Constrcuting batches... Batch size set to {batch_size}.")
        ######################
        # SW502 Modification
        ######################
        if(is_predicting):
            log("Constructing only test batches...")
            self.construct_batches_all_test(test_data, test_labels, test_labels_weight,
                                   cat_test, con_test, test_counts, test_vel_lists, batch_size)
        # End of mod.
        else:
            log("Splitting batches into train, val, tests... ")
            val_x, test_x, val_y, test_y, \
            val_y_weight, test_y_weight, \
            val_con, test_con, val_cat, \
            test_cat, val_count, test_count, \
            val_vel_list, test_vel_list = \
                train_test_split(test_data, test_labels, test_labels_weight,
                                 con_test, cat_test, test_counts,
                                 test_vel_lists, test_size=0.8)

            print("Reshaping tensors...")
            # Split train, val, tests data into batches
            self.construct_batches(train_data, train_labels, train_label_weight,
                                   cat_train, con_train, train_counts, train_vel_lists, batch_size)
            self.construct_batches(val_x, val_y, val_y_weight,
                                   val_cat, val_con, val_count, val_vel_list, batch_size)
            self.construct_batches(test_x, test_y, test_y_weight,
                                   test_cat, test_con, test_count, test_vel_list, batch_size)

        print("Data load done. Number of batches in train: %d, val: %d, test: %d"
              % (self.sizes[0], self.sizes[1], self.sizes[2]))

        self.adj = adj
        self.batch_idx = [0, 0, 0]






    def split_into_batch(self, data_array, batch_size=-1, sample_size=20):

        shape_list = list(data_array.shape)
        if batch_size == -1:
            batch_size = sample_size * \
                         int(math.floor(shape_list[0] / sample_size))

        data_array = data_array[: batch_size *
                                  int(math.floor(shape_list[0] / batch_size)), ...]
        shape_list[0] = batch_size
        reshape_size = [-1] + shape_list
        batches = list(data_array.reshape(reshape_size))

        return batches

    def construct_batches(self, train_data, train_labels, train_label_weight,
                          cat_train, con_train, count_train, vel_list, batch_size):


        # Split train, val, test data into batches
        train_data_batches = self.split_into_batch(train_data, batch_size, 20)
        train_label_batches = self.split_into_batch(
            train_labels, batch_size, 20)
        train_label_weight_batches = self.split_into_batch(
            train_label_weight, batch_size, 20)
        train_count_batches = self.split_into_batch(
            count_train, batch_size, 20)
        vel_list_batches = self.split_into_batch(vel_list, batch_size, 20)
        # cat_train_batches = self.split_into_batch(cat_train, batch_size)
        # con_train_batches = self.split_into_batch(con_train, batch_size)
        self.all_batches.append([train_data_batches, train_label_batches,
                                 train_label_weight_batches, train_count_batches,
                                 vel_list_batches])
        self.sizes.append(len(train_data_batches))

    def construct_batches_all_test(self, train_data, train_labels, train_label_weight,
                          cat_train, con_train, count_train, vel_list, batch_size):

        sample_size = 20
        # Split train, val, test data into batches
        train_data_batches = self.split_into_batch(train_data, batch_size, sample_size)
        train_label_batches = self.split_into_batch(
            train_labels, batch_size, sample_size)
        train_label_weight_batches = self.split_into_batch(
            train_label_weight, batch_size, sample_size)
        train_count_batches = self.split_into_batch(
            count_train, batch_size, sample_size)
        vel_list_batches = self.split_into_batch(vel_list, batch_size, sample_size)
        # cat_train_batches = self.split_into_batch(cat_train, batch_size)
        # con_train_batches = self.split_into_batch(con_train, batch_size)

        # Append to index 2 (index for test batches)
        self.all_batches.append([])
        self.all_batches.append([])

        # Insert at testing index.
        self.all_batches.append([train_data_batches, train_label_batches,
                                 train_label_weight_batches, train_count_batches,
                                 vel_list_batches])

        # Append to index 2 (index for test batches)
        self.sizes.append(0)
        self.sizes.append(0)
        self.sizes.append(len(train_data_batches))



    def next_batch(self, split_idx):
        # cycle around to beginning
        if self.batch_idx[split_idx] >= self.sizes[split_idx]:
            self.batch_idx[split_idx] = 0
        idx = self.batch_idx[split_idx]
        self.batch_idx[split_idx] = self.batch_idx[split_idx] + 1
        return self.all_batches[split_idx][0][idx], \
               self.all_batches[split_idx][1][idx], \
               self.all_batches[split_idx][2][idx], \
               self.all_batches[split_idx][3][idx], \
               self.all_batches[split_idx][4][idx]

    def reset_batch_pointer(self, split_idx, batch_idx=None):
        if batch_idx == None:
            batch_idx = 0
        self.batch_idx[split_idx] = batch_idx

    def data_generator(self, base_dir, data_dir, sname, mode, target,
                       sample_rate, win_size, hist_range, s_month,
                       s_date, e_month, e_date, data_rm, is_predicting):

        try:
            os.stat(data_dir)
        except:
            os.makedirs(data_dir)

        if sname == 'server_kdd':
            year = 2016
            Training_start_date = datetime(year, 7, 19)
        elif sname == 'chengdu':
            year = 2014
            Training_start_date = datetime(year, 8, 2)
        elif sname == 'sw502':
            year = 2016
            Training_start_date = datetime(year, 10, 31)

        Val_start_date = datetime(year, s_month, s_date)
        Val_end_date = datetime(year, e_month, e_date)

        print("Val_start_date: ", Val_start_date)
        print("Val_end_date: ", Val_end_date)
        print("Training start date: ", Training_start_date)


        cat_head = []  # ['time_index', 'dayofweek']
        con_head = []
        prep_param = {'data_dir': data_dir,
                      'base_dir': base_dir,
                      'server_name': sname,
                      'conf_dir': CONF_DIR,
                      'random_node': True,
                      'data_rm_ratio': data_rm,
                      'cat_head': cat_head,
                      'con_head': con_head,
                      'sample_rate': sample_rate,
                      'window_size': win_size,
                      'start_date': Training_start_date,
                      'small_threshold': 3.0,
                      'big_threshold': 40.0,
                      'min_nb': 5,
                      'test_start_date': Val_start_date,
                      'test_end_date': Val_end_date}
        try:
            if sname == 'server_kdd':
                dataset = ds.KDD_Data(**prep_param)
            elif sname == 'chengdu':
                prep_param['topk'] = 5000
                dataset = ds.GPS_Data(**prep_param)
            elif sname == 'sw502':
                prep_param['topk'] = 1000
                dataset = ds.MapMatchData(**prep_param)

            least_threshold = 0.5

            dict_normal, train_data_dict, validate_data_dict = \
                dataset.prepare_est_pred_with_date(
                    method=target,
                    window=win_size,
                    mode=mode,
                    hist_range=hist_range,
                    least=True,
                    least_threshold=least_threshold, is_predicting = is_predicting)
        except KeyboardInterrupt:
            print("Ctrl-C is pressed, quiting...")
            sys.exit(0)