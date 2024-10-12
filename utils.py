import pickle
import csv
from pathlib import Path
from scipy.sparse.linalg import eigs
import scipy.sparse as sp
import copy
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime
from distutils.util import strtobool
from scipy.stats import skew, kurtosis, entropy
from scipy.special import softmax
import random, os
from scipy.interpolate import interp1d


######################################################################
# dataset processing
######################################################################
class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        generate data batches
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        :param shuffle:
        """
        self.batch_size = batch_size
        self.current_ind = 0  # index
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]  # ...代替多个:
                y_i = self.ys[start_ind: end_ind, ...]
                yield x_i, y_i
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len * test_ratio):]
    val_data = data_len[-int(data_len * (val_ratio + test_ratio)): -int(data_len * test_ratio)]
    train_data = data[: -int(data_len * (val_ratio + test_ratio))]

    return train_data, val_data, test_data


def Add_Window_Horizon(data, window=3, horizon=1, single=False):
    """
    data format for seq2seq task or seq to single value task.
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :param single:
    :return: X is [B, W, ...], Y is [B, H, ...]
    """
    length = len(data)
    end_index = length - horizon - window + 1
    X = []  # windows
    Y = []  # horizon
    index = 0
    if single:  # 预测一个值
        while index < end_index:
            X.append(data[index: index + window])
            Y.append(data[index + window + horizon - 1: index + window + horizon])
            index += 1
    else:  # 预测下一个序列
        while index < end_index:
            X.append(data[index: index + window])
            Y.append(data[index + window: index + window + horizon])
            index += 1
    X = np.array(X).astype('float32')
    Y = np.array(Y).astype('float32')

    return X, Y


def load_dataset(data_dir, batch_size, test_batch_size=None, **kwargs):
    """
    generate dataset
    :param data_dir:
    :param batch_size:
    :param test_batch_size:
    :param kwargs:
    :return:
    """
    data = {}
    if 'pollution' not in data_dir and 'weather' not in data_dir:  # 数据集已分割
        for category in ['train', 'val', 'test']:
            cat_data = np.load(Path().joinpath(data_dir, category + '.npz'))
            data['x_' + category] = cat_data['x']
            data['y_' + category] = cat_data['y']
            # data['x_' + category] = cat_data['x'].astype('float32')  # astype导致性能下降，为什么？
            # data['y_' + category] = cat_data['y'].astype('float32')

        # 只考虑特征的第一维？计算loss和mse也是只考虑y的第一维
        scalar = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
        # Data format
        for category in ['train', 'val', 'test']:  # norm?
            data['x_' + category][..., 0] = scalar.transform(data['x_' + category][..., 0])

        train_len = len(data['x_train'])
        permutation = np.random.permutation(train_len)
        data['x_train_1'] = data['x_train'][permutation][:int(train_len / 2)]
        data['y_train_1'] = data['y_train'][permutation][:int(train_len / 2)]
        data['x_train_2'] = data['x_train'][permutation][int(train_len / 2):]
        data['y_train_2'] = data['y_train'][permutation][int(train_len / 2):]
        data['x_train_3'] = copy.deepcopy(data['x_train_2'])
        data['y_train_3'] = copy.deepcopy(data['y_train_2'])
        data['train_loader_1'] = DataLoader(data['x_train_1'], data['y_train_1'], batch_size)
        data['train_loader_2'] = DataLoader(data['x_train_2'], data['y_train_2'], batch_size)
        data['train_loader_3'] = DataLoader(data['x_train_3'], data['y_train_3'], batch_size)

        data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
        data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size)
        data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
        data['scaler'] = scalar

        return data
    else:  # 分割并生成数据集
        dataset = np.load(data_dir, allow_pickle=True)
        data_train, data_val, data_test = split_data_by_ratio(dataset, 0.1, 0.2)
        x_tr, y_tr = Add_Window_Horizon(data_train, 12, 12, False)
        x_tr_orig = x_tr.copy()
        x_val, y_val = Add_Window_Horizon(data_val, 12, 12, False)
        x_test, y_test = Add_Window_Horizon(data_test, 12, 12, False)
        data['x_train'] = x_tr
        data['y_train'] = y_tr
        data['x_val'] = x_val
        data['y_val'] = y_val
        data['x_test'] = x_test
        data['y_test'] = y_test

        real_scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
        # Data format
        for category in ['train', 'val', 'test']:
            for i in range(x_tr.shape[-1]):
                scaler = StandardScaler(mean=x_tr_orig[..., i].mean(), std=x_tr_orig[..., i].std())
                data['x_' + category][..., i] = scaler.transform(data['x_' + category][..., i])
            print('x_' + category, data['x_' + category].shape)

        data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
        data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size)
        data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
        data['scaler'] = real_scaler

        return data


def load_adj(pkl_filename):
    """
    为什么gw的邻接矩阵要做对称归一化，而dcrnn的不做？其实做了，在不同的地方，是为了执行双向随机游走算法。
    所以K-order GCN需要什么样的邻接矩阵？
    这个应该参考ASTGCN，原始邻接矩阵呢？参考dcrnn
    为什么ASTGCN不采用对称归一化的拉普拉斯矩阵？
    :param pkl_filename: adj_mx.pkl
    :return:
    """
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)

    return sensor_ids, sensor_id_to_ind, adj_mx
    # return sensor_ids, sensor_id_to_ind, adj_mx.astype('float32')


def load_pickle(pkl_filename):
    try:
        with Path(pkl_filename).open('rb') as f:
            pkl_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with Path(pkl_filename).open('rb') as f:
            pkl_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pkl_filename, ':', e)
        raise

    return pkl_data


def load_PEMSD7_adj(adj_path):
    adj = sp.load_npz(os.path.join(adj_path))
    adj = adj.todense()

    return adj


######################################################################
# generating chebyshev polynomials
######################################################################
def scaled_Laplacian(W):
    """
    compute \tilde{L}
    :param W: adj_mx
    :return: scaled laplacian matrix
    """
    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real  # k largest real part of eigenvalues

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    """
    compute a list of chebyshev polynomials from T_0 to T{K-1}
    :param L_tilde: scaled laplacian matrix
    :param K: the maximum order of chebyshev polynomials
    :return: list(np.ndarray), length: K, from T_0 to T_{K-1}
    """
    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


######################################################################
# generating diffusion convolution adj
######################################################################
def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


######################################################################
# metrics
######################################################################
def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    # mask = (loss > 1000)
    # print(loss[mask])
    # print(preds[mask])
    # print(labels[mask])
    return torch.mean(loss) * 100


def RRSE(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.sum((pred - true) ** 2)) / torch.sqrt(torch.sum((pred - true.mean()) ** 2))


def CORR(pred, true, mask_value=None):
    # input B, T, N, D or B, N, D or B, N
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(dim=1).unsqueeze(dim=1)
        true = true.unsqueeze(dim=1).unsqueeze(dim=1)
    elif len(pred.shape) == 3:
        pred = pred.transpose(1, 2).unsqueeze(dim=1)
        true = true.transpose(1, 2).unsqueeze(dim=1)
    elif len(pred.shape) == 4:
        # B, T, N, D -> B, T, D, N
        pred = pred.transpose(2, 3)
        true = true.transpose(2, 3)
    else:
        raise ValueError
    dims = (0, 1, 2)
    pred_mean = pred.mean(dim=dims)
    true_mean = true.mean(dim=dims)
    pred_std = pred.std(dim=dims)
    true_std = true.std(dim=dims)
    correlation = ((pred - pred_mean) * (true - true_mean)).mean(dim=dims) / (pred_std * true_std)
    index = (true_std != 0)
    correlation = (correlation[index]).mean()
    return correlation


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse


def single_step_metric(pred, real):
    rrse = RRSE(pred, real, 0.0).item()
    corr = CORR(pred, real, 0.0).item()

    return rrse, corr


######################################################################
# Exponential annealing for softmax temperature
######################################################################
class Temp_Scheduler(object):
    def __init__(self, total_epochs, curr_temp, base_temp, temp_min=0.05, last_epoch=-1):
        self.total_epochs = total_epochs
        self.curr_temp = curr_temp
        self.base_temp = base_temp
        self.temp_min = temp_min
        self.last_epoch = last_epoch
        self.step(last_epoch + 1)

    def step(self, epoch=None):
        return self.decay_whole_process()

    def decay_whole_process(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        # self.curr_temp = (1 - self.last_epoch / self.total_epochs) * (self.base_temp - self.temp_min) + self.temp_min
        # if self.curr_temp < self.temp_min:
        #     self.curr_temp = self.temp_min

        self.curr_temp = max(self.base_temp * 0.90 ** self.last_epoch, self.temp_min)

        return self.curr_temp


def generate_data(graph_signal_matrix_name, task, train_len, pred_len, in_dim, type, batch_size, test_batch_size=None,
                  ratio=[0.6, 0.2, 0.2],
                  transformer=None):
    """shape=[num_of_samples, 12, num_of_vertices, 1]"""
    data = data_preprocess(graph_signal_matrix_name, task, train_len, pred_len, in_dim, type, ratio)

    scalar = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())

    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scalar.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scalar  # 有没有问题？只用一半训练数据的时候

    return data


def generate_data_1(graph_signal_matrix_name, task, train_len, pred_len, in_dim, type, batch_size, test_batch_size=None,
                    ratio=[0.6, 0.2, 0.2],
                    transformer=None):
    """shape=[num_of_samples, 12, num_of_vertices, 1]"""
    arr = np.load(graph_signal_matrix_name, allow_pickle=True).astype('float')
    # 获取数组的形状
    nan_mask = np.isnan(arr)

    # 计算沿轴0的均值，但在计算之前检查轴上是否有NaN值
    mean_values = np.where(np.all(nan_mask, axis=2, keepdims=True), 0, np.nanmean(arr, axis=2, keepdims=True))

    # 使用 np.where 将NaN值替换为均值
    arr_filled = np.where(nan_mask, mean_values, arr)

    origin_data = arr_filled
    length = len(origin_data)

    train_ratio, val_ratio, test_ratio = ratio
    data = {}
    train_line, val_line = int(length * train_ratio), int(length * (train_ratio + val_ratio))
    for key, line1, line2 in (('train', 0, train_line),
                              ('val', train_line, val_line),
                              ('test', val_line, length)):
        x, _ = generate_seq(origin_data[line1: line2], task, train_len, 0, in_dim)
        print(x.shape)
        data['x_' + key] = x[:, :, :, :10].astype('float32')
        data['y_' + key] = x[:, :, :, -1:].astype('float32')

    scalar = None
    for i in range(data['x_train'].shape[-1]):
        scalar = StandardScaler(mean=data['x_train'][..., i].mean(), std=data['x_train'][..., i].std())
        for category in ['train', 'val', 'test']:
            data['x_' + category][..., i] = scalar.transform(data['x_' + category][..., i])

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scalar  # 有没有问题？只用一半训练数据的时候

    return data


def generate_from_train_val_test(origin_data, train_len, pred_len, in_dim, transformer=None):
    data = {}
    for key in ('train', 'val', 'test'):
        x, y = generate_seq(origin_data[key], train_len, pred_len, in_dim)
        data['x_' + key] = x.astype('float32')
        data['y_' + key] = y.astype('float32')
        # if transformer:  # 啥意思？
        #     x = transformer(x)
        #     y = transformer(y)

    return data


def generate_from_data(origin_data, length, task, train_len, pred_len, in_dim, ratio, transformer=None):
    """origin_data shape: [17856, 170, 3]"""
    data = generate_sample(origin_data, task, train_len, pred_len, in_dim)
    train_ratio, val_ratio, test_ratio = ratio
    train_line, val_line = int(length * train_ratio), int(length * (train_ratio + val_ratio))
    for key, line1, line2 in (('train', 0, train_line),
                              ('val', train_line, val_line),
                              ('test', val_line, length)):
        x, y = generate_seq(origin_data[line1: line2], task, train_len, pred_len, in_dim)
        print(x.shape)
        data['x_' + key] = x.astype('float32')
        data['y_' + key] = y.astype('float32')
        # if transformer:  # 啥意思？
        #     x = transformer(x)
        #     y = transformer(y)

    return data


def generate_sample(origin_data, task, train_len, pred_len, in_dim):
    data = {}
    data['origin'] = origin_data
    x, y = generate_seq(origin_data, task, train_len, pred_len, in_dim)
    data['x'] = x.astype('float32')
    data['y'] = y.astype('float32')
    return data


def generate_seq(data, task, train_length, pred_length, in_dim):
    if task == 'multi':
        seq = np.concatenate([np.expand_dims(
            data[i: i + train_length + pred_length], 0)
            for i in range(data.shape[0] - train_length - pred_length + 1)],
            axis=0)[:, :, :, 0: in_dim]
        if train_length == pred_length:
            return np.split(seq, 2, axis=1)
        else:
            return np.split(seq, [train_length], axis=1)
    elif task == 'single':
        return generate_seq_for_single_step(data, train_length, pred_length, in_dim)
    else:
        raise ValueError


def generate_seq_for_single_step(data, train_length, pred_index, in_dim):
    seq = np.concatenate([np.expand_dims(
        data[i: i + train_length + pred_index], 0)
        for i in range(data.shape[0] - train_length - pred_index + 1)],
        axis=0)[:, :, :, 0: in_dim]

    X, Y = np.split(seq, [train_length], axis=1)
    Y = Y[:, pred_index - 1:pred_index, :, :]
    return X, Y


def sample_split(data, train_length, overlap=0):
    seq = np.concatenate([np.expand_dims(
        data[i: i + train_length], 0)
        for i in range(0, data.shape[0] - train_length + 1, train_length - overlap)],
        axis=0)[:, :, :, :]

    return seq


def dim_uniform(origin_data):
    if origin_data.ndim == 1:
        data = origin_data.reshape((origin_data.shape[0], 1, 1))
    elif origin_data.ndim == 2:
        data = origin_data.reshape((origin_data.shape[0], origin_data.shape[1], 1))
    else:
        data = origin_data

    return data


def data_preprocess(data_path, task, train_len, pred_len, in_dim, type='csv', ratio=[0.6, 0.2, 0.2], transformer=None):
    if type == 'csv':
        origin_data = pd.read_csv(data_path)
        if 'date' in origin_data.columns:
            origin_data.set_index('date', inplace=True)

        origin_data = origin_data.values
        # origin_data = np.expand_dims(origin_data, -1)
        origin_data = dim_uniform(origin_data)

        length = len(origin_data)
        data = generate_from_data(origin_data, length, task, train_len, pred_len, in_dim, ratio)

    elif type == 'txt':
        origin_data = np.loadtxt(data_path, delimiter=',')

        origin_data = np.array(origin_data)
        origin_data = dim_uniform(origin_data)
        # origin_data = np.expand_dims(origin_data, -1)
        length = len(origin_data)
        data = generate_from_data(origin_data, length, task, train_len, pred_len, in_dim, ratio)
    elif type == 'tsf':
        origin_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(
            data_path)
        origin_data = origin_data.iloc[:, 1:]
        # origin_data = origin_data.T

        origin_data = np.array(origin_data.values)

        data = [[origin_data[i][0][j] for i in range(origin_data.size)] for j in range(origin_data[0][0].size)]
        origin_data = np.array(data)

        # origin_data = np.expand_dims(origin_data, -1)
        origin_data = dim_uniform(origin_data)
        length = len(origin_data)
        data = generate_from_data(origin_data, length, task, train_len, pred_len, in_dim, ratio)
    elif type == 'npz' or type == 'subset':
        origin_data = np.load(data_path)
        try:  # shape=[17856, 170, 3]
            keys = origin_data.keys()
            if 'train' in keys and 'val' in keys and 'test' in keys:
                data = generate_from_train_val_test(dim_uniform(origin_data['data']), train_len, pred_len, in_dim,
                                                    ratio, transformer)

            elif 'data' in keys:
                length = origin_data['data'].shape[0]
                data = generate_from_data(dim_uniform(origin_data['data']), length, task, train_len, pred_len, in_dim,
                                          ratio,
                                          transformer)

        except:
            length = origin_data.shape[0]
            data = generate_from_data(dim_uniform(origin_data), length, task, train_len, pred_len, in_dim, ratio,
                                      transformer)
    elif type == 'h5':
        origin_data = pd.read_hdf(data_path)
        origin_data = np.array(origin_data)
        origin_data = dim_uniform(origin_data)

        length = len(origin_data)
        data = generate_from_data(origin_data, length, task, train_len, pred_len, in_dim, ratio)
    elif type == 'npy':
        arr = np.load(data_path, allow_pickle=True).astype('float')
        # 获取数组的形状
        nan_mask = np.isnan(arr)

        # 计算沿轴0的均值，但在计算之前检查轴上是否有NaN值
        mean_values = np.where(np.all(nan_mask, axis=2, keepdims=True), 0, np.nanmean(arr, axis=2, keepdims=True))

        # 使用 np.where 将NaN值替换为均值
        arr_filled = np.where(nan_mask, mean_values, arr)

        origin_data = arr_filled
        length = len(origin_data)
        data = generate_from_data(origin_data, length, task, train_len, pred_len, in_dim, ratio)

    return data


def convert_tsf_to_dataframe(
        full_file_path_and_name,
        replace_missing_vals_with="NaN",
        value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                    len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                    len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                                numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)
        loaded_data = loaded_data.iloc[:, 1:3]
        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


# Example of usage
# loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(r"D:\大学学习\下载的东东\Edge浏览器下载\nn5_weekly_dataset.tsf")
# print(loaded_data)
def get_adj_matrix(distance_df_filename, num_of_vertices, type_='connectivity', id_filename=None):
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A

    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':  # 啥意思啊，表里有的就置1？
                A[i, j] = 1
                A[j, i] = 1
            elif type_ == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be connectivity or distance!")

    return A


def calculate_statistical(data, seq_len):
    N = data.shape[1]  # numnodes
    T = data.shape[0]  # length
    D = data.shape[2]

    Mean = np.mean(data, axis=0)
    Skewness = skew(data, axis=0)
    Kurtosis = kurtosis(data, axis=0)
    Variance = np.var(data, axis=0)
    Slope = np.zeros((N, D))
    StandardDeviation = np.std(data, axis=0)

    x = np.arange(T)
    for i in range(N):
        for j in range(D):
            Slope[i, j] = np.polyfit(x, data[:, i, j], 1)[0]

    entropy_data = softmax(data, axis=0)
    Entropy = np.zeros((N, D))
    for i in range(N):
        for j in range(D):
            Entropy[i, j] = entropy(entropy_data[:, i, j], base=2)

    Ema = np.zeros((N, D))
    for i in range(N):
        for j in range(D):
            Ema[i, j] = pd.Series(data[:, i, j]).ewm(span=5, adjust=False).mean().values[-1]

    Statistics = np.stack([Mean, Skewness, Kurtosis, Variance, Slope, StandardDeviation, Entropy, Ema])

    Statistics = Statistics.mean(axis=1)
    Statistics = Statistics.mean(axis=1)
    addition_feature = np.array([seq_len, T, N, D])
    Statistics = np.append(Statistics, addition_feature, axis=0)
    return Statistics


######################################################################
# MLP for spatial attention
######################################################################
class MLP(nn.Module):
    def __init__(self, hiddens, input_size, activation_function, out_act, dropout_ratio=0.):
        """
        多个线性层的叠加
        :param hiddens: 隐层维度列表
        :param input_size: memory_size
        :param activation_function: 每一层都采用相同的激活函数？为啥不搞个激活函数列表？
        :param out_act: 是否对输出层加激活函数，False表示不加
        :param dropout_ratio: 不加dropout？是因为效果不好吗？
        """
        super(MLP, self).__init__()
        # dropout_ratio = 0.2
        # layers = [nn.Dropout(dropout_ratio)]
        layers = []  # 包含线性层和相应的激活函数

        previous_h = input_size
        for i, h in enumerate(hiddens):
            # out_act为false的时候，输出层不加激活
            activation = None if i == len(hiddens) - 1 and not out_act else activation_function
            layers.append(nn.Linear(previous_h, h))

            if activation is not None:
                layers.append(activation)

            # layers.append(nn.Dropout(dropout_ratio))
            previous_h = h
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)


######################################################################
# Darts utils
######################################################################
if __name__ == '__main__':
    adj = get_adj_matrix('data/pems/PEMS08/PEMS08.csv', 170)
    print(adj)
    # dataloader = load_dataset('data/METR-LA', 64, 64)
    # train_iterator = dataloader['train_loader_1'].get_iterator()
    # val_iterator = dataloader['train_loader_2'].get_iterator()
    # train_val = dataloader['train_loader'].get_iterator()
    # print(len(list(train_iterator)))


######################################################################
# AHC dataset processing
######################################################################


class AHC_DataLoader(object):
    def __init__(self, arch_pairs, task_dict, batch_size, pad_with_last_sample=True):
        """
        generate data batches
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0  # index
        task_name, x0, x1, y = zip(*arch_pairs)
        if pad_with_last_sample:
            num_padding = (batch_size - (len(x0) % batch_size)) % batch_size
            task_name_padding = np.repeat(task_name[-1:], num_padding, axis=0)
            x0_padding = np.repeat(x0[-1:], num_padding, axis=0)
            x1_padding = np.repeat(x1[-1:], num_padding, axis=0)
            y_padding = np.repeat(y[-1:], num_padding, axis=0)
            task_name = np.concatenate([task_name, task_name_padding], axis=0)
            x0 = np.concatenate([x0, x0_padding], axis=0)
            x1 = np.concatenate([x1, x1_padding], axis=0)
            y = np.concatenate([y, y_padding], axis=0)
        self.size = len(x0)
        self.num_batch = int(self.size // self.batch_size)
        self.task_name = task_name
        self.x0 = x0
        self.x1 = x1
        self.y = y
        self.task_dict = task_dict

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        task_name, x0, x1, y = self.task_name[permutation], self.x0[permutation], self.x1[permutation], self.y[
            permutation]
        self.task_name = task_name
        self.x0 = x0
        self.x1 = x1
        self.y = y

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                task_i = [self.task_dict[i] for i in self.task_name[start_ind: end_ind, ...]]
                x0_i = self.x0[start_ind: end_ind, ...]  # ...代替多个:
                x1_i = self.x1[start_ind: end_ind, ...]
                y_i = self.y[start_ind: end_ind, ...]
                yield task_i, x0_i, x1_i, y_i
                self.current_ind += 1

        return _wrapper()


class AHC_DataLoader_1(object):
    def __init__(self, arch_pairs, task_dict, statistics_dict, batch_size, pad_with_last_sample=True):
        """
        generate data batches
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0  # index
        task_name, x0, x1, y = zip(*arch_pairs)
        if pad_with_last_sample:
            num_padding = (batch_size - (len(x0) % batch_size)) % batch_size
            task_name_padding = np.repeat(task_name[-1:], num_padding, axis=0)
            x0_padding = np.repeat(x0[-1:], num_padding, axis=0)
            x1_padding = np.repeat(x1[-1:], num_padding, axis=0)
            y_padding = np.repeat(y[-1:], num_padding, axis=0)
            task_name = np.concatenate([task_name, task_name_padding], axis=0)
            x0 = np.concatenate([x0, x0_padding], axis=0)
            x1 = np.concatenate([x1, x1_padding], axis=0)
            y = np.concatenate([y, y_padding], axis=0)
        self.size = len(x0)
        self.num_batch = int(self.size // self.batch_size)
        self.task_name = task_name
        self.x0 = x0
        self.x1 = x1
        self.y = y
        self.task_dict = task_dict
        self.statistics_dict = statistics_dict

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        task_name, x0, x1, y = self.task_name[permutation], self.x0[permutation], self.x1[permutation], self.y[
            permutation]
        self.task_name = task_name
        self.x0 = x0
        self.x1 = x1
        self.y = y

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                task_i = [self.task_dict[i] for i in self.task_name[start_ind: end_ind, ...]]
                statistics_i = [np.squeeze(self.statistics_dict[i])[0:256] for i in
                                self.task_name[start_ind: end_ind, ...]]
                x0_i = self.x0[start_ind: end_ind, ...]  # ...代替多个:
                x1_i = self.x1[start_ind: end_ind, ...]
                y_i = self.y[start_ind: end_ind, ...]
                yield task_i, statistics_i, x0_i, x1_i, y_i
                self.current_ind += 1

        return _wrapper()


class AHC_DataLoader_linear(object):
    def __init__(self, arch_pairs, task_dict, batch_size, pad_with_last_sample=True):
        """
        generate data batches
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0  # index
        self.if_padding = pad_with_last_sample
        self.task_dict = task_dict
        self.arch_pairs = arch_pairs

        self.generate_all_pairs()

    def generate_all_pairs(self):
        task_class_pairs = {}
        for arch_pair in self.arch_pairs:
            if arch_pair[0] in task_class_pairs.keys():
                task_class_pairs[arch_pair[0]].append(arch_pair)
            else:
                task_class_pairs[arch_pair[0]] = []
        arch_pairs = []

        for task_name in self.task_dict.keys():
            if task_name in task_class_pairs.keys():
                arch_pairs += self.generate_pairs(task_class_pairs[task_name])

        self.generated_arch_pairs = arch_pairs

        if self.if_padding:
            self.padding()

    def generate_pairs(self, arch_pair_set):
        arch_pairs = []
        arch_pair_set_0 = arch_pair_set.copy()
        np.random.shuffle(arch_pair_set_0)
        for i in range(len(arch_pair_set)):
            if arch_pair_set[i][2] < arch_pair_set_0[i][2]:
                arch_pairs.append((arch_pair_set[i][0], arch_pair_set[i][1], arch_pair_set_0[i][1], 1))
            else:
                arch_pairs.append((arch_pair_set[i][0], arch_pair_set[i][1], arch_pair_set_0[i][1], 0))
        return arch_pairs

    def padding(self):
        task_name, x0, x1, y = zip(*self.generated_arch_pairs)

        num_padding = (self.batch_size - (len(x0) % self.batch_size)) % self.batch_size
        task_name_padding = np.repeat(task_name[-1:], num_padding, axis=0)
        x0_padding = np.repeat(x0[-1:], num_padding, axis=0)
        x1_padding = np.repeat(x1[-1:], num_padding, axis=0)
        y_padding = np.repeat(y[-1:], num_padding, axis=0)
        task_name = np.concatenate([task_name, task_name_padding], axis=0)
        x0 = np.concatenate([x0, x0_padding], axis=0)
        x1 = np.concatenate([x1, x1_padding], axis=0)
        y = np.concatenate([y, y_padding], axis=0)
        self.generated_arch_pairs = list(zip(task_name, x0, x1, y))

    def shuffle(self):
        np.random.shuffle(self.arch_pairs)
        self.generate_all_pairs()

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            task_name, x0, x1, y = zip(*self.generated_arch_pairs)
            self.num_batch = len(task_name) // self.batch_size
            self.size = len(x0)
            task_name = np.array(task_name)
            x0 = np.array(x0)
            x1 = np.array(x1)
            y = np.array(y)
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                task_i = [self.task_dict[i] for i in task_name[start_ind: end_ind, ...]]
                x0_i = x0[start_ind: end_ind, ...]  # ...代替多个:
                x1_i = x1[start_ind: end_ind, ...]
                y_i = y[start_ind: end_ind, ...]
                yield task_i, x0_i, x1_i, y_i
                self.current_ind += 1

        return _wrapper()


class AHC_DataLoader_linear_1(object):
    def __init__(self, arch_pairs, task_dict, statistics_dict, batch_size, pad_with_last_sample=True):
        """
        generate data batches
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0  # index
        self.if_padding = pad_with_last_sample
        self.task_dict = task_dict
        self.arch_pairs = arch_pairs
        self.statistics_dict = statistics_dict

        self.generate_all_pairs()

    def generate_all_pairs(self):
        task_class_pairs = {}
        for arch_pair in self.arch_pairs:
            if arch_pair[0] in task_class_pairs.keys():
                task_class_pairs[arch_pair[0]].append(arch_pair)
            else:
                task_class_pairs[arch_pair[0]] = []
        arch_pairs = []

        for task_name in self.task_dict.keys():
            if task_name in task_class_pairs.keys():
                arch_pairs += self.generate_pairs(task_class_pairs[task_name])

        self.generated_arch_pairs = arch_pairs

        if self.if_padding:
            self.padding()

    def generate_pairs(self, arch_pair_set):
        arch_pairs = []
        arch_pair_set_0 = arch_pair_set.copy()
        np.random.shuffle(arch_pair_set_0)
        for i in range(len(arch_pair_set)):
            if arch_pair_set[i][2] < arch_pair_set_0[i][2]:
                arch_pairs.append((arch_pair_set[i][0], arch_pair_set[i][1], arch_pair_set_0[i][1], 1))
            else:
                arch_pairs.append((arch_pair_set[i][0], arch_pair_set[i][1], arch_pair_set_0[i][1], 0))
        return arch_pairs

    def padding(self):
        task_name, x0, x1, y = zip(*self.generated_arch_pairs)

        num_padding = (self.batch_size - (len(x0) % self.batch_size)) % self.batch_size
        task_name_padding = np.repeat(task_name[-1:], num_padding, axis=0)
        x0_padding = np.repeat(x0[-1:], num_padding, axis=0)
        x1_padding = np.repeat(x1[-1:], num_padding, axis=0)
        y_padding = np.repeat(y[-1:], num_padding, axis=0)
        task_name = np.concatenate([task_name, task_name_padding], axis=0)
        x0 = np.concatenate([x0, x0_padding], axis=0)
        x1 = np.concatenate([x1, x1_padding], axis=0)
        y = np.concatenate([y, y_padding], axis=0)
        self.generated_arch_pairs = list(zip(task_name, x0, x1, y))

    def shuffle(self):
        np.random.shuffle(self.arch_pairs)
        self.generate_all_pairs()

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            task_name, x0, x1, y = zip(*self.generated_arch_pairs)
            self.num_batch = len(task_name) // self.batch_size
            self.size = len(x0)
            task_name = np.array(task_name)
            x0 = np.array(x0)
            x1 = np.array(x1)
            y = np.array(y)
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                task_i = [self.task_dict[i] for i in task_name[start_ind: end_ind, ...]]
                statistics_i = [np.squeeze(self.statistics_dict[i])[0:256] for i in task_name[start_ind: end_ind, ...]]
                x0_i = x0[start_ind: end_ind, ...]  # ...代替多个:
                x1_i = x1[start_ind: end_ind, ...]
                y_i = y[start_ind: end_ind, ...]
                yield task_i, statistics_i, x0_i, x1_i, y_i
                self.current_ind += 1

        return _wrapper()


class AP_DataLoader(object):
    def __init__(self, arch_pairs, task_dict, statistics_dict, batch_size, pad_with_last_sample=True):
        """
        generate data batches
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0  # index
        self.task_dict = task_dict
        self.statistics_dict = statistics_dict
        task_name, x, y = zip(*arch_pairs)
        if pad_with_last_sample:
            num_padding = (batch_size - (len(x) % batch_size)) % batch_size
            x_padding = np.repeat(x[-1:], num_padding, axis=0)
            y_padding = np.repeat(y[-1:], num_padding, axis=0)
            task_name_padding = np.repeat(task_name[-1:], num_padding, axis=0)

            x = np.concatenate([x, x_padding], axis=0)
            y = np.concatenate([y, y_padding], axis=0)
            task_name = np.concatenate([task_name, task_name_padding], axis=0)

        self.size = len(x)
        self.num_batch = int(self.size // self.batch_size)
        self.x = x
        self.y = y
        self.task_name = task_name

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        x, y, task_name = self.x[permutation], self.y[permutation], self.task_name[permutation]
        self.x = x
        self.y = y
        self.task_name = task_name

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.x[start_ind: end_ind, ...]  # ...代替多个:
                y_i = self.y[start_ind: end_ind, ...]
                task_i = [self.task_dict[i] for i in self.task_name[start_ind: end_ind, ...]]
                statistics_i = [np.squeeze(self.statistics_dict[i])[0:256] for i in
                                self.task_name[start_ind: end_ind, ...]]
                yield x_i, task_i, statistics_i, y_i
                self.current_ind += 1

        return _wrapper()

    def __len__(self):
        return self.num_batch


def set_seed(seed):
    """
        Fix all seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def list_mle(y_pred, y_true, k=None):
    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(0)
        y_true = y_true.unsqueeze(0)

    if k is not None:
        sublist_indices = (y_pred.shape[1] * torch.rand(size=k)).long()
        y_pred = y_pred[:, sublist_indices]
        y_true = y_true[:, sublist_indices]

    # 降序还是升序本质影响了ap预测mae/accuracy的侧重
    _, indices = y_true.sort(descending=False, dim=-1)
    pred_sorted_by_true = y_pred.gather(dim=1, index=indices)
    cumsums = pred_sorted_by_true.exp().flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
    listmle_loss = torch.log(cumsums + 1e-10) - pred_sorted_by_true
    loss = listmle_loss.sum(dim=1).mean()
    return loss
