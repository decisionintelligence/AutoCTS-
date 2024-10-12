import os
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle
from .utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_forecast_h5(name, ratio):
    origin_data = pd.read_hdf(f'../data/{name}.h5', index_col='date', parse_dates=True)

    dt_embed = _get_time_features(origin_data.index)
    n_covariate_cols = dt_embed.shape[-1]

    data = origin_data.to_numpy()
    train_ratio, val_ratio, test_ratio = ratio
    train_slice = slice(None, int(train_ratio * len(data)))
    valid_slice = slice(int(train_ratio * len(data)), int((train_ratio + val_ratio) * len(data)))
    test_slice = slice(int((1 - test_ratio) * len(data)), None)

    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.transpose(data, (1, 0))
    data = np.expand_dims(data, axis=-1)

    if n_covariate_cols > 0:
        dt_embed_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_embed_scaler.transform(dt_embed), axis=0)

    data = np.concatenate([data, np.repeat(dt_embed, data.shape[0], axis=0)], axis=-1)

    return data, train_slice, valid_slice, test_slice, scaler, n_covariate_cols


def load_forecast_PEMS(name):
    origin_data = np.load(f'../data/{name}.npz')['data'][:, :, 0:1]

    data = np.reshape(origin_data, (origin_data.shape[0], -1))
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)

    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.reshape(data, (origin_data.shape[0], origin_data.shape[1], origin_data.shape[2]))

    data = np.transpose(data, (1, 0, 2))

    return data, train_slice, valid_slice, test_slice, scaler


def load_subset_npy(name):
    origin_data = np.load(f'../subsets/{name}.npy')[:, :, 0:1]

    data = np.reshape(origin_data, (origin_data.shape[0], -1))
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)

    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.reshape(data, (origin_data.shape[0], origin_data.shape[1], origin_data.shape[2]))

    data = np.transpose(data, (1, 0, 2))

    return data, train_slice, valid_slice, test_slice, scaler


def load_npy(name, ratio):
    arr = np.load(f'../data/{name}.npy', allow_pickle=True)[:, :, 0:].astype('float')

    # 获取数组的形状
    nan_mask = np.isnan(arr)

    # 计算沿轴0的均值，但在计算之前检查轴上是否有NaN值
    mean_values = np.where(np.all(nan_mask, axis=2, keepdims=True), 0, np.nanmean(arr, axis=2, keepdims=True))

    # 使用 np.where 将NaN值替换为均值
    arr_filled = np.where(nan_mask, mean_values, arr)

    origin_data = arr_filled
    train_ratio, val_ratio, test_ratio = ratio

    data = np.reshape(origin_data, (origin_data.shape[0], -1))
    train_slice = slice(None, int(train_ratio * len(data)))
    valid_slice = slice(int(train_ratio * len(data)), int((train_ratio + val_ratio) * len(data)))
    test_slice = slice(int((1 - test_ratio) * len(data)), None)

    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.reshape(data, (origin_data.shape[0], origin_data.shape[1], origin_data.shape[2]))

    data = np.transpose(data, (1, 0, 2))

    return data, train_slice, valid_slice, test_slice, scaler


def load_forecast_txt(name):
    data = np.loadtxt(f'../data/{name}.txt', delimiter=',')
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)

    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, -1)

    data = np.transpose(data, (1, 0, 2))

    return data, train_slice, valid_slice, test_slice, scaler


def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)


def load_csv(name, ratio):
    if 'ETT' in name:
        data = pd.read_csv(f'../data/{name}.csv', index_col='date', parse_dates=True)
    else:
        data = pd.read_csv(f'../data/{name}.csv', parse_dates=False)
    # if 'date' in data.columns:
    #     data = data.set_index('date')
    #     dt_embed = _get_time_features(data.index)
    #     n_covariate_cols = dt_embed.shape[-1]
    # else:
    #     n_covariate_cols = 0
    if 'date' in data.columns:
        data = data.drop(columns='date')

    n_covariate_cols = 0
    dt_embed = None

    train_ratio, val_ratio, test_ratio = ratio

    data = data.values

    train_slice = slice(None, int(train_ratio * len(data)))
    valid_slice = slice(int(train_ratio * len(data)), int((train_ratio + val_ratio) * len(data)))
    test_slice = slice(int((1 - test_ratio) * len(data)), None)

    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)

    data = np.expand_dims(data, 0)
    data = np.transpose(data, (2, 1, 0))

    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([data, np.repeat(dt_embed, data.shape[0], axis=0)], axis=-1)

    return data, train_slice, valid_slice, test_slice, scaler


def load_forecast_csv(name, univar=False):
    data = pd.read_csv(f'../data/{name}.csv', index_col='date', parse_dates=True)
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]

    name = name.split('/')[-1]
    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]
        elif name == 'electricity':
            data = data[['MT_001']]
        else:
            data = data.iloc[:, -1:]

    data = data.to_numpy()
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12 * 30 * 24)
        valid_slice = slice(12 * 30 * 24, 16 * 30 * 24)
        test_slice = slice(16 * 30 * 24, 20 * 30 * 24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12 * 30 * 24 * 4)
        valid_slice = slice(12 * 30 * 24 * 4, 16 * 30 * 24 * 4)
        test_slice = slice(16 * 30 * 24 * 4, 20 * 30 * 24 * 4)
    else:
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)

    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    if name in ('electricity'):
        data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    else:
        data = np.expand_dims(data, 0)

    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([data, np.repeat(dt_embed, data.shape[0], axis=0)], axis=-1)

    return data, train_slice, valid_slice, test_slice, scaler, n_covariate_cols


def gen_ano_train_data(all_train_data):
    maxl = np.max([len(all_train_data[k]) for k in all_train_data])
    pretrain_data = []
    for k in all_train_data:
        train_data = pad_nan_to_target(all_train_data[k], maxl, axis=0)
        pretrain_data.append(train_data)
    pretrain_data = np.expand_dims(np.stack(pretrain_data), 2)
    return pretrain_data
