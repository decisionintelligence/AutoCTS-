import os
import argparse
import numpy as np
import random
import torch
import math
import utils
from utils import get_adj_matrix, load_adj, data_preprocess

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Args for generating clean set')
parser.add_argument('--dataset', type=str, default='pems/PEMS03',
                    help='the location of  dataset')
parser.add_argument('--datatype', type=str, default='csv',
                    help='type of dataset')
parser.add_argument('--seed', type=int, default=301, help='random seed')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=170)
parser.add_argument('--seq_len', type=int, default=12)
parser.add_argument('--sample_num', type=int, default=20, help='num of the subsets of each dataset')

args = parser.parse_args()
torch.set_num_threads(3)

# 按照数据集自身的长度的比例设置变量维数和时间戳长度的区间，同时设置上下界，筛除一些过小或者过大的样例
variable_sup = 300
variable_inf = 3
time_sup = 25000
time_inf = 7000

rates = [0.25, 0.5, 0.75, 1.0]
gap = 0.25


def random_sample(origin_data, adj_mx):
    root_dir = f"../subsets/{args.dataset.split('/')[0]}"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    save_dir = f'../subsets/{args.dataset}'

    variable_len, time_len = origin_data.shape[1], origin_data.shape[0]
    max_sample_num = args.sample_num
    sample_num = math.ceil(max_sample_num / len(rates) ** 2)

    variable_l = variable_inf
    variable_r = min(variable_sup, variable_len)

    time_l = time_inf
    time_r = min(time_sup, time_len)

    variable_sample_num = []
    time_sample_length = []
    cnt = 0
    for i in rates:  # variable
        for j in rates:  # time

            variable_s = variable_l + round((i - gap) * (variable_r - variable_l))
            variable_e = variable_s + round(gap * (variable_r - variable_l))
            variable_num = np.random.choice(
                range(variable_s, variable_e),
                sample_num)

            time_s = time_l + round((j - gap) * (time_r - time_l))
            time_e = time_s + round(gap * (time_r - time_l))
            time_length = np.random.choice(
                range(time_s, time_e),
                sample_num)

            for num, length in zip(variable_num, time_length):
                variable_sample_num.append(num)
                time_sample_length.append(length)

            cnt += 1

    variable_sample_id = [sorted(random.sample(range(0, variable_len), variable_sample_num[i])) for i in
                          range(len(variable_sample_num))]

    time_sample_index_l = [random.randint(0, time_len - time_sample_length[i]) for i in
                           range(len(time_sample_length))]

    time_sample_index_r = [time_sample_index_l[i] + time_sample_length[i] for i in range(len(time_sample_length))]

    time_sample_interval = list(zip(time_sample_index_l, time_sample_index_r))

    subset = []
    for i in range(len(time_sample_interval)):
        time_l, time_r = sorted(time_sample_interval[i])
        subset.append(origin_data[time_l:time_r, variable_sample_id[i], ...])
        with open(save_dir + '.log', 'a') as f:
            print(f'the No.{i} subset:', file=f)
            print(f'the shape of subset:({variable_sample_num[i]},{time_r - time_l},1)', file=f)
            print(f'the variable chosen:{variable_sample_id[i]}', file=f)
            print(f'the time interval:[{time_l}:{time_r}]\n', file=f)

    for i in range(len(subset)):
        set = subset[i]
        dir = save_dir + f'_{i}'
        np.save(dir, set)

    if not np.all(adj_mx == 0):
        for i in range(len(time_sample_interval)):
            id_list = variable_sample_id[i]
            new_adj_mx = np.zeros((len(id_list), len(id_list)))
            for j in range(len(id_list)):
                for k in range(j + 1, len(id_list)):
                    if adj_mx[id_list[j]][id_list[k]] == 1:
                        new_adj_mx[j][k] = 1
                    else:
                        new_adj_mx[j][k] = 0
                    if adj_mx[id_list[k]][id_list[j]] == 1:
                        new_adj_mx[k][j] = 1
                    else:
                        new_adj_mx[k][j] = 0
            dir = save_dir + f'_{i}_adj'
            np.save(dir, new_adj_mx)


def main():
    utils.set_seed(args.seed)
    adj_mx = np.zeros((args.num_nodes, args.num_nodes))
    if args.datatype == 'csv':
        data_dir = os.path.join('../data', args.dataset + '.csv')

    elif args.datatype == 'txt':
        data_dir = os.path.join('../data', args.dataset + '.txt')

    elif args.datatype == 'npz':
        data_dir = os.path.join('../data', args.dataset + '.npz')

        adj_dir = os.path.join('../data', args.dataset + '.csv')
        if args.dataset == 'pems/PEMS03':
            adj_mx = get_adj_matrix(adj_dir, args.num_nodes, id_filename='../data/pems/PEMS03.txt')
        else:
            adj_mx = get_adj_matrix(adj_dir, args.num_nodes)

    elif args.datatype == 'tsf':
        data_dir = os.path.join('../data', args.dataset + '.tsf')
    elif args.datatype == 'h5':
        data_dir = os.path.join('../data', args.dataset + '.h5')
        if 'metr-la' in args.dataset:
            adj_dir = '../data/METR-LA/adj_mx.pkl'
            _, _, adj_mx = load_adj(adj_dir)

    data = data_preprocess(data_dir, args.seq_len, args.seq_len, args.in_dim, args.datatype)
    origin_data = data['origin']
    random_sample(origin_data, adj_mx)


if __name__ == '__main__':
    main()
