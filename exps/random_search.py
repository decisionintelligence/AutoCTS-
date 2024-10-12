import argparse
import os
import random
import time
import numpy as np
import torch
import json
import torch.nn as nn
import torch.optim as optim
from functools import cmp_to_key
from pathlib import Path
import concurrent.futures
import utils
from NAS_Net.AHC.ahc_engine import AHC, train_task_oriented_ahc, evaluate

from NAS_Net.genotypes import PRIMITIVES
from utils import AHC_DataLoader, AHC_DataLoader_linear
from exist_file_map import *
from scipy.stats import spearmanr
from tqdm import tqdm
import heapq

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Args for zero-cost NAS')
parser.add_argument('--seed', type=int, default=301, help='random seed')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=64)  # 要调整，64的时候效果更好？
parser.add_argument('--ahc_lr', type=float, default=0.0001)  # 要调整
parser.add_argument('--layers', type=int, default=4)  # 0.001+adam or 0.0001+adam
parser.add_argument('--d_model', type=int, default=128)  # 0.001+adam or 0.0001+adam

parser.add_argument('--steps', type=int, default=4, help='number of nodes of a cell')
parser.add_argument('--dataset', type=str, default='pems/PEMS03', help='location of dataset')
parser.add_argument('--sample_scale', type=int, default=200000, help='the number of samples')
parser.add_argument('--mode', type=str, default='train', help='the mode of the comparator')
parser.add_argument('--loader_mode', type=str, default='quadratic', help='[quadratic linear]')

parser.add_argument('--seq_len', type=int, default=12, help='the sequence length of the sample')
parser.add_argument('--exp_id', type=int, default=59, help='the exp_id used to identify the experiment')
parser.add_argument('--epochs', type=int, default=100)  # 架构比较器训练轮数
parser.add_argument('--num_threads', type=int, default=5)  # 并行找大小的线程数
parser.add_argument('--top_k', type=int, default=5)  # 找前几的架构

args = parser.parse_args()

torch.set_num_threads(3)


# best_arch: [[0,0],[0,3],[1,3],[1,1],[2,2],[2,3],[3,1]]
def load_noisy_set(data_path):
    valid_mae_list = []
    with Path(data_path).open() as fr:  # [18.11(15.20), 24.94, 15.17(15.15), 17.02(15.12)]
        for line in fr:
            if line[0] == '(':
                valid_mae_list.append(float(line[1:8]))
    # print(valid_mae_list)
    print(len(valid_mae_list))
    print(min(valid_mae_list))  # 只考虑平均值是不够的

    arch_acc = []
    for i in range(len(valid_mae_list) // 5):
        arch_acc.append(min(valid_mae_list[i * 5: (i + 1) * 5]))
    return arch_acc


def load_task_feature(feature_path):
    task_feature_dir = os.path.join('../task_feature', feature_path)
    task_feature = np.load(task_feature_dir).squeeze()
    return task_feature


def load_my_seeds(sets):
    task_dict = {}
    train_pairs = []
    train_set = []
    archs = np.load('../seeds/exist/noisy.npy')
    for set in sets:
        offset = offset_dict[set]
        acc_path = seed_dict[set]
        embedding_path = ts2vec_embedding_dict[set]
        acc = load_noisy_set(os.path.join('../seeds/exist', acc_path))
        arch = archs[(offset - 1) * 500:offset * 500]
        task = [set] * 500
        task_feature = load_task_feature(embedding_path)
        ahc_set = list(zip(task, arch, acc))
        task_dict[set] = task_feature

        train_pairs += generate_task_pairs(ahc_set)
        train_set += ahc_set

    return train_pairs, train_set, task_dict


def load_my_same_op_seeds(sets):
    task_dict = {}
    train_pairs = []
    noisy_sets = []
    for set in sets:
        embedding_path = ts2vec_embedding_dict[set]
        task_feature = load_task_feature(embedding_path)
        task_dict[set] = task_feature
        train_pair, noisy_set = load_noisy_seeds(f'../seeds/same_ops/{args.dataset}/trial3', set)
        train_pairs += train_pair
        noisy_sets += noisy_set
    return train_pairs, noisy_sets, task_dict


def load_pretrain_noisy_seeds(seq_len):
    root_dir = f'../seeds/noisy_{seq_len}'
    dirs = os.listdir(root_dir)
    task_dict = {}
    noisy_set = []

    for dir in dirs:
        sub_dir = os.path.join(root_dir, dir)
        sub_files = os.listdir(sub_dir)
        for sub_file in sub_files:
            file_path = os.path.join(sub_dir, sub_file)
            task_name = f'{seq_len}_' + dir + '/' + sub_file

            task_path = os.path.join('../task_feature', dir, sub_file, f'{seq_len}_ts2vec_task_feature.npy')
            task_feature = np.load(task_path)
            task_dict[task_name] = task_feature

            seeds_dir = os.listdir(file_path)
            arch_pairs = []
            for seed_dir in seeds_dir:
                if 'valid' in seed_dir:
                    with open(os.path.join(file_path, seed_dir), 'r') as f:
                        archs = json.load(f)
                    arch_pairs += archs
            if len(arch_pairs) < 40:
                print(f'{dir}_{sub_file} < 40')
            for arch_pair in arch_pairs:
                arch = arch_pair['arch']
                info = arch_pair['info'][:5]
                mae = sorted(info, key=lambda x: x[0])[0][0]
                noisy_set.append((task_name, arch, mae))

    return noisy_set, task_dict


def load_clean_08_seeds():
    # pems08上采样的干净种子
    clean_set = []
    for i in range(1, 4):
        if i == 1:
            dir = '../seeds/exist/clean_trans.json'
        else:
            dir = '../seeds/exist/' + f'clean_trans{i}.json'

        with open(dir, "r") as f:
            arch_pairs = json.load(f)

        # 采样的时候考虑了none算子，实际过程中发现考虑这个不好，直接去掉包含none的架构
        for arch_pair in arch_pairs:
            arch = arch_pair['arch']
            flag = False
            for p in range(len(arch)):
                arch[p][1] -= 1
                if arch[p][1] == -1:
                    flag = True
                    break
            if flag:
                continue

            info = arch_pair['info'][:100]
            mae = sorted(info, key=lambda x: x[0])[0][0]
            if mae < 50:
                clean_set.append(('pems0812', arch, mae))

    # 必须用numpy的shuffle
    # np.random.shuffle(clean_set)
    train_set = []
    valid_set = []

    for i, (task0, arch, mae) in enumerate(clean_set):
        small_gap = 0
        for j, (task1, arch2, mae2) in enumerate(valid_set):
            if abs(mae - mae2) < 0.08:
                small_gap = 1
                break
        if small_gap == 0:
            valid_set.append((task0, arch, mae))
        else:
            train_set.append((task0, arch, mae))

    valid_pairs = generate_task_pairs(valid_set)
    train_pairs = generate_task_pairs(train_set)

    return train_pairs, valid_pairs, train_set, valid_set


def load_clean_04_seeds():
    # pems08上采样的干净种子
    clean_set = []
    for i in range(1, 3):
        dir = f'../seeds/exist/cellout_clean04_{i}.json'

        with open(dir, "r") as f:
            arch_pairs = json.load(f)

        # 采样的时候考虑了none算子，实际过程中发现考虑这个不好，直接去掉包含none的架构
        for arch_pair in arch_pairs:
            arch = arch_pair['arch']
            flag = False
            for p in range(len(arch)):
                arch[p][1] -= 1
                if arch[p][1] == -1:
                    flag = True
                    break
            if flag:
                continue

            info = arch_pair['info'][:100]
            mae = sorted(info, key=lambda x: x[0])[0][0]
            if mae < 50:
                clean_set.append(('pems0412', arch, mae))

    # np.random.shuffle(clean_set)
    train_set = []
    valid_set = []

    for i, (task0, arch, mae) in enumerate(clean_set):
        small_gap = 0
        for j, (task1, arch2, mae2) in enumerate(valid_set):
            if abs(mae - mae2) < 0.08:
                small_gap = 1
                break
        if small_gap == 0:
            valid_set.append((task0, arch, mae))
        else:
            train_set.append((task0, arch, mae))

    valid_pairs = generate_task_pairs(valid_set)
    train_pairs = generate_task_pairs(train_set)

    return train_pairs, valid_pairs, clean_set, valid_set


def load_noisy_08_seeds():
    # pems08上采样的noisy种子
    noisy_set = []

    dir = '../seeds/exist/' + 'cellout_noisy08_1.json'
    with open(dir, "r") as f:
        arch_pairs = json.load(f)

    # 采样的时候考虑了none算子，实际过程中发现考虑这个不好，直接去掉包含none的架构
    for arch_pair in arch_pairs:
        arch = arch_pair['arch']
        flag = False
        for p in range(len(arch)):
            arch[p][1] -= 1
            if arch[p][1] == -1:
                flag = True
                break
        if flag:
            continue
        info = arch_pair['info'][-1]  # 还得比较一下第一个epoch的效果
        mae = info[0]
        if mae < 50:
            noisy_set.append(('pems0812', arch, mae))

    # np.random.shuffle(noisy_set)

    train_pairs = generate_task_pairs(noisy_set)

    return train_pairs


def load_noisy_seeds(noisy_data_dir, task):
    noisy_set = []
    arch_pairs = []
    files = os.listdir(noisy_data_dir)

    for file in files:
        with open(os.path.join(noisy_data_dir, file), "r") as f:
            archs = json.load(f)
        arch_pairs += archs
    for arch_pair in arch_pairs:
        arch = arch_pair['arch']
        info = arch_pair['info'][:5]
        mae = sorted(info, key=lambda x: x[0])[0][0]
        if mae < 50:
            noisy_set.append((task, arch, mae))

    # np.random.shuffle(noisy_set)
    train_pairs = generate_task_pairs(noisy_set)
    return train_pairs, noisy_set


def get_valid_test_seeds(noisy_data_dir, task):
    noisy_set = []
    arch_pairs = []
    files = os.listdir(noisy_data_dir)
    for file in files:
        with open(os.path.join(noisy_data_dir, file), "r") as f:
            archs = json.load(f)
        arch_pairs += archs
    for arch_pair in arch_pairs:
        arch = arch_pair['arch']
        info = arch_pair['info'][:5]
        mae = sorted(info, key=lambda x: x[0])[0][0]
        if mae < 50:
            noisy_set.append((task, arch, mae))

    noisy_train_set = []
    noisy_valid_set = []
    # valid和test需要固定，从留出来的一个数据集上筛选出valid和test，组成完全的pair，需要设置gap！
    for i, (task0, arch, mae) in enumerate(noisy_set):
        small_gap = 0
        for j, (task1, arch2, mae2) in enumerate(noisy_valid_set):
            if abs(mae - mae2) < 0.08:
                small_gap = 1
                break
        if small_gap == 0:
            noisy_valid_set.append((task0, arch, mae))
        else:
            noisy_train_set.append((task0, arch, mae))

    final_noisy_train_set = []
    noisy_test_set = []
    for i, (task0, arch, mae) in enumerate(noisy_train_set):
        small_gap = 0
        for j, (task1, arch2, mae2) in enumerate(noisy_test_set):
            if abs(mae - mae2) < 0.08:
                small_gap = 1
                break
        if small_gap == 0:
            noisy_test_set.append((task0, arch, mae))
        else:
            final_noisy_train_set.append((task0, arch, mae))

    valid_pairs = generate_task_pairs(noisy_valid_set)
    test_pairs = generate_task_pairs(noisy_test_set)
    return final_noisy_train_set, valid_pairs, test_pairs


def load_clean_seeds(clean_data_dir, task):
    clean_set = []
    arch_pairs = []
    files = os.listdir(clean_data_dir)
    for file in files:
        if 'valid' in file:
            with open(os.path.join(clean_data_dir, file), "r") as f:
                archs = json.load(f)
            arch_pairs += archs

    for arch_pair in arch_pairs:
        arch = arch_pair['arch']
        info = arch_pair['info'][:100]
        mae = sorted(info, key=lambda x: x[0])[0][0]
        if mae < 50:
            clean_set.append((task, arch, mae))

    # np.random.shuffle(clean_set)
    train_set = []
    valid_set = []

    for i, (task0, arch, mae) in enumerate(clean_set):
        small_gap = 0
        for j, (task1, arch2, mae2) in enumerate(valid_set):
            if abs(mae - mae2) < 0.08:
                small_gap = 1
                break
        if small_gap == 0:
            valid_set.append((task0, arch, mae))
        else:
            train_set.append((task0, arch, mae))

    valid_pairs = generate_task_pairs(valid_set)
    train_pairs = generate_task_pairs(train_set)

    return train_pairs, valid_pairs, train_set, valid_set


def load_clean_fine_tune_seeds(clean_data_dir, task):
    clean_set = []
    files = os.listdir(clean_data_dir)
    arch_pairs = []
    for file in files:
        if 'valid' in file:
            with open(os.path.join(clean_data_dir, file), "r") as f:
                archs = json.load(f)
            arch_pairs += archs

    for arch_pair in arch_pairs:
        arch = arch_pair['arch']
        info = arch_pair['info'][:100]
        mae = sorted(info, key=lambda x: x[0])[0][0]
        if mae < 50:
            clean_set.append((task, arch, mae))

    # np.random.shuffle(clean_set)
    train_pairs = generate_task_pairs(clean_set)

    return train_pairs, clean_set


def generate_task_pairs(data):
    # data: [(task, arch, loss)]
    pairs = []
    data = sorted(data, key=lambda x: x[2])
    for i in range(len(data) - 1):
        for j in range(i + 1, len(data)):
            assert data[i][0] == data[j][0]
            pairs.append((data[i][0], data[i][1], data[j][1], 1))
            pairs.append((data[i][0], data[j][1], data[i][1], 0))

    return pairs


def sample_arch():
    num_ops = len(PRIMITIVES)
    n_nodes = args.steps

    arch = []
    for i in range(n_nodes):
        if i == 0:
            ops = np.random.choice(range(num_ops), 1)
            nodes = np.random.choice(range(i + 1), 1)
            arch.extend([(nodes[0], ops[0])])
        else:
            ops = np.random.choice(range(num_ops), 2)  # 两条input edge对应两个op（可以相同）
            nodes = np.random.choice(range(i), 1)  # 只有一条可以选择的边
            # nodes = np.random.choice(range(i + 1), 2, replace=False)
            arch.extend([(nodes[0], ops[0]), (i, ops[1])])

    return arch


def calculate_spearmanr(ahc, task_dict, set):
    features, _, _ = zip(*set)
    task_feature = task_dict[features[0]]

    ahc.eval()

    def compare(x0, x1):
        with torch.no_grad():
            outputs = ahc([x0[1]], [x1[1]], [task_feature])
            pred = torch.round(outputs)
        if pred == 0:
            return 1
        else:
            return -1

    # 先按照ahc的比较规则对于所有排序，得到一个从小到大的顺序rank
    ahc_sorted_set = sorted(set, key=cmp_to_key(compare))
    ahc_sorted_index = list(range(len(ahc_sorted_set)))

    # 再根据真实的metric指标进行排序，然后获得每一个位置被排序后的大小rank
    _, _, accs = zip(*ahc_sorted_set)
    acc_sorted_index = np.argsort(accs)
    rank_index = [0] * len(ahc_sorted_set)
    for i, id in enumerate(acc_sorted_index):
        rank_index[id] = i

    rho = spearmanr(ahc_sorted_index, rank_index)

    return rho


def main():
    print(args)
    if args.cuda and not torch.cuda.is_available():
        args.cuda = False

    utils.set_seed(args.seed)
    if args.cuda:
        torch.backends.cudnn.deterministic = True
    if args.loader_mode == 'linear':
        DataLoader = AHC_DataLoader_linear
    else:
        DataLoader = AHC_DataLoader

    ahc = AHC(n_nodes=None, n_ops=len(PRIMITIVES), n_layers=args.layers, embedding_dim=args.d_model).to(DEVICE)
    criterion = nn.BCELoss()  # 要不要把学习率改成余弦退火？？？或者给GIN加dropout？？？
    ahc_optimizer = optim.Adam(ahc.parameters(), lr=args.ahc_lr)
    model_dir = '../NAS_Net/AHC/AHC_param'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if args.mode == 'pretrain':
        with open(model_dir + f'/train_log_{args.exp_id}.txt', 'w') as f:
            print(
                f'''pretrain the Task-Oriented AHC:
                the args:{args}''',
                file=f)

        ahc.train()
        noisy_12_set, task_dict_12 = load_pretrain_noisy_seeds(12)
        noisy_48_set, task_dict_48 = load_pretrain_noisy_seeds(48)
        # _, exist_noisy_train_set, task_dict_1 = load_my_seeds(
        #     ['pems0312', 'pems0412', 'la12', 'ETTh1_12', 'ETTm1_12', 'solar_12', 'exchange_rate_12'])
        noisy_same_train_pairs, noisy_same_train_set, task_dict_2 = load_my_same_op_seeds(
            ['pems0312', 'pems0412', 'la12', 'ETTh1_12', 'ETTm1_12', 'solar_12', 'exchange_rate_12'])
        exist_noisy_train_pairs, exist_noisy_train_set, task_dict_1 = load_my_seeds(
            ['pems0812'])
        task_dict = {**task_dict_12, **task_dict_48, **task_dict_2, **task_dict_1}
        # noisy_set = noisy_12_set + noisy_48_set + noisy_same_train_set + exist_noisy_train_set
        final_train_set, valid_pairs, test_pairs = get_valid_test_seeds(
            '/home/AutoCTS+_new/seeds/same_ops/pems/PEMS08/trial3',
            'pems0812')

        # noisy_set = final_train_set + exist_noisy_train_set
        noisy_set = generate_task_pairs(final_train_set) + exist_noisy_train_pairs

        task_dict['pems0812'] = load_task_feature('pems/PEMS08/12_ts2vec_task_feature.npy')
        noisy_valid_loader = AHC_DataLoader(valid_pairs, task_dict, args.batch_size)
        noisy_test_loader = AHC_DataLoader(test_pairs, task_dict, args.batch_size)
        train_loader = DataLoader(noisy_set, task_dict, args.batch_size)
        his_loss = 100
        tolerance = 0
        train_noisy_loop = tqdm(range(args.epochs), ncols=250, desc='pretrain ahc with noisy_set')

        for epoch in train_noisy_loop:  # 训练NAC多少个epochs？好像只能凭经验，因为没有test set
            train_loss, train_acc, valid_loss, acc = train_task_oriented_ahc(train_loader,
                                                                             noisy_valid_loader,
                                                                             ahc,
                                                                             criterion,
                                                                             ahc_optimizer)

            train_noisy_loop.set_description(f'Epoch {epoch}:')
            train_noisy_loop.set_postfix(train_loss=train_loss, train_acc=train_acc, valid_loss=valid_loss, acc=acc)
            with open(model_dir + f'/train_log_{args.exp_id}.txt', 'a') as f:
                print(
                    f'''Noisy Epoch {epoch}: train loss:{train_loss}, train acc:{train_acc}, valid loss:{valid_loss}, valid acc rate:{acc}, loss {"decreases and model is saved" if valid_loss < his_loss else "doesn't decrease"}''',
                    file=f)
            if valid_loss < his_loss:
                train_noisy_loop.set_description(f'valid loss decreases [{his_loss}->{valid_loss}]')

                tolerance = 0
                his_loss = valid_loss
                torch.save(ahc.state_dict(), model_dir + f"/AHC_{args.exp_id}.pth")
            else:
                tolerance += 1
            if tolerance >= 3:
                break
            # torch.save(ahc.state_dict(), model_dir + f"/AHC_{args.exp_id}.pth")

        ahc.load_state_dict(torch.load(model_dir + f"/AHC_{args.exp_id}.pth"))

        test_acc, test_loss = evaluate(test_loader=noisy_test_loader, ahc=ahc, criterion=criterion)
        with open(model_dir + f'/train_log_{args.exp_id}.txt', 'a') as f:
            print(
                f'''\nTEST RESULT: test loss:{test_loss}, test acc:{test_acc}''',
                file=f)

    elif args.mode == 'search':
        torch.set_num_threads(args.num_threads)
        ahc.load_state_dict(torch.load(os.path.join(model_dir, f'AHC_{args.exp_id}.pth')))

        task_feature = load_task_feature(os.path.join(args.dataset, f'{args.seq_len}_ts2vec_task_feature.npy'))
        ahc.eval()

        def compare(arch0, arch1):
            with torch.no_grad():
                outputs = ahc([arch0], [arch1], [task_feature])
                pred = torch.round(outputs)
            # pred == 0代表arch0的性能不如arch1
            if pred == 0:
                return -1
            else:
                return 1

        archs = []
        for i in range(args.sample_scale):
            archs.append(sample_arch())
        t1 = time.time()

        if args.num_threads > 1:
            # 并行
            slice_len = args.sample_scale // args.num_threads
            archs_slices = [
                archs[i * slice_len:(i + 1) * slice_len] if i < args.num_threads - 1 else archs[i * slice_len:]
                for i in range(args.num_threads)]

            def process_item(item):
                # 快速排序，控制比较次数在nlogn
                # return sorted(item, key=cmp_to_key(compare),reverse=True)
                # top-k堆，控制比较次数在klogn
                return heapq.nlargest(args.top_k, item, key=cmp_to_key(compare))

            # 使用 ThreadPoolExecutor 并行处理
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(process_item, archs_slices))

            # 要选择出全局的top_k个，在并行过程中需要把每个分块的top_k个都放进去，因为这样能保证真实的top_k个一定在这些里面
            archs = []
            for i in range(len(results)):
                archs += results[i][0:args.top_k]

            # 将args.num_threads * args.top_k个候选项从大到小排序一遍，得到真实的top_k个
            # 快速排序
            # sorted_archs = sorted(archs, key=cmp_to_key(compare), reverse=True)

            # 两两比较计分
            score_array = [0] * len(archs)
            for i in range(len(archs)):
                for j in range(i + 1, len(archs)):
                    if compare(archs[i], archs[j]) == 1:
                        score_array[i] += 1
                    else:
                        score_array[j] += 1
            indices = np.argsort(-np.array(score_array))
            sorted_archs = [archs[i] for i in indices]

        else:
            sorted_archs = sorted(archs, key=cmp_to_key(compare), reverse=True)

        # 保留排名前五的arch
        print(f'pred time: {time.time() - t1}')
        print(sorted_archs[:args.top_k])
        param_dir = os.path.join('../results/searched_archs', args.dataset)
        if not os.path.exists(param_dir):
            os.makedirs(param_dir)
        np.save(param_dir + f'/{args.seq_len}_param_{args.exp_id}.npy', np.array(sorted_archs[:args.top_k]))

        with open(param_dir + f'/{args.seq_len}_param_{args.exp_id}.txt', 'w') as f:
            for arch in sorted_archs[:args.top_k]:
                print(arch, file=f)


if __name__ == '__main__':
    main()
