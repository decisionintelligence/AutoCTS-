import os
import argparse
import numpy as np
import pandas as pd
import torch
import time
import json
import matplotlib.pyplot as plt
import utils
from utils import generate_data, get_adj_matrix, load_adj, masked_mae, masked_mape, masked_rmse, single_step_metric, \
    metric, load_PEMSD7_adj
from NAS_Net.genotypes import PRIMITIVES
from NAS_Net.st_net import Network

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Args for generating clean set')
parser.add_argument('--dataset', type=str, default='pems/PEMS03',
                    help='the location of  dataset')
parser.add_argument('--datatype', type=str, default='csv',
                    help='type of dataset')
parser.add_argument('--mode', type=str, default='noisy_seeds',
                    help='the training mode')
parser.add_argument('--seed', type=int, default=301, help='random seed')
parser.add_argument('--sample_num', type=int, default=20, help='number of archs')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--in_dim', type=int, default=9, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=170)
parser.add_argument('--hid_dim', type=int, default=32,
                    help='for residual_channels and dilation_channels')
parser.add_argument('--randomadj', type=bool, default=True,
                    help='whether random initialize adaptive adj')
parser.add_argument('--seq_len', type=int, default=12)
parser.add_argument('--output_len', type=int, default=12)
parser.add_argument('--task', type=str, default='multi')

parser.add_argument('--layers', type=int, default=4, help='number of cells')
parser.add_argument('--steps', type=int, default=4, help='number of nodes of a cell')
parser.add_argument('--lr', type=float, default=0.001, help='init learning rate')
# parser.add_argument('--lr_min', type=float, default=0.0, help='min learning rate')
# parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
# parser.add_argument('--grad_clip', type=float, default=5,
#                     help='gradient clipping')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--exp_id', type=int, default=0, help='the exp_id used to identify the experiment')
parser.add_argument('--ratio', nargs='+', type=float, default=[0.6, 0.2, 0.2])

args = parser.parse_args()
torch.set_num_threads(3)


class Random_NAS:
    def __init__(self, dataloader, adj_mx, scaler, save_dir):
        self.save_dir = save_dir
        self.dataloader = dataloader
        self.adj_mx = adj_mx
        self.scaler = scaler

        utils.set_seed(args.seed)
        if args.cuda:
            torch.backends.cudnn.deterministic = True

    @staticmethod
    def sample_arch(use_seed=True):

        if not use_seed:
            np.random.seed(None)

        num_ops = len(PRIMITIVES)
        n_nodes = args.steps

        arch = []  # 要改，不能选none？再加个pad函数，补齐为6个节点
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

        if not use_seed:
            np.random.seed(args.seed)
        return arch

    def run(self):

        archs = []

        if args.mode == 'clean_seeds':
            args.epochs = 100
            for i in range(args.sample_num):
                archs.append(Random_NAS.sample_arch())
        elif args.mode == 'noisy_seeds':
            args.epochs = 5
            for i in range(args.sample_num):
                archs.append(Random_NAS.sample_arch(use_seed=True))
            for i in range(args.sample_num):
                archs.append(Random_NAS.sample_arch(use_seed=False))
        elif args.mode == 'train':
            args.epochs = 100
            archs = np.load(f'../results/searched_archs/{args.dataset}/{args.seq_len}_param_{args.exp_id}.npy')
        elif args.mode == 'manual':
            args.epochs = 100

            archs = [[[0, 1], [0, 3], [1, 3], [1, 1], [2, 2], [2, 3], [3, 1]],
                     [[0, 2], [0, 0], [1, 0], [0, 1], [2, 1], [1, 2], [3, 3]],
                     [[0, 1], [0, 0], [1, 3], [1, 2], [2, 2], [2, 1], [3, 2]],
                     [[0, 3], [0, 2], [1, 2], [0, 1], [2, 1], [1, 4], [3, 3]],
                     [[0, 2], [0, 1], [1, 2], [0, 1], [2, 0], [2, 3], [3, 3]]]

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        for i in range(len(archs)):
            print(f'arch number: {i}')
            arch = archs[i]

            # 返回每个epoch的train valid test metrics,test仅用于观察
            t1 = time.time()
            info1, info2, info3 = train_arch_from_scratch(self.dataloader, self.adj_mx, self.scaler, arch)
            t2 = time.time()
            time_cost = t2 - t1

            self.result_process(arch, i, info1, info2, info3, time_cost)

    def result_process(self, arch, id, info1, info2, info3, time_cost):
        clean_set = []
        test_set = []
        train_set = []

        train_set.append({"arch": np.array(arch).tolist(), "info": np.array(info1).tolist()})
        clean_set.append({"arch": np.array(arch).tolist(), "info": np.array(info2).tolist()})
        test_set.append({"arch": np.array(arch).tolist(), "info": np.array(info3).tolist()})
        if args.mode == 'train':
            with open(self.save_dir + f'/train{id}_{args.exp_id}.json', "w") as fw:
                json.dump(train_set, fw)

            with open(self.save_dir + f'/test{id}_{args.exp_id}.json', 'w') as tw:
                json.dump(test_set, tw)

        with open(self.save_dir + f'/valid{id}_{args.exp_id}.json', "w") as vw:
            json.dump(clean_set, vw)

        train_mae = [info1[i][0] for i in range(args.epochs)]
        train_rmse = [info1[i][1] for i in range(args.epochs)]
        train_mape = [info1[i][2] for i in range(args.epochs)]

        valid_mae = [info2[i][0] for i in range(args.epochs)]
        valid_rmse = [info2[i][1] for i in range(args.epochs)]
        valid_mape = [info2[i][2] for i in range(args.epochs)]

        test_mae = [info3[i][0] for i in range(args.epochs)]
        test_rmse = [info3[i][1] for i in range(args.epochs)]
        test_mape = [info3[i][2] for i in range(args.epochs)]
        test_rrse = [info3[i][3] for i in range(args.epochs)]
        test_corr = [info3[i][4] for i in range(args.epochs)]

        bestid = np.argmin(valid_mae)

        with open(self.save_dir + f'/log_{args.exp_id}.txt', 'a') as f:
            print(f'arch:{arch}', file=f)
            print(f'total train time:{time_cost}', file=f)
            print(f'best_epoch:{bestid}', file=f)
            print(f'valid mae:{valid_mae[bestid]},valid rmse:{valid_rmse[bestid]},valid mape:{valid_mape[bestid]}',
                  file=f)
            if args.task == 'multi':
                print(f'mae:{test_mae[bestid]},rmse:{test_rmse[bestid]},mape:{test_mape[bestid]}', file=f)
            else:
                print(f'rrse:{test_rrse[bestid]},corr:{test_corr[bestid]}', file=f)

        figure_save_path = self.save_dir
        if not os.path.exists(figure_save_path):
            os.makedirs(figure_save_path)

        x = np.array([i for i in range(args.epochs)])

        if 'train' in args.mode or 'manual' in args.mode:
            plt.figure(3 * id)
            plt.plot(x, np.array(train_mae))
            plt.plot(x, np.array(valid_mae))
            plt.plot(x, np.array(test_mae))
            plt.title('mae')
            plt.xlabel('epochs')
            plt.ylabel('value')
            plt.legend(['train', 'valid', 'test'], loc='upper center')
            plt.savefig(os.path.join(figure_save_path, f'mae{id}_{args.exp_id}.png'))
            plt.clf()
            plt.close()

            plt.figure(3 * id + 1)
            plt.plot(x, np.array(train_rmse))
            plt.plot(x, np.array(valid_rmse))
            plt.plot(x, np.array(test_rmse))
            plt.title('rmse')
            plt.xlabel('epochs')
            plt.ylabel('value')
            plt.legend(['train', 'valid', 'test'], loc='upper center')
            plt.savefig(os.path.join(figure_save_path, f'rmse{id}_{args.exp_id}.png'))
            plt.clf()
            plt.close()

            plt.figure(3 * id + 2)
            plt.plot(x, np.array(train_mape))
            plt.plot(x, np.array(valid_mape))
            plt.plot(x, np.array(test_mape))
            plt.title('mape')
            plt.xlabel('epochs')
            plt.ylabel('value')
            plt.legend(['train', 'valid', 'test'], loc='upper center')
            plt.savefig(os.path.join(figure_save_path, f'mape{id}_{args.exp_id}.png'))
            plt.clf()
            plt.close()


def main():
    # Fill in with root output path
    adj_mx = np.zeros((args.num_nodes, args.num_nodes))
    if args.datatype == 'csv':
        data_dir = os.path.join('../data', args.dataset + '.csv')
        if args.dataset == 'sz_taxi/sz_speed':
            adj_mx = pd.read_csv('../data/sz_taxi/sz_adj.csv', header=None).values.astype(np.float32)
        if args.dataset == 'los_loop/los_speed':
            adj_mx = pd.read_csv('../data/los_loop/los_adj.csv', header=None).values.astype(np.float32)
    elif args.datatype == 'txt':
        data_dir = os.path.join('../data', args.dataset + '.txt')

    elif args.datatype == 'npz':
        data_dir = os.path.join('../data', args.dataset + '.npz')
        adj_dir = os.path.join('../data', args.dataset + '.csv')
        if args.dataset == 'NYC_TAXI/NYC_TAXI':
            adj_mx = pd.read_csv(adj_dir, header=None).values.astype(np.float32)
        elif args.dataset == 'NYC_BIKE/NYC_BIKE':
            adj_mx = pd.read_csv(adj_dir, header=None).values.astype(np.float32)
        elif args.dataset == 'pems/PEMS03':
            adj_mx = get_adj_matrix(adj_dir, args.num_nodes, id_filename='../data/pems/PEMS03.txt')
        elif args.dataset == 'PEMSD7M/PEMSD7M':
            adj_dir = os.path.join('../data/PEMSD7M/adj.npz')
            adj_mx = load_PEMSD7_adj(adj_dir)
        else:
            adj_mx = get_adj_matrix(adj_dir, args.num_nodes)
    elif args.datatype == 'npy':
        data_dir = os.path.join('../data', args.dataset + '.npy')
        adj_dir = os.path.join('../data', args.dataset + '_adj.npy')
        if os.path.exists(adj_dir):
            adj_mx = np.load(os.path.join('../data', args.dataset + '_adj.npy'))
    elif args.datatype == 'tsf':
        data_dir = os.path.join('../data', args.dataset + '.tsf')
    elif args.datatype == 'h5':
        data_dir = os.path.join('../data', args.dataset + '.h5')
        if 'metr-la' in args.dataset:
            adj_dir = '../data/METR-LA/adj_mx.pkl'
            _, _, adj_mx = load_adj(adj_dir)
        elif 'pems-bay' in args.dataset:
            adj_dir = '../data/PEMS-BAY/adj_mx_bay.pkl'
            _, _, adj_mx = load_adj(adj_dir)

    elif args.datatype == 'subset':
        data_dir = os.path.join('../subsets', args.dataset + '.npy')
        adj_dir = os.path.join('../subsets', args.dataset + '_adj.npy')
        if os.path.exists(adj_dir):
            adj_mx = np.load(adj_dir)
            args.num_nodes = adj_mx.shape[0]
        else:
            data = np.load(data_dir)
            args.num_nodes = data.shape[1]
            adj_mx = np.zeros((args.num_nodes, args.num_nodes))

    test_batch_size = 1
    if 'train' in args.mode or 'manual' in args.mode:
        save_dir = os.path.join('../results/test', args.dataset, f'task_{args.seq_len}')
    if args.mode == 'clean_seeds':
        save_dir = os.path.join('../seeds', f'clean_{args.seq_len}', args.dataset)
    if args.mode == 'noisy_seeds':
        save_dir = os.path.join('../seeds', f'noisy_{args.seq_len}', args.dataset)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(args)
    if args.cuda and not torch.cuda.is_available():
        args.cuda = False

    dataloader = generate_data(data_dir, args.task, args.seq_len, args.output_len, args.in_dim, args.datatype,
                               args.batch_size,
                               test_batch_size, args.ratio)
    scaler = dataloader['scaler']

    searcher = Random_NAS(dataloader, adj_mx, scaler, save_dir)
    searcher.run()


def train_arch_from_scratch(dataloader, adj_mx, scaler, arch):
    model = Network(adj_mx, scaler, args, arch)

    if args.cuda:
        model = model.cuda()

    # train
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    train_metrics_list = []
    valid_metrics_list = []
    test_metrics_list = []
    for epoch_num in range(args.epochs):
        print(f'epoch num: {epoch_num}')
        model = model.train()

        dataloader['train_loader'].shuffle()
        t2 = time.time()
        train_loss = []
        train_rmse = []
        train_mape = []
        for i, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            x = torch.Tensor(x).to(DEVICE)

            x = x.transpose(1, 3)

            y = torch.Tensor(y).to(DEVICE)  # [64, 12, 207, 2]

            y = y.transpose(1, 3)[:, 0, :, :]

            optimizer.zero_grad()
            output = model(x)  # [64, 12, 207, 1]
            output = output.transpose(1, 3)
            y = torch.unsqueeze(y, dim=1)
            predict = scaler.inverse_transform(output)  # unnormed x

            loss = masked_mae(predict, y, 0.0)  # y也是unnormed
            train_loss.append(loss.item())
            rmse = masked_rmse(predict, y, 0.0)
            train_rmse.append(rmse.item())
            mape = masked_mape(predict, y, 0.0)
            train_mape.append(mape.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        train_metrics_list.append((np.mean(train_loss), np.mean(train_rmse), np.mean(train_mape)))
        print(f'train epoch time: {time.time() - t2}')

        # eval
        with torch.no_grad():
            model = model.eval()

            valid_loss = []
            valid_rmse = []
            valid_mape = []
            for i, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                x = torch.Tensor(x).to(DEVICE)
                x = x.transpose(1, 3)
                y = torch.Tensor(y).to(DEVICE)
                y = y.transpose(1, 3)[:, 0, :, :]  # [64, 207, 12]

                output = model(x)
                output = output.transpose(1, 3)  # [64, 1, 207, 12]
                y = torch.unsqueeze(y, dim=1)
                predict = scaler.inverse_transform(output)

                loss = masked_mae(predict, y, 0.0)
                rmse = masked_rmse(predict, y, 0.0)
                mape = masked_mape(predict, y, 0.0)
                valid_loss.append(loss.item())
                valid_rmse.append(rmse.item())
                valid_mape.append(mape.item())
            valid_metrics_list.append((np.mean(valid_loss), np.mean(valid_rmse), np.mean(valid_mape)))
            print((np.mean(valid_loss), np.mean(valid_rmse), np.mean(valid_mape)))

        with torch.no_grad():
            model = model.eval()

            y_p = []
            y_t = torch.Tensor(dataloader['y_test']).to(DEVICE)
            y_t = y_t.transpose(1, 3)[:, 0, :, :]
            for i, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
                x = torch.Tensor(x).to(DEVICE)
                x = x.transpose(1, 3)

                output = model(x)
                output = output.transpose(1, 3)  # [64, 1, 207, 12]
                y_p.append(output.squeeze(1))

            y_p = torch.cat(y_p, dim=0)
            y_p = y_p[:y_t.size(0), ...]

            amae = []
            amape = []
            armse = []
            rrse = []
            corr = []
            if args.task == 'multi':
                for i in range(args.seq_len):
                    pred = scaler.inverse_transform(y_p[:, :, i])
                    real = y_t[:, :, i]

                    metrics = metric(pred, real)
                    print(f'{i + 1}, MAE:{metrics[0]}, MAPE:{metrics[1]}, RMSE:{metrics[2]}')
                    amae.append(metrics[0])
                    amape.append(metrics[1])
                    armse.append(metrics[2])
            else:
                pred = scaler.inverse_transform(y_p)
                real = y_t
                metrics = single_step_metric(pred, real)
                print(f'{i + 1}, RRSE:{metrics[0]}, CORR:{metrics[1]}')
                rrse.append(metrics[0])
                corr.append(metrics[1])

            test_metrics_list.append((np.mean(amae), np.mean(armse), np.mean(amape), np.mean(rrse), np.mean(corr)))
            if args.task == 'multi':
                print(f'On average over {args.seq_len} horizons, '
                      f'Test MAE: {np.mean(amae)}, Test MAPE: {np.mean(amape)}, Test RMSE: {np.mean(armse)}')
            else:
                print(
                    f'Test RRSE: {np.mean(rrse)}, Test CORR: {np.mean(np.mean(corr))}')

    return train_metrics_list, valid_metrics_list, test_metrics_list


if __name__ == '__main__':
    main()
