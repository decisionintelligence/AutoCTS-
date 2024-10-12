import random
import numpy as np
import argparse
import os
from utils import set_seed
import time
import datetime

import utils
from NAS_Net.ts2vec import TS2Vec
import NAS_Net.ts2vec_model.datautils as datautils
from NAS_Net.ts2vec_model.utils import init_dl_program, name_with_datetime



def save_checkpoint_callback(
        save_every=1,
        unit='epoch'
):
    assert unit in ('epoch', 'iter')

    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')

    return callback


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='The dataset name')
    parser.add_argument('run_name',
                        help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, required=True,
                        help='The data loader used to load the experimental data.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr_dims', type=int, default=128, help='The representation dimension (defaults to 320)')
    parser.add_argument('--sample_num', type=int, default=100, help='samples num')

    parser.add_argument('--max-train-length', type=int, default=3000,
                        help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None,
                        help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=301, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None,
                        help='The maximum allowed number of threads used by this process')
    parser.add_argument('--seq_len', type=int, default=12, help='train length')
    parser.add_argument('--ratio', nargs='+', type=float, default=[0.6, 0.2, 0.2])

    args = parser.parse_args()

    set_seed(args.seed)
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    print('Loading data... ', end='')
    if args.loader == 'PEMS':
        data, train_slice, valid_slice, test_slice, scaler = datautils.load_forecast_PEMS(args.dataset)
        train_data = data[:, train_slice]
    elif args.loader == 'ETT':
        data, train_slice, valid_slice, test_slice, scaler, n_covariate_cols = datautils.load_forecast_csv(
            args.dataset)
        train_data = data[:, train_slice]
    elif args.loader == 'h5':
        data, train_slice, valid_slice, test_slice, scaler, n_covariate_cols = datautils.load_forecast_h5(
            args.dataset, args.ratio)
        train_data = data[:, train_slice]
    elif args.loader == 'txt':
        data, train_slice, valid_slice, test_slice, scaler = datautils.load_forecast_txt(
            args.dataset)
        train_data = data[:, train_slice]
    elif args.loader == 'subset':
        data, train_slice, valid_slice, test_slice, scaler = datautils.load_subset_npy(args.dataset)
        train_data = data[:, train_slice]
    elif args.loader == 'csv':
        data, train_slice, valid_slice, test_slice, scaler = datautils.load_csv(args.dataset, args.ratio)
        train_data = data[:, train_slice]
    elif args.loader == 'npy':
        data, train_slice, valid_slice, test_slice, scaler = datautils.load_npy(args.dataset, args.ratio)
        train_data = data[:, train_slice]
    else:
        raise ValueError(f"Unknown loader {args.loader}.")

    print('done')

    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )

    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    run_dir = '../NAS_Net/ts2vec_model/training/' + args.dataset + '__' + name_with_datetime(args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    t = time.time()

    model = TS2Vec(
        input_dims=train_data.shape[-1],
        device=device,
        **config
    )

    print('Training...')
    loss_log = model.fit(
        train_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )
    model.save(f'{run_dir}/model.pkl')

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    print("Encoding...")
    all_repr = model.encode(
        data,
        causal=True,
        sliding_length=100,
        sliding_padding=200,
        batch_size=256
    )

    all_repr = all_repr.transpose(1, 0, 2)  # T x N x D
    all_repr = utils.sample_split(all_repr, args.seq_len, 0)  # bn x seq_len x N x D
    all_repr = all_repr.transpose(0, 2, 1, 3)  # bn x N x seq_len x D
    temp_repr = np.mean(all_repr, axis=1, keepdims=False)  # bn x seq_len x D

    sample_num = args.sample_num
    sample_index = sorted(np.random.choice(range(0, temp_repr.shape[0]), sample_num))
    sample_repr = temp_repr[sample_index]  # bc x seq_len x D

    dir = f'../task_feature/{args.dataset}'
    if not os.path.exists(dir):
        os.makedirs(dir)
    np.save(os.path.join(dir, f'{args.seq_len}_ts2vec_task_feature.npy'), sample_repr)

    print("Finished.")
