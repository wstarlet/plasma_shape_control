# -*-coding:Utf-8-*
# @File: dynamics_train_in44_out25_update.py
# author: wnn
# Time: 2025/6/9

import numpy as np
import argparse
import torch
import torch.nn.functional as F
import sys, os
import json
import random
from torch.utils.data import DataLoader, Dataset, TensorDataset
from data_process.data_process_train_test import data_process_train, data_process_test
from data_process.min_max_scale_load import min_max_scale_load
from dynamics.dynamics_ensemble_ddp_update import EnsembleDynamics

# DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

import socket
from contextlib import closing

# Specify the worker card, with the first one as the primary, to print the training log information.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def setup_ddp(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()

curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add to system path

min_max_path = os.path.abspath(os.path.join(curr_path, '.', 'data', '250331', 'min_max_scale_1241_11908'))
max_ic, min_ic, max_u, min_u, max_TF, min_TF, max_vloop, min_vloop, max_ip, min_ip, max_r, min_r,\
max_z, min_z, max_a, min_a, max_e, min_e, max_Tr_down, min_Tr_down, max_Tr_up, min_Tr_up = min_max_scale_load(min_max_path)

max_var_out = np.concatenate((max_ic, max_a, max_e, max_Tr_down, max_Tr_up, max_ip, max_r, max_z))
min_var_out = np.concatenate((min_ic, min_a, min_e, min_Tr_down, min_Tr_up, min_ip, min_r, min_z))
var_out_sub = max_var_out - min_var_out

number = np.load(r"data/250331/data_1241_11908_250331_half_ramp_no_down/number.npy", allow_pickle=True)
t_full = np.load(r"data/250331/data_1241_11908_250331_half_ramp_no_down/t.npy", allow_pickle=True)

def model_test(args, test_data_index, test_number, test_t, test_load_path, test_save_folder):
    test_data = data_process_test(test_data_index, args.seq_len, args.seq_interval, args.mean_len)
    resp_model = EnsembleDynamics(args)
    resp_model.model_test_ensemble(test_data, test_number, test_t, test_load_path, test_save_folder)

def load_parameters(param_file):
    with open(param_file, 'r') as f:
        params = json.load(f)
    return params

def main_worker(rank, world_size, args, free_port):
    """
        Main function for each DDP process.
    """
    print(f"Running DDP on rank {rank}.")
    setup_ddp(rank, world_size, free_port)

    if args.seed is not None:
        seed = args.seed + rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Rank {rank} seed set to {seed}")

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    args.device = device
    print(f"Rank {rank} is using device: {device}")

    train_data_norm = data_process_train(args.train_data_index, args.seq_len, args.mean_len,
                                                     args.seq_interval, args.smooth_type)
    train_x, train_y = train_data_norm[0], train_data_norm[1]
    train_dataset = TensorDataset(train_x, train_y)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler
    )

    val_data, mask, origin_length = None, None, None
    if rank == 0:
        test_data = data_process_test(args.test_data_index, args.seq_len, args.seq_interval, args.mean_len)
        test_data_padded = []
        mask_list = []
        max_length = max([sequence.shape[0] for sequence, _ in test_data])
        origin_length_list = []
        for j in range(len(test_data)):
            data, target = test_data[j]
            origin_length_list.append(data.size(0))
            padded_data = F.pad(data, pad=(0, 0, 0, 0, 0, max_length - data.size(0)))
            padded_target = F.pad(target, pad=(0, 0, 0, max_length - target.size(0)))
            mask_list.append([1] * data.size(0) + [0] * (max_length - data.size(0)))
            test_data_padded.append((padded_data, padded_target))

        val_data = test_data_padded
        mask = torch.tensor(mask_list).to(device)
        origin_length = torch.tensor(origin_length_list).reshape(-1, 1).to(device)

    resp_model = EnsembleDynamics(args, rank)

    resp_model.train(train_loader, train_sampler, val_data, mask, origin_length)

    cleanup_ddp()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--mean_len', type=int, default=5)
    parser.add_argument('--seq_interval', type=int, default=15)
    parser.add_argument('--dynamics_ensemble', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--train_epochs', type=int, default=300)
    parser.add_argument('--in_dim', type=int, default=44)
    parser.add_argument('--out_dim', type=int, default=25)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr_dynamics', type=float, default=1e-3)
    parser.add_argument('--lr_weight', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.98)  # decay rate for lr
    parser.add_argument('--use_SS', type=bool, default=True)  # scheduled sampling
    parser.add_argument('--use_AW', type=bool, default=True)  # auto weight adjustment
    parser.add_argument('--use_LN', type=bool, default=False)  # LayerNorm
    parser.add_argument('--use_Attention', type=bool, default=True)  # use attention in LSTM
    parser.add_argument('--init', type=bool, default=False)  # auto weight init
    parser.add_argument('--norm', type=bool, default=False)  # auto weight norm
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--smooth_type', type=int, default=1)
    parser.add_argument('--initial_ratio', type=float, default=1)
    parser.add_argument('--final_ratio', type=float, default=0)
    parser.add_argument('--eval_type', type=str, default='mean')  # single or mean
    parser.add_argument('--decay_epoch', type=int, default=5)

    # file setting
    parser.add_argument('--save_path', default=curr_path +'/dynamics/dynamics_model_train_log_250331/in44_out25_dynamics/')
    args = parser.parse_args()
    args.load_path = None
    args.seed = 1

    if args.eval_type == 'single':
        args.test_min_loss = [100] * args.dynamics_ensemble
    elif args.eval_type == 'mean':
        args.test_min_loss = 100

    full_index = np.arange(len(number))

    delete_shot_base_6226 = [6602, 6603, 6605, 6606, 6607, 6608, 6611, 11649, 17716, 11667, 11665]
    delete_shot_base_6226_index = []
    for delete_shot in delete_shot_base_6226:
        idx = np.where(number == delete_shot)[0]
        if len(idx) > 0:
            delete_shot_base_6226_index.append(idx[0])

    index_shot_before_5000 = [i for i, num in enumerate(number) if num < 5000]
    off_index = np.concatenate((delete_shot_base_6226_index, index_shot_before_5000))

    valid_indices = np.setdiff1d(full_index, off_index)

    np.random.seed(args.seed)
    np.random.shuffle(valid_indices)

    test_set_size = int(len(valid_indices) * 0.2)
    test_data_index = valid_indices[:test_set_size]
    train_data_index = valid_indices[test_set_size:]

    test_number = number[test_data_index]
    test_t = t_full[test_data_index]
    args.train_len = len(train_data_index)
    args.test_len = len(test_data_index)
    args.test_number = test_number
    args.var_out_sub = var_out_sub
    args.min_var_out = min_var_out

    args.train_data_index = train_data_index
    args.test_data_index = test_data_index

    free_port = find_free_port()
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs. Spawning DDP processes on port {free_port}.")

    mp.spawn(main_worker,
             args=(world_size, args, free_port),
             nprocs=world_size,
             join=True)

    '''
    # test
    if torch.cuda.is_available():
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        test_load_path = 'dynamics_model_train_log_250331/in42_out25_dynamics_no_vloop_TF/2025-04-02_11-22-36-lstm_model_5/'
        test_save_folder = 'dynamics_model_train_log_250331/in42_out25_dynamics_no_vloop_TF/2025-04-02_11-22-36-lstm_model_5/test'
        os.makedirs(test_save_folder, exist_ok=True)
        model_test(args, test_data_index, test_number, test_t, test_load_path, test_save_folder)
    '''




