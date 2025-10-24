# -*-coding:Utf-8-*
# @File: dynamics_ensemble_ddp_update.py
# author: wnn
# Time: 2025/6/9
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
import json
from datetime import datetime
from torch.optim.lr_scheduler import ExponentialLR
from plot.plot_test_dynamics import plot_test_update
from data_process.min_max_scale_load import min_max_scale_load
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import wandb

curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
min_max_path = os.path.abspath(os.path.join(curr_path, '..', 'data', '250331', 'min_max_scale_1241_11908'))
max_ic, min_ic, max_u, min_u, max_TF, min_TF, max_vloop, min_vloop, max_ip, min_ip, max_r, min_r, \
max_z, min_z, max_a, min_a, max_e, min_e, max_Tr_down, min_Tr_down, max_Tr_up, min_Tr_up = min_max_scale_load(
    min_max_path)


class lstm_model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout, init, use_LN, use_attention, device):
        super(lstm_model, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.init = init
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers, batch_first=True, dropout=self.dropout)
        self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.fc_3 = nn.Linear(int(hidden_dim / 2), self.output_dim)
        self.use_LN = use_LN
        self.use_att = use_attention

        if self.use_LN:
            self.lstm_layer_norm = nn.LayerNorm(self.hidden_dim)
            self.hidden_layer_norm = nn.LayerNorm(self.hidden_dim)
            self.fc1_layer_norm = nn.LayerNorm(self.hidden_dim)
            self.fc2_layer_norm = nn.LayerNorm(int(hidden_dim / 2))

        self.active = nn.ReLU()
        if self.init:
            self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)

    def forward(self, x, teacher_forcing_ratio=1, flag=1):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        out, hidden = self.lstm(x[:, :-1, :], (h0.detach(), c0.detach()))
        if self.use_LN:
            out = self.lstm_layer_norm(out)
            hidden = (self.hidden_layer_norm(hidden[0]), hidden[1])

        # Attention
        if self.use_att:
            e = torch.bmm(out, out.permute(0, 2, 1))
            attention = F.softmax(e, dim=-1)
            out_attention = torch.bmm(attention, out)

            out_1 = self.fc_1(out_attention[:, -1, :])
        else:
            out_1 = self.fc_1(out[:, -1, :])

        if self.use_LN:
            out_1 = self.fc1_layer_norm(out_1)
        out_2 = self.active(out_1)

        out_3 = self.fc_2(out_2)
        if self.use_LN:
            out_3 = self.fc2_layer_norm(out_3)
        out_4 = self.active(out_3)

        pred = self.fc_3(out_4)
        batch = x.shape[0]

        if flag == 0:  # train
            next_iprz = torch.where(
                (torch.rand(batch) < teacher_forcing_ratio).reshape(-1, 1).to(self.device),
                x[:, -1, -self.output_dim:], pred
            )

        elif flag == 1: # inference
            next_iprz = x[:, -1, -self.output_dim:]

        next_input = x[:, -1, :].clone()
        next_input[:, -self.output_dim:] = next_iprz
        next_input = torch.cat([x[:, :-1, :], next_input.unsqueeze(1)], dim=1)  # batch*seq*44

        out_1, hidden_1 = self.lstm(next_input, hidden)
        if self.use_LN:
            out_1 = self.lstm_layer_norm(out_1)

        if self.use_att:
            e_1 = torch.bmm(out_1, out_1.permute(0, 2, 1))
            attention_1 = F.softmax(e_1, dim=-1)
            out_1_attention = torch.bmm(attention_1,
                                        out_1)
            out_1_1 = self.fc_1(out_1_attention[:, -1, :])
        else:
            out_1_1 = self.fc_1(out_1[:, -1, :])

        if self.use_LN:
            out_1_1 = self.fc1_layer_norm(out_1_1)
        out_2_1 = self.active(out_1_1)

        out_3_1 = self.fc_2(out_2_1)
        if self.use_LN:
            out_3_1 = self.fc2_layer_norm(out_3_1)
        out_4_1 = self.active(out_3_1)
        pred = self.fc_3(out_4_1)
        return pred


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num, norm):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)
        self.norm = norm

    def forward(self, x):
        if self.norm:
            params_normalized = torch.nn.functional.softmax(self.params, dim=0)
        else:
            params_normalized = self.params
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (params_normalized ** 2) * loss + torch.log(1 + params_normalized ** 2)
        return torch.mean(loss_sum), params_normalized


class ModelOptimizer:
    def __init__(self, model, optimizer, scheduler, loss_fn_w):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn_w = loss_fn_w


class EnsembleDynamics:
    def __init__(self, args, rank=0):
        self.args = args
        self.is_ddp = dist.is_initialized()
        self.rank = rank if self.is_ddp else 0
        self.n_models = args.dynamics_ensemble
        self.models_optimizers = []
        self.loss_fn = torch.nn.MSELoss(reduction='none')

        if not self.is_ddp:
            self.device = args.device
        else:
            self.device = torch.device(f"cuda:{self.rank}")

        # for RL
        self.flags = torch.zeros(18, device=self.device)

        # current time
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # makedirs
        self.save_dir = args.save_path
        self.save_model_folder = os.path.join(self.save_dir, current_time + '-lstm_model_5')

        for _ in range(args.dynamics_ensemble):
            LSTM_model = lstm_model(input_dim=args.in_dim, output_dim=args.out_dim, hidden_dim=args.hidden_dim,
                                    num_layers=args.num_layers, dropout=args.dropout, init=args.init,
                                    use_LN=args.use_LN,
                                    use_attention=args.use_Attention, device=args.device).to(args.device)
            if self.is_ddp:
                LSTM_model = DDP(LSTM_model, device_ids=[rank])
            loss_fn_w = AutomaticWeightedLoss(num=args.out_dim, norm=args.norm).to(args.device)

            if args.use_AW:
                LSTM_optimizer = torch.optim.Adam(
                    [{'params': LSTM_model.parameters(), 'lr': args.lr_dynamics, 'weight_decay': 1e-5},
                     {'params': loss_fn_w.parameters(), 'lr': args.lr_weight, 'weight_decay': 1e-5}])
            else:
                LSTM_optimizer = torch.optim.Adam(LSTM_model.parameters(), lr=args.lr_dynamics, weight_decay=1e-5)

            LSTM_scheduler = ExponentialLR(LSTM_optimizer, gamma=args.gamma)

            self.models_optimizers.append(ModelOptimizer(LSTM_model, LSTM_optimizer, LSTM_scheduler, loss_fn_w))

    def linear_decay_teacher_forcing_ratio(self, initial_ratio, final_ratio, current_epoch, total_epochs):
        ratio = initial_ratio - (current_epoch / total_epochs) * (initial_ratio - final_ratio)
        return ratio

    def exponential_decay_teacher_forcing_ratio(self, initial_ratio, decay_factor, current_epoch):
        ratio = initial_ratio * (decay_factor ** current_epoch)
        return ratio

    def _get_underlying_model(self, ddp_model):
        if self.is_ddp:
            return ddp_model.module
        else:
            return ddp_model

    def train(self, train_loader, train_sampler, val_data, mask, origin_length):
        if self.rank == 0:
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(self.save_model_folder, exist_ok=True)

            # init logging
            log_file = os.path.join(self.save_model_folder, "training_log.log")

            # setting
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s - %(levelname)s - %(message)s',
                                filename=log_file,
                                filemode='w')
            logging.info("Training Parameters:")
            params = {}
            for arg, value in vars(self.args).items():
                logging.info(f"{arg}: {value}")

                if isinstance(value, np.ndarray):
                    params[arg] = value.tolist()
                elif isinstance(value, torch.device):
                    params[arg] = str(value)
                else:
                    params[arg] = value


            param_file = os.path.join(self.save_model_folder, "parameters.json")
            with open(param_file, 'w') as f:
                json.dump(params, f, indent=4)

            model1 = self.models_optimizers[0].model
            print(model1)

            model1_info = str(model1)
            logging.info('Resp_Model Information: %s', model1_info)


        if self.args.load_path != None:
            if self.args.eval_type == 'mean':
                checkpoint = torch.load(self.args.load_path, map_location=self.device)
                for i in range(1, self.args.dynamics_ensemble + 1):
                    saved_params = checkpoint[i]
                    self.args.initial_ratio = saved_params['initial_ratio']
                    self.args.final_ratio = saved_params['final_ratio']
                    self.args.train_epochs = saved_params['num_epochs']
                    break_epoch = saved_params['epoch']
                    self.args.test_min_loss = saved_params['test_min_loss']
                    self.models_optimizers[i - 1].model.load_state_dict(saved_params['model_state_dict'])
                    self.models_optimizers[i - 1].optimizer.load_state_dict(saved_params['optimizer_state_dict'])
                    self.models_optimizers[i - 1].scheduler.load_state_dict(saved_params['scheduler_state_dict'])

                    if self.args.use_AW and self.models_optimizers[i - 1].loss_fn_w is not None:
                        self.models_optimizers[i - 1].loss_fn_w.load_state_dict(saved_params['loss_fn_state_dict'])

            elif self.args.eval_type == 'single':
                model_folder = self.args.load_path
                break_epoch_ensemble = []
                for i in range(1, self.args.dynamics_ensemble + 1):
                    model_file = os.path.join(model_folder,
                                              f'model_{i}_params_seq{self.args.seq_len}_seqinterval{self.args.seq_interval}'
                                              f'_mean{self.args.mean_len}.pth')

                    checkpoint = torch.load(model_file, map_location=self.device)
                    saved_params = checkpoint

                    self.args.initial_ratio = saved_params['initial_ratio']
                    self.args.final_ratio = saved_params['final_ratio']
                    self.args.train_epochs = saved_params['num_epochs']
                    break_epoch_sub = saved_params['epoch']
                    self.args.test_min_loss[i - 1] = saved_params['test_min_loss']
                    break_epoch_ensemble.append(break_epoch_sub)

                    self.models_optimizers[i - 1].model.load_state_dict(saved_params['model_state_dict'])
                    self.models_optimizers[i - 1].optimizer.load_state_dict(saved_params['optimizer_state_dict'])
                    self.models_optimizers[i - 1].scheduler.load_state_dict(saved_params['scheduler_state_dict'])

                    if self.args.use_AW and self.models_optimizers[i - 1].loss_fn_w is not None:
                        self.models_optimizers[i - 1].loss_fn_w.load_state_dict(saved_params['loss_fn_state_dict'])

        for epoch in range(self.args.train_epochs):
            train_sampler.set_epoch(epoch)

            if self.args.load_path != None and self.args.eval_type == 'mean':
                current_epoch = break_epoch + epoch + 1
                teacher_forcing_ratio = self.linear_decay_teacher_forcing_ratio(self.args.initial_ratio,
                                                                                self.args.final_ratio,

                                                                                current_epoch, self.args.train_epochs)
                if epoch == 0:
                    print(f'current_epoch:{current_epoch}\n')
                    logging.info(f'current_epoch:{current_epoch}\n')

            elif self.args.load_path == None:
                teacher_forcing_ratio = self.linear_decay_teacher_forcing_ratio(self.args.initial_ratio,
                                                                                self.args.final_ratio,
                                                                                epoch, self.args.train_epochs)
            loss_LSTMs = [0] * self.args.dynamics_ensemble

            for step, (batch_train_x, batch_target_x) in enumerate(train_loader):
                batch_train_x = batch_train_x.to(self.device)
                batch_target_x = batch_target_x.to(self.device)

                for i in range(self.args.dynamics_ensemble):
                    LSTM_model = self.models_optimizers[i].model
                    LSTM_model.train()

                    if self.args.load_path != None and self.args.eval_type == 'single' and step == 0:
                        current_epoch = break_epoch_ensemble[i] + epoch + 1
                        teacher_forcing_ratio = self.linear_decay_teacher_forcing_ratio(self.args.initial_ratio,
                                                                                        self.args.final_ratio,
                                                                                        current_epoch,
                                                                                        self.args.train_epochs)
                        if epoch == 0 and step == 0:
                            print(f'i:{i + 1}, current_epoch:{current_epoch}\n')
                            logging.info(f'i:{i + 1}, current_epoch:{current_epoch}\n')

                    pred = LSTM_model(batch_train_x, teacher_forcing_ratio, flag=0)
                    mse = self.loss_fn(pred, batch_target_x)
                    if self.args.use_AW:
                        loss, weight = self.models_optimizers[i].loss_fn_w(mse)
                        loss = loss / self.args.batch_size
                    else:
                        loss = torch.mean(mse)

                    optimizer = self.models_optimizers[i].optimizer
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_LSTMs[i] += loss.item()

            if (epoch + 1) % self.args.decay_epoch == 0:
                if self.args.use_AW:
                    if self.rank==0:
                        current_lr_model_LSTM = self.models_optimizers[0].optimizer.state_dict()['param_groups'][0]['lr']
                        current_lr_weight = self.models_optimizers[0].optimizer.state_dict()['param_groups'][1]['lr']
                        log_msg = f"current_lr_LSTM_model:{current_lr_model_LSTM:.6f}, current_lr_weight:{current_lr_weight:.6f}"
                        print(log_msg)
                        logging.info(log_msg)

                    for i in range(self.args.dynamics_ensemble):
                        self.models_optimizers[i].scheduler.step()


                    if self.rank==0:
                        adjudted_lr_LSTM = self.models_optimizers[0].scheduler.get_lr()
                        log_msg = f"adjust_lr_LSTM_model:{adjudted_lr_LSTM[0]:.6f}, adjust_lr_weight:{adjudted_lr_LSTM[1]:.6f}"
                        print(log_msg)
                        logging.info(log_msg)
                        print("---------------------------------------")

                else:
                    if self.rank == 0:
                        current_lr_model_LSTM = self.models_optimizers[0].optimizer.state_dict()['param_groups'][0]['lr']
                        log_msg = f"current_lr_LSTM_model:{current_lr_model_LSTM:.6f}"
                        print(log_msg)
                        logging.info(log_msg)

                    for i in range(self.args.dynamics_ensemble):
                        self.models_optimizers[i].scheduler.step()
                    if self.rank == 0:
                        adjudted_lr_LSTM = self.models_optimizers[0].scheduler.get_lr()
                        log_msg = f"adjust_lr_LSTM_model:{adjudted_lr_LSTM[0]:.6f}"
                        print(log_msg)
                        logging.info(log_msg)
                        print("---------------------------------------")
            if self.args.use_AW:
                if self.rank == 0:
                    print(
                        f'Epoch:{epoch + 1}/{self.args.train_epochs}\n'
                        f'loss_lstm:{loss_LSTMs}\n'
                        f'auto_weight:{weight}')
                    logging.info(
                        f'Epoch:{epoch + 1}/{self.args.train_epochs}\n'
                        f'loss_lstm:{loss_LSTMs}\n'
                        f'auto_weight:{weight}')
            else:
                if self.rank == 0:
                    print(
                        f'Epoch:{epoch + 1}/{self.args.train_epochs}\n'
                        f'loss_lstm:{loss_LSTMs}')
                    logging.info(
                        f'Epoch:{epoch + 1}/{self.args.train_epochs}\n'
                        f'loss_lstm:{loss_LSTMs}\n')

            if epoch >= 1 and self.rank==0:
                self.model_eval(val_data, mask, origin_length, epoch)

    def model_eval(self, val_data, mask, origin_length, epoch):
        val_data_stack = torch.stack([item[0] for item in val_data]).to(self.device)
        val_target_stack = torch.stack([item[1] for item in val_data]).to(self.device)
        loss_test = torch.zeros(val_data[0][0].shape[0], len(val_data), self.args.out_dim).to(self.device)  # len*n*25
        ensemble_losses = torch.zeros(self.args.dynamics_ensemble, val_data[0][0].shape[0], len(val_data),
                                      self.args.out_dim).to(self.device)
        if self.args.eval_type == 'mean':
            with torch.no_grad():
                for i in range(self.args.dynamics_ensemble):
                    self.models_optimizers[i].model.eval()

                for t in range(val_data[0][0].shape[0]):
                    if t == 0:
                        test_x = val_data_stack[:, t, :, :].to(self.device)
                    else:
                        test_ic = val_data_stack[:, t, :, :-self.args.out_dim].to(self.device)
                        test_iprz = torch.cat([test_x[:, 1:, -self.args.out_dim:], pred_test.unsqueeze(1).detach()],
                                              dim=1)
                        test_x = torch.cat([test_ic, test_iprz], dim=-1)

                    test_target = val_target_stack[:, t, :].to(self.device)
                    pred_test = torch.zeros(len(val_data), self.args.out_dim).to(self.device)

                    for index in range(self.args.dynamics_ensemble):
                        model = self.models_optimizers[index].model
                        pred_test_sub = model(test_x, teacher_forcing_ratio=1, flag=1)
                        pred_test += (pred_test_sub / self.args.dynamics_ensemble)

                    mse = self.loss_fn(pred_test.detach(), test_target)
                    loss_test[t, :, :] = mse

                    if t == 0:
                        test_out = pred_test
                    else:
                        test_out = torch.cat([test_out, pred_test], dim=0)

                mask_1 = mask.permute(1, 0)
                mask_2 = (mask_1 == 0).unsqueeze(-1).expand_as(loss_test)
                loss_test[mask_2] = 0

                mask_3 = (mask_2 == 0)
                non_zero_loss_test = torch.where(mask_3, loss_test, torch.tensor(float('nan')).to(loss_test.device))
                loss_mean_overall = torch.nanmean(non_zero_loss_test, dim=[0, 1]).unsqueeze(0)

                val_loss_overall = loss_mean_overall.mean().item()
                print(f"val_loss_overall = {val_loss_overall}")
                logging.info(f"val_loss_overall = {val_loss_overall}")

                if val_loss_overall < self.args.test_min_loss:
                    self.args.test_min_loss = val_loss_overall

                    checkpoint = {}
                    for i in range(self.args.dynamics_ensemble):
                        model_opt = self.models_optimizers[i]
                        model_to_save = self._get_underlying_model(model_opt.model)
                        checkpoint[i + 1] = {
                            'model_state_dict': model_to_save.state_dict(),
                            'optimizer_state_dict': model_opt.optimizer.state_dict(),
                            'scheduler_state_dict': model_opt.scheduler.state_dict(),
                            'epoch': epoch,
                            'num_epochs': self.args.train_epochs,
                            'initial_ratio': self.args.initial_ratio,
                            'final_ratio': self.args.final_ratio,
                            'loss_fn_state_dict': model_opt.loss_fn_w.state_dict() if self.args.use_AW else None,
                            'test_min_loss': val_loss_overall
                        }
                    resp_model_save_path = os.path.join(self.save_model_folder,
                                                        f'dynamics_model_params_seq{self.args.seq_len}_seqinterval'
                                                        f'{self.args.seq_interval}_mean{self.args.mean_len}.pth')
                    torch.save(checkpoint, resp_model_save_path)

                    loss_a = loss_mean_overall[0, -7].item()
                    loss_e = loss_mean_overall[0, -6].item()
                    loss_Tr_down = loss_mean_overall[0, -5].item()
                    loss_Tr_up = loss_mean_overall[0, -4].item()
                    loss_ip = loss_mean_overall[0, -3].item()
                    loss_r = loss_mean_overall[0, -2].item()
                    loss_z = loss_mean_overall[0, -1].item()
                    print(
                        f"===================loss_average = {val_loss_overall},"
                        f"loss_iprz = {loss_a:.6f},{loss_e:.6f},{loss_Tr_down:.6f},{loss_Tr_up:.6f},"
                        f"{loss_ip:.6f},{loss_r:.6f},{loss_z:.6f}, Model Save!============================")
                    logging.info(
                        f"===================loss_average = {val_loss_overall},"
                        f"loss_iprz = {loss_a:.6f},{loss_e:.6f},{loss_Tr_down:.6f},{loss_Tr_up:.6f},"
                        f"{loss_ip:.6f},{loss_r:.6f},{loss_z:.6f}, Model Save!============================")

        if self.args.eval_type == 'single':

            with torch.no_grad():
                for index in range(self.args.dynamics_ensemble):
                    model = self.models_optimizers[index].model
                    model.eval()


                    pred_test = torch.zeros(len(val_data), val_data[0][0].shape[0], self.args.out_dim).to(
                        self.device)

                    for t in range(val_data[0][0].shape[0]):
                        if t == 0:
                            test_x = val_data_stack[:, t, :, :].to(self.device)
                        else:
                            test_u = val_data_stack[:, t, :, :-self.args.out_dim].to(self.device)
                            test_iprz = torch.cat(
                                [test_x[:, 1:, -self.args.out_dim:], pred_test[:, t - 1, :].unsqueeze(1).detach()],
                                dim=1)  # n*30*25
                            test_x = torch.cat([test_u, test_iprz], dim=-1)

                        test_target = val_target_stack[:, t, :].to(self.device)

                        pred_test_sub = model(test_x, teacher_forcing_ratio=1, flag=1)

                        pred_test[:, t, :] = pred_test_sub.detach()

                        mse = self.loss_fn(pred_test_sub.detach(), test_target)
                        ensemble_losses[index, t, :, :] = mse

                    mask_1 = mask.permute(1, 0)
                    mask_2 = (mask_1 == 0).unsqueeze(-1).expand_as(ensemble_losses[index])
                    ensemble_losses[index][mask_2] = 0

                    mask_3 = (mask_2 == 0)
                    non_zero_loss_test = torch.where(mask_3, ensemble_losses[index],
                                                     torch.tensor(float('nan')).to(ensemble_losses.device))
                    loss_mean_overall = torch.nanmean(non_zero_loss_test, dim=[0, 1]).unsqueeze(0)

                    val_loss_overall = loss_mean_overall.mean().item()
                    print(f"Model {index + 1}: val_loss_overall = {val_loss_overall}")
                    logging.info(f"Model {index + 1}: val_loss_overall = {val_loss_overall}")


                    if val_loss_overall < self.args.test_min_loss[index]:
                        self.args.test_min_loss[index] = val_loss_overall
                        model_to_save = self._get_underlying_model(model)
                        checkpoint = {
                            'model_state_dict': model_to_save.state_dict(),
                            'optimizer_state_dict': self.models_optimizers[index].optimizer.state_dict(),
                            'scheduler_state_dict': self.models_optimizers[index].scheduler.state_dict(),
                            'epoch': epoch,
                            'num_epochs': self.args.train_epochs,
                            'initial_ratio': self.args.initial_ratio,
                            'final_ratio': self.args.final_ratio,
                            'loss_fn_state_dict': self.models_optimizers[
                                index].loss_fn_w.state_dict() if self.args.use_AW else None,
                            'test_min_loss': val_loss_overall
                        }

                        resp_model_save_path = os.path.join(self.save_model_folder,
                                                            f'model_{index + 1}_params_seq{self.args.seq_len}_seqinterval'
                                                            f'{self.args.seq_interval}_mean{self.args.mean_len}.pth')
                        torch.save(checkpoint, resp_model_save_path)

                        loss_a = loss_mean_overall[0, -7].item()
                        loss_e = loss_mean_overall[0, -6].item()
                        loss_Tr_down = loss_mean_overall[0, -5].item()
                        loss_Tr_up = loss_mean_overall[0, -4].item()
                        loss_ip = loss_mean_overall[0, -3].item()
                        loss_r = loss_mean_overall[0, -2].item()
                        loss_z = loss_mean_overall[0, -1].item()
                        print(
                            f"Model {index + 1}: loss_iprz = {loss_a:.6f},{loss_e:.6f},{loss_Tr_down:.6f},{loss_Tr_up:.6f},"
                            f"{loss_ip:.6f},{loss_r:.6f},{loss_z:.6f}, Model Saved!")
                        logging.info(
                            f"Model {index + 1}: loss_iprz = {loss_a:.6f},{loss_e:.6f},{loss_Tr_down:.6f},{loss_Tr_up:.6f},"
                            f"{loss_ip:.6f},{loss_r:.6f},{loss_z:.6f}, Model Saved!")

    @torch.no_grad()
    def model_test_ensemble(self, test_data, test_number, test_t, model_path, save_path):
        # init logging
        save_pic_folder = save_path
        log_file = os.path.join(save_pic_folder, "test_log.log")

        # setting
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=log_file)

        model_folder = model_path
        if self.args.eval_type == 'single':
            for i in range(1, self.args.dynamics_ensemble + 1):
                model_file = os.path.join(model_folder,
                                          f'model_{i}_params_seq{self.args.seq_len}_seqinterval'
                                          f'{self.args.seq_interval}_mean{self.args.mean_len}.pth')

                checkpoint = torch.load(model_file, map_location=self.device)
                saved_params = checkpoint

                self.models_optimizers[i - 1].model.load_state_dict(saved_params['model_state_dict'])

        elif self.args.eval_type == 'mean':
            model_file = os.path.join(model_folder,
                                      f'dynamics_model_params_seq{self.args.seq_len}_seqinterval'
                                      f'{self.args.seq_interval}_mean{self.args.mean_len}.pth')
            checkpoint = torch.load(model_file, map_location=self.device)

            for i in range(self.args.dynamics_ensemble):
                saved_params = checkpoint[i + 1]
                self.models_optimizers[i].model.load_state_dict(saved_params['model_state_dict'])
                self.models_optimizers[i].model.eval()

        with torch.no_grad():
            loss_mean_overall_full = torch.zeros(self.args.dynamics_ensemble, 25).to(self.device)
            loss_test_full = 0

            for i in range(len(test_data)):

                item = test_data[i]
                item_list = list(item)
                pred_test = torch.zeros(self.args.dynamics_ensemble + 1, item_list[0].shape[0], self.args.out_dim).to(
                    self.device)
                pred_test_avg = torch.zeros(item_list[0].shape[0], self.args.out_dim).to(self.device)

                ensemble_losses = torch.zeros(self.args.dynamics_ensemble, item_list[0].shape[0], self.args.out_dim).to(
                    self.device)
                avg_losses = torch.zeros(item_list[0].shape[0], self.args.out_dim).to(self.device)

                for t in range(item_list[0].shape[0]):
                    if t == 0:
                        test_x = item_list[0][t, :, :].to(self.device)
                    else:
                        test_u = item_list[0][t, :, :-self.args.out_dim].to(
                            self.device)

                        test_iprz = torch.cat(
                            [test_x[1:, -self.args.out_dim:], pred_test_avg[t - 1, :].detach().unsqueeze(0)],
                            dim=0)

                        test_x = torch.cat([test_u, test_iprz], dim=1)

                    test_target = item_list[1][t, :].unsqueeze(0).to(self.device)

                    for index in range(self.args.dynamics_ensemble):
                        model = self._get_underlying_model(self.models_optimizers[index].model)
                        model.eval()

                        pred_test_sub = model(test_x.unsqueeze(0), teacher_forcing_ratio=1, flag=1)

                        pred_test[index, t, :] = pred_test_sub.squeeze(0)

                        mse = self.loss_fn(pred_test_sub.detach(), test_target)
                        ensemble_losses[index, t, :] = mse

                    pred_test_avg[t, :] = torch.mean(pred_test[:self.args.dynamics_ensemble, t, :], dim=0)

                    avg_losses[t, :] = self.loss_fn(pred_test_avg[t, :].detach(), item_list[1][t, :].to(self.device))

                loss_mean_overall = torch.nanmean(ensemble_losses, dim=1)
                loss_mean_overall_avg = torch.nanmean(avg_losses, dim=0)

                if i == 0:
                    loss_mean_overall_full == loss_mean_overall / len(test_data)
                    loss_mean_overall_avg_full = loss_mean_overall_avg / len(test_data)
                else:
                    loss_mean_overall_full += (loss_mean_overall / len(test_data))
                    loss_mean_overall_avg_full += (loss_mean_overall_avg / len(test_data))

                pred_test[self.args.dynamics_ensemble, :, :] = torch.mean(pred_test[:self.args.dynamics_ensemble, :, :],
                                                                          dim=0)
                pred_test_inv = (pred_test.cpu().numpy()) * self.args.var_out_sub + self.args.min_var_out
                test_label_inv = item_list[1].cpu().numpy() * self.args.var_out_sub + self.args.min_var_out
                loss_mean_overall_arr = np.mean((np.abs(pred_test_inv[-1, :, :] - test_label_inv)), axis=0)[-7:]
                loss_test_full += (loss_mean_overall_arr / len(test_data))
                number_test = test_number[i]
                t_test = test_t[i]
                plot_test_update(pred_test_inv, test_label_inv, loss_mean_overall_arr, number_test, t_test,
                                 save_pic_folder)

            for index in range(self.args.dynamics_ensemble):
                val_loss_overall = loss_mean_overall_full[index, :].mean().item()
                print(f"Model {index + 1}: val_loss_overall = {val_loss_overall}")
                logging.info(f"Model {index + 1}: val_loss_overall = {val_loss_overall}")

                loss_a = loss_mean_overall[index, -7].item()
                loss_e = loss_mean_overall[index, -6].item()
                loss_Tr_down = loss_mean_overall[index, -5].item()
                loss_Tr_up = loss_mean_overall[index, -4].item()
                loss_ip = loss_mean_overall[index, -3].item()
                loss_r = loss_mean_overall[index, -2].item()
                loss_z = loss_mean_overall[index, -1].item()
                print(
                    f"Model {index + 1}: loss_iprz = {loss_a:.6f},{loss_e:.6f},{loss_Tr_down:.6f},{loss_Tr_up:.6f},"
                    f"{loss_ip:.6f},{loss_r:.6f},{loss_z:.6f}")
                logging.info(
                    f"Model {index + 1}: loss_iprz = {loss_a:.6f},{loss_e:.6f},{loss_Tr_down:.6f},{loss_Tr_up:.6f},"
                    f"{loss_ip:.6f},{loss_r:.6f},{loss_z:.6f}")

            loss_a_avg = loss_mean_overall_avg_full[-7].item()
            loss_e_avg = loss_mean_overall_avg_full[-6].item()
            loss_Tr_down_avg = loss_mean_overall_avg_full[-5].item()
            loss_Tr_up_avg = loss_mean_overall_avg_full[-4].item()
            loss_ip_avg = loss_mean_overall_avg_full[-3].item()
            loss_r_avg = loss_mean_overall_avg_full[-2].item()
            loss_z_avg = loss_mean_overall_avg_full[-1].item()
            print(
                f"Model_avg : loss_iprz = {loss_a_avg:.6f},{loss_e_avg:.6f},{loss_Tr_down_avg:.6f},{loss_Tr_up_avg:.6f},"
                f"{loss_ip_avg:.6f},{loss_r_avg:.6f},{loss_z_avg:.6f}")
            print(f"loss_test_full:{loss_test_full}")

            logging.info(
                f"Model_avg : loss_iprz = {loss_a_avg:.6f},{loss_e_avg:.6f},{loss_Tr_down_avg:.6f},{loss_Tr_up_avg:.6f},"
                f"{loss_ip_avg:.6f},{loss_r_avg:.6f},{loss_z_avg:.6f}")
            logging.info(f"loss_test_full:{loss_test_full}")

            print('test_ip_14 finished!')


    def load(self):

        if self.args.eval_type == 'single':
            for i in range(1, self.args.dynamics_ensemble + 1):
                model_file = os.path.join(self.args.dynamics_model_path,
                                          f'model_{i}_params_seq{self.args.seq_len}_seqinterval{self.args.seq_interval}_mean{self.args.mean_len}.pth')
                checkpoint = torch.load(model_file, map_location=self.device)
                saved_params = checkpoint
                self.models_optimizers[i - 1].model.load_state_dict(saved_params['model_state_dict'])

        elif self.args.eval_type == 'mean':
            model_file = os.path.join(self.args.dynamics_model_path,
                                      f'dynamics_model_params_seq{self.args.seq_len}_seqinterval{self.args.seq_interval}_mean{self.args.mean_len}.pth')
            checkpoint = torch.load(model_file, map_location=self.device)

            for i in range(1, self.args.dynamics_ensemble + 1):
                saved_params = checkpoint[i]
                self.models_optimizers[i - 1].model.load_state_dict(saved_params['model_state_dict'])






