import torch
import numpy as np
import copy
import torch.nn as nn
import torch.nn.functional as F
import wandb
import torch.nn.init as init
import os
import sys
from utils.normalization import Normalization, RewardScaling

curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add to system path

min_max_path = r"./data/250331/min_max_scale_1241_11908/"
max_ic = np.load((min_max_path)+"max_ic_round.npy", allow_pickle=True)
min_ic = np.load((min_max_path)+"min_ic_round.npy", allow_pickle=True)
max_u = np.load((min_max_path)+"max_u_round.npy", allow_pickle=True)
min_u = np.load((min_max_path)+"min_u_round.npy", allow_pickle=True)
max_TF = np.atleast_1d(np.load((min_max_path)+"max_TF_round.npy", allow_pickle=True))
min_TF = np.atleast_1d(np.load((min_max_path)+"min_TF_round.npy", allow_pickle=True))
max_vloop = np.atleast_1d(np.load((min_max_path)+"max_vloop_round.npy", allow_pickle=True))
min_vloop = np.atleast_1d(np.load((min_max_path)+"min_vloop_round.npy", allow_pickle=True))

max_ip = np.atleast_1d(np.load((min_max_path)+"max_ip_round.npy", allow_pickle=True))
min_ip = np.atleast_1d(np.load((min_max_path)+"min_ip_round.npy", allow_pickle=True))
max_r = np.atleast_1d(np.load((min_max_path)+"max_r_round.npy", allow_pickle=True))
min_r = np.atleast_1d(np.load((min_max_path)+"min_r_round.npy", allow_pickle=True))
max_z = np.atleast_1d(np.load((min_max_path)+"max_z_round.npy", allow_pickle=True))
min_z = np.atleast_1d(np.load((min_max_path)+"min_z_round.npy", allow_pickle=True))
max_a = np.atleast_1d(np.load((min_max_path)+"max_a_round.npy", allow_pickle=True))
min_a = np.atleast_1d(np.load((min_max_path)+"min_a_round.npy", allow_pickle=True))
max_e = np.atleast_1d(np.load((min_max_path)+"max_e_round.npy", allow_pickle=True))
min_e = np.atleast_1d(np.load((min_max_path)+"min_e_round.npy", allow_pickle=True))
max_Tr_up = np.atleast_1d(np.load((min_max_path)+"max_Tr_up_round.npy", allow_pickle=True))
min_Tr_up = np.atleast_1d(np.load((min_max_path)+"min_Tr_up_round.npy", allow_pickle=True))
max_Tr_down = np.atleast_1d(np.load((min_max_path)+"max_Tr_down_round.npy", allow_pickle=True))
min_Tr_down = np.atleast_1d(np.load((min_max_path)+"min_Tr_down_round.npy", allow_pickle=True))

data_path = r'./data/250331/data_1241_11908_250331_half_ramp_no_down/'
number = np.load(data_path + "number.npy", allow_pickle=True)
ip = np.load(data_path + "ip.npy", allow_pickle=True)
r = np.load(data_path + "r.npy", allow_pickle=True)
z = np.load(data_path + "z.npy", allow_pickle=True)
a = np.load(data_path + "a.npy", allow_pickle=True)
e = np.load(data_path + "e.npy", allow_pickle=True)
Tr_down = np.load(data_path + "Tr_down.npy", allow_pickle=True)
Tr_up = np.load(data_path + "Tr_up.npy", allow_pickle=True)
ic = np.load(data_path + "ic.npy", allow_pickle=True)
u = np.load(data_path + "u.npy", allow_pickle=True)
vloop = np.load(data_path + "vloop.npy", allow_pickle=True)
TF = np.load(data_path + "TF.npy", allow_pickle=True)
t = np.load(data_path + "t.npy", allow_pickle=True)

class lstm_model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, init, use_LN, use_attention, device):
        super(lstm_model, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.init = init
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers, batch_first=True)
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


def shot_base_data_extraction(shot_number, start_time):
    index = np.where(number == shot_number)[0][0]

    ip_true = ip[index].reshape(-1, 1)
    r_true = r[index].reshape(-1, 1)
    z_true = z[index].reshape(-1, 1)
    a_true = a[index].reshape(-1, 1)
    e_true = e[index].reshape(-1, 1)
    Tr_down_true = Tr_down[index].reshape(-1, 1)
    Tr_up_true = Tr_up[index].reshape(-1, 1)
    vloop_true = vloop[index].reshape(-1, 1)
    TF_true = TF[index].reshape(-1, 1)
    t_true = t[index].reshape(-1, 1)
    ic_true = ic[index].reshape(-1, 18)
    u_true = u[index].reshape(-1, 17)

    start_time_index = np.where(t_true == start_time)[0][0]

    # norm
    u_norm_data = (u_true - min_u) / (max_u - min_u)
    ip_norm = (ip_true - min_ip) / (max_ip - min_ip)
    ic_norm_data = (ic_true - min_ic) / (max_ic - min_ic)
    r_norm = (r_true - min_r) / (max_r - min_r)
    z_norm = (z_true - min_z) / (max_z - min_z)
    a_norm = (a_true - min_a) / (max_a - min_a)
    e_norm = (e_true - min_e) / (max_e - min_e)

    Tr_down_norm = (Tr_down_true - min_Tr_down) / (max_Tr_down - min_Tr_down)
    Tr_up_norm = (Tr_up_true- min_Tr_up) / (max_Tr_up - min_Tr_up)
    vloop_norm = (vloop_true - min_vloop) / (max_vloop - min_vloop)
    TF_norm = (TF_true - min_TF) / (max_TF - min_TF)

    target_s_norm = np.concatenate([a_norm, e_norm, Tr_down_norm, Tr_up_norm, ip_norm, r_norm, z_norm],axis=1)

    lv_TF_norm_data = np.concatenate([vloop_norm, TF_norm], axis=1)

    target_tensor_norm_7 = torch.from_numpy(target_s_norm).type(torch.Tensor)
    u_tensor_norm_17 = torch.from_numpy(u_norm_data).type(torch.Tensor)
    ic_tensor_norm_18 = torch.from_numpy(ic_norm_data).type(torch.Tensor)
    lv_TF_tensor_norm_2 = torch.from_numpy(lv_TF_norm_data).type(torch.Tensor)

    return target_tensor_norm_7, u_tensor_norm_17, lv_TF_tensor_norm_2, ic_tensor_norm_18, start_time_index


class env_lstm():
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.seq_len = args.seq_len
        self.use_ref = args.use_ref
        self.ic_seq = torch.zeros(1, self.seq_len, 18, device=self.device)
        self.s_ = torch.zeros(1, self.seq_len, 25, device=self.device)
        self.max_step = args.max_episode_steps
        self.flags = torch.zeros(18, device=self.device)
        self.ic_zero_upper_norm = args.ic_zero_upper_norm
        self.ic_zero_lower_norm = args.ic_zero_lower_norm
        self.reward_scaling = RewardScaling(args)
        self.use_reward_scaling = args.use_reward_scaling
        self.reward_w = args.reward_w
        self.use_noise = args.use_noise
        self.noise_std = args.noise_std
        self.delta_e = args.delta_e_norm
        self.use_init_noise = args.use_init_noise

        self.lstm_models = []
        for _ in range(args.dynamics_ensemble):
            model = lstm_model(input_dim=args.in_dim, output_dim=args.out_dim,
                               hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                               init=args.init,
                               use_LN=args.use_LN, use_attention=args.use_Attention,
                               device=args.device).to(args.device)
            self.lstm_models.append(model)

        self._load_lstm_models()

        self.lstm_model1 = self.lstm_models[0]
        self.lstm_model2 = self.lstm_models[1]
        self.lstm_model3 = self.lstm_models[2]
        self.lstm_model4 = self.lstm_models[3]
        self.lstm_model5 = self.lstm_models[4]

        for param in self.lstm_model1.parameters():
            param.requires_grad = False
        for param in self.lstm_model2.parameters():
            param.requires_grad = False
        for param in self.lstm_model3.parameters():
            param.requires_grad = False
        for param in self.lstm_model4.parameters():
            param.requires_grad = False
        for param in self.lstm_model5.parameters():
            param.requires_grad = False

        self.lstm_model1.eval()
        self.lstm_model2.eval()
        self.lstm_model3.eval()
        self.lstm_model4.eval()
        self.lstm_model5.eval()

        self.alpha = args.smoothmax_alpha
        self.reward_mode = args.reward_mode  # 0：no penalty  1: delta_u constraint  2：delta_u constraint + ic_constraint
        self.u_threshold = args.u_threshold
        self.diff_u = args.diff_u

        self.target_tensor_norm_7, self.u_tensor_norm_17, \
        self.lv_TF_tensor_norm_2, self.ic_tensor_norm_18, self.start_time_index = shot_base_data_extraction(shot_number=args.base_shot,
                                                                                             start_time=args.start_time)

        self.target_tensor_norm_7_concat = torch.cat([self.target_tensor_norm_7[:self.start_time_index + 500 + 1, :],
                                                 torch.flip(
                                                     self.target_tensor_norm_7[self.start_time_index:self.start_time_index + 500, :],
                                                     [0])], dim=0)

    def _load_lstm_models(self):
        if self.args.eval_type == 'single':
            for i in range(1, self.args.dynamics_ensemble + 1):
                model_file = os.path.join(self.args.dynamics_model_path,
                                          f'model_{i}_params_seq{self.args.seq_len}_seqinterval{self.args.seq_interval}_mean{self.args.mean_len}.pth')

                checkpoint = torch.load(model_file, map_location=self.device)
                saved_params = checkpoint

                self.lstm_models[i - 1].load_state_dict(saved_params['model_state_dict'])

        elif self.args.eval_type == 'mean':
            model_file = os.path.join(self.args.dynamics_model_path,
                                      f'dynamics_model_params_seq{self.args.seq_len}_seqinterval{self.args.seq_interval}_mean{self.args.mean_len}.pth')
            checkpoint = torch.load(model_file, map_location=self.device)

            for i in range(1, self.args.dynamics_ensemble + 1):
                saved_params = checkpoint[i]
                self.lstm_models[i - 1].load_state_dict(saved_params['model_state_dict'])

    def reset(self):
        s_norm = self.target_tensor_norm_7.to(self.device)
        u_norm = self.u_tensor_norm_17.to(self.device)
        with torch.no_grad():
            s_init = s_norm[self.start_time_index + 1 - self.seq_len: self.start_time_index + 1, :]
            if self.use_init_noise:
                s_init[:, 0] = s_init[:, 0] + torch.from_numpy((5 / (max_a - min_a))).type(torch.Tensor).to(self.device)  # a

                s_init[:, 1] = s_init[:, 1] + torch.from_numpy((0.2 / (max_e - min_e))).type( torch.Tensor).to(self.device) # e

                s_init[:, 2] = s_init[:,  2] - torch.from_numpy((0.2 / (max_Tr_down - min_Tr_down))).type(torch.Tensor).to(self.device)  # delta_l

                s_init[:, 3] = s_init[:, 3] - torch.from_numpy((0.2 / (max_Tr_up - min_Tr_up))).type(torch.Tensor).to(self.device)  # delta_u

                s_init[:, 4] = s_init[:, 4] - torch.from_numpy((5 / (max_ip - min_ip))).type(torch.Tensor).to(self.device) # ip

                s_init[:, 5] = s_init[:, 5] + torch.from_numpy((5 / (max_r - min_r))).type(torch.Tensor).to(self.device)  # R
                s_init[:, 6] = s_init[:, 6] + torch.from_numpy((2 / (max_z - min_z))).type(torch.Tensor).to(self.device)  # Z
            u = u_norm[self.start_time_index + 1 - self.seq_len:self.start_time_index + 1, :]
            s_target = s_norm[self.start_time_index + 1, -7:].unsqueeze(0)
            s0 = s_init.unsqueeze(0)
            a = u.unsqueeze(0)
            state_14 = torch.cat([s0[:, -1, -7:], s_target], dim=-1)  # 1*14

        return s0, a, state_14

    def model_test_step(self, s, step):
        u_norm = self.u_tensor_norm_17.to(self.device)
        s_lv_TF = self.lv_TF_tensor_norm_2[self.start_time_index + 1 + step - self.seq_len: self.start_time_index + 1 + step].unsqueeze(0).to(self.device)
        a = u_norm[self.start_time_index + 2 + step - self.seq_len:self.start_time_index + step + 2, :].unsqueeze(0).to(self.device)

        target = self.target_tensor_norm_7[self.start_time_index + 1 + step, -7:].unsqueeze(0).to(self.device)
        target_ = self.target_tensor_norm_7[self.start_time_index + 2 + step, -7:].unsqueeze(0).to(self.device)

        if step == 0:
            self.ic_seq = self.ic_tensor_norm_18[self.start_time_index + 1 - self.seq_len: self.start_time_index + 1].unsqueeze(0).to(self.device)
        else:
            self.ic_seq = torch.cat([self.ic_seq[:, 1:, :], self.s_[:, :-7].unsqueeze(0)], dim=1)

        input = torch.cat([a, s_lv_TF, self.ic_seq, s], dim=2)
        iprz_pred1 = self.lstm_model1(input, flag=1)
        iprz_pred2 = self.lstm_model2(input, flag=1)
        iprz_pred3 = self.lstm_model3(input, flag=1)
        iprz_pred4 = self.lstm_model4(input, flag=1)
        iprz_pred5 = self.lstm_model5(input, flag=1)  # flag=1,inference
        iprz_pred = (iprz_pred1 + iprz_pred2 + iprz_pred3 + iprz_pred4 + iprz_pred5) / 5
        self.s_ = iprz_pred.detach()
        s_ = self.s_[:, -7:]
        ic_pred = self.s_[:, :-7]
        if self.use_noise:
            s_noise = torch.normal(mean=0, std=self.noise_std, size=(1, 7)).to(self.device)
            s_ += s_noise
        s_14 = torch.cat([s_, target_], dim=-1)

        if self.reward_mode == 0:  # no penalty
            reward = self.reward_fun_abs(s_, target, self.alpha, step)
            # reward = self.reward_norm2(s_[:, -7:], target, self.alpha, step)
            if step < (self.max_step - 1):
                terminal = False
            else:
                terminal = True

        elif self.reward_mode == 1:
            reward_base = self.reward_fun_abs(s_[:, -7:], target, self.alpha, step)
            penalty = 0
            for i in range(18):  # Check for multiple zero-crossings in the coil current
                if (self.ic_seq[:, -1, i] > self.ic_zero_lower_norm[i] and self.s_[:, i] < self.ic_zero_lower_norm[i]) \
                        or (
                        self.ic_seq[:, -1, i] < self.ic_zero_lower_norm[i] and self.s_[:, i] > self.ic_zero_lower_norm[
                    i]) \
                        or (
                        self.ic_seq[:, -1, i] < self.ic_zero_upper_norm[i] and self.s_[:, i] > self.ic_zero_upper_norm[
                    i]) \
                        or (
                        self.ic_seq[:, -1, i] > self.ic_zero_upper_norm[i] and self.s_[:, i] < self.ic_zero_upper_norm[
                    i]):
                    if self.flags[i] == 0:
                        self.flags[i] = 1
                    elif self.flags[i] == 1:
                        penalty += (-0.05)  # Zero-crossing penalty

            reward = reward_base + penalty
            if step < (self.max_step - 1):
                terminal = False
            else:
                terminal = True

        return s_14, ic_pred, reward, terminal

    def step(self, s, a, step):
        a_now = a[:, -1, :]
        a_last = a[:, -2, :]
        # delta_a = torch.abs(torch.abs(a_now) - torch.abs(a_last))
        delta_a = torch.abs(a_now - a_last)
        '''
        wandb.log({'delta_CS': delta_a[:, 0] * self.diff_u[0], 'delta_PF1U': delta_a[:, 1] * self.diff_u[1],
                   'delta_PF2U': delta_a[:, 3] * self.diff_u[3], 'delta_PF3U': delta_a[:, 5] * self.diff_u[5],
                   'delta_PF4L': delta_a[:, 8] * self.diff_u[8], 'delta_PF5U': delta_a[:, 9] * self.diff_u[9],
                   'delta_PF6U': delta_a[:, 11] * self.diff_u[11], 'delta_PF7U': delta_a[:, 13] * self.diff_u[13],
                   'delta_PF8U': delta_a[:, 15] * self.diff_u[15]})
        '''

        s_lv_TF = self.lv_TF_tensor_norm_2[
                  self.start_time_index + 1 + step - self.seq_len: self.start_time_index + 1 + step].unsqueeze(0).to(
            self.device)
        if self.use_ref:
            target = self.target_tensor_norm_7[self.start_time_index, -7:].unsqueeze(0).to(self.device)
            target_ = self.target_tensor_norm_7[self.start_time_index, -7:].unsqueeze(0).to(self.device)
            target[1, :] = target[1:] + self.delta_e
            target_[1, :] = target_[1:] + self.delta_e
        else:
            target = self.target_tensor_norm_7[self.start_time_index + 1 + step, -7:].unsqueeze(0).to(self.device)
            target_ = self.target_tensor_norm_7[self.start_time_index + 2 + step, -7:].unsqueeze(0).to(self.device)

        if step == 0:
            self.ic_seq = self.ic_tensor_norm_18[self.start_time_index + 1 - self.seq_len: self.start_time_index + 1].unsqueeze(0).to(
                self.device)
        else:
            self.ic_seq = torch.cat([self.ic_seq[:, 1:, :], self.s_[:, :-7].unsqueeze(0)], dim=1)

        input = torch.cat([a, s_lv_TF, self.ic_seq, s], dim=2)

        iprz_pred1 = self.lstm_model1(input, flag=1)
        iprz_pred2 = self.lstm_model2(input, flag=1)
        iprz_pred3 = self.lstm_model3(input, flag=1)
        iprz_pred4 = self.lstm_model4(input, flag=1)
        iprz_pred5 = self.lstm_model5(input, flag=1)
        iprz_pred = (iprz_pred1 + iprz_pred2 + iprz_pred3 + iprz_pred4 + iprz_pred5) / 5
        self.s_ = iprz_pred.detach()
        s_ = self.s_[:, -7:]
        ic_pred = self.s_[:, :-7]
        if self.use_noise:
            s_noise = torch.normal(mean=0, std=self.noise_std, size=(1, 7)).to(self.device)
            s_ += s_noise
        s_14 = torch.cat([s_, target_], dim=-1)

        if self.reward_mode == 0:  # no penalty
            reward = self.reward_fun_abs(s_, target, self.alpha, step)
            # reward = self.reward_norm2(s_[:, -7:], target, self.alpha, step)
            if step < (self.max_step - 1):
                terminal = False
            else:
                terminal = True

        elif self.reward_mode == 1:
            reward_base = self.reward_fun_abs(s_[:, -7:], target, self.alpha, step)
            penalty = 0
            for i in range(18):
                if (self.ic_seq[:, -1, i] > self.ic_zero_lower_norm[i] and self.s_[:, i] < self.ic_zero_lower_norm[i]) \
                        or (
                        self.ic_seq[:, -1, i] < self.ic_zero_lower_norm[i] and self.s_[:, i] > self.ic_zero_lower_norm[
                    i]) \
                        or (
                        self.ic_seq[:, -1, i] < self.ic_zero_upper_norm[i] and self.s_[:, i] > self.ic_zero_upper_norm[
                    i]) \
                        or (
                        self.ic_seq[:, -1, i] > self.ic_zero_upper_norm[i] and self.s_[:, i] < self.ic_zero_upper_norm[
                    i]):
                    if self.flags[i] == 0:
                        self.flags[i] = 1
                    elif self.flags[i] == 1:
                        penalty += (-0.05)

            reward = reward_base + penalty
            if step < (self.max_step - 1):
                terminal = False
            else:
                terminal = True

        return s_14, ic_pred, reward, terminal

    def reward_fun_abs(self, state, target, alpha, step):
        s_pred = state
        s_reward_0 = -torch.abs(s_pred - target)
        s_reward_w = torch.mul(s_reward_0, self.reward_w)

        # reward scaling
        if self.use_reward_scaling:
            if step == 0:
                self.reward_scaling.reset()
            s_reward_np = s_reward_w.squeeze(0).cpu().numpy()
            s_reward_norm = self.reward_scaling(s_reward_np)
            s_reward = torch.from_numpy(s_reward_norm).type(torch.Tensor).unsqueeze(0).to(self.device)
        else:
            s_reward = s_reward_w

        # component reward
        a_reward = s_reward[:, 0]
        e_reward = s_reward[:, 1]
        Tr_down_reward = s_reward[:, 2]
        Tr_up_reward = s_reward[:, 3]
        ip_reward = s_reward[:, 4]
        r_reward = s_reward[:, 5]
        z_reward = s_reward[:, 6]

        wandb.log({'a_reward': a_reward, 'e_reward': e_reward, 'Tr_down_reward': Tr_down_reward, 'Tr_up_reward': Tr_up_reward,
                   'ip_reward': ip_reward, 'r_reward': r_reward, 'z_reward': z_reward})

        # reward = torch.sum(iprz_reward)
        reward = self.smoothmax(s_reward, alpha)
        return reward

    def reward_norm2(self, state, target, alpha, step):
        s_pred = state
        s_reward_0 = -((s_pred - target) ** 2)
        s_reward_w = torch.mul(s_reward_0, self.reward_w)

        # reward scaling
        if self.use_reward_scaling:
            if step == 0:
                self.reward_scaling.reset()
            s_reward_np = s_reward_w.squeeze(0).cpu().numpy()
            s_reward_norm = self.reward_scaling(s_reward_np)
            s_reward = torch.from_numpy(s_reward_norm).type(torch.Tensor).unsqueeze(0).to(self.device)
        else:
            s_reward = s_reward_w

        # component reward
        a_reward = s_reward[:, 0]
        e_reward = s_reward[:, 1]
        Tr_down_reward = s_reward[:, 2]
        Tr_up_reward = s_reward[:, 3]
        ip_reward = s_reward[:, 4]
        r_reward = s_reward[:, 5]
        z_reward = s_reward[:, 6]

        wandb.log(
            {'a_reward': a_reward, 'e_reward': e_reward, 'Tr_down_reward': Tr_down_reward, 'Tr_up_reward': Tr_up_reward,
             'ip_reward': ip_reward, 'r_reward': r_reward, 'z_reward': z_reward})

        reward = self.smoothmax(s_reward, alpha)
        return reward

    def smoothmax(self, tensor, alpha):
        exp_tensor = torch.exp(alpha * tensor)
        exp_sum = torch.sum(exp_tensor)
        numerator = torch.sum(tensor * exp_tensor)
        smoothmax_val = numerator / exp_sum
        return smoothmax_val



