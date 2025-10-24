import numpy as np
import torch
import os
from data_process.min_max_scale_load import min_max_scale_load

current_dir = os.path.dirname(os.path.abspath(__file__))

data_folder = os.path.abspath(os.path.join(current_dir, '..', 'data', '250331', 'data_1241_11908_250331_half_ramp_no_down'))

a = np.load(os.path.join(data_folder, 'a.npy'), allow_pickle=True)
e = np.load(os.path.join(data_folder, 'e.npy'), allow_pickle=True)
Tr_down = np.load(os.path.join(data_folder, 'Tr_down.npy'), allow_pickle=True)
Tr_up = np.load(os.path.join(data_folder, 'Tr_up.npy'), allow_pickle=True)
ip = np.load(os.path.join(data_folder, 'ip.npy'), allow_pickle=True)
r = np.load(os.path.join(data_folder, 'r.npy'), allow_pickle=True)
z = np.load(os.path.join(data_folder, 'z.npy'), allow_pickle=True)

number = np.load(os.path.join(data_folder, 'number.npy'), allow_pickle=True)
ic = np.load(os.path.join(data_folder, 'ic.npy'), allow_pickle=True)
u = np.load(os.path.join(data_folder, 'u.npy'), allow_pickle=True)
t_full = np.load(os.path.join(data_folder, 't.npy'), allow_pickle=True)
vloop = np.load(os.path.join(data_folder, 'vloop.npy'), allow_pickle=True)
TF = np.load(os.path.join(data_folder, 'TF.npy'), allow_pickle=True)

min_max_path = os.path.abspath(os.path.join(current_dir , '..', 'data', '250331', 'min_max_scale_1241_11908'))
max_ic,min_ic,max_u,min_u,max_TF,min_TF,max_vloop,min_vloop,max_ip,min_ip,max_r,min_r,\
max_z,min_z,max_a,min_a,max_e,min_e,max_Tr_down,min_Tr_down,max_Tr_up,min_Tr_up = min_max_scale_load(min_max_path)

max_var_out = np.concatenate((max_ic, max_a, max_e, max_Tr_down, max_Tr_up, max_ip, max_r, max_z))
min_var_out = np.concatenate((min_ic, min_a, min_e, min_Tr_down, min_Tr_up, min_ip, min_r, min_z))
var_out_sub = max_var_out - min_var_out

shot_number_600_700 = [6660, 6661]
shot_number_600_500 = [6658, 6662, 6663, 6664, 6226, 9999, 11638]  # step current, 600->500
shot_number_600_400 = [6659]  # step current, 600->400
shot_number_300_250 = [10299, 10300, 10301, 10302, 10303, 10304, 10305, 10306, 10307, 10308, 10309, 10310, 10311,
                       10314,
                       10315, 10316, 10317, 10318, 10321, 10322, 10326, ]
data_repeat = 20
shot_step = shot_number_600_700+shot_number_600_500+shot_number_600_400+shot_number_300_250

def data_process_train(train_data_index, seq, mean_len, seq_interval, smooth_type):
    input_u = []
    input_ic = []
    input_ip = []
    input_r = []
    input_z = []
    input_a = []
    input_e = []
    input_Tr_down = []
    input_Tr_up = []
    input_vloop = []
    input_TF = []

    target_ic = []
    target_ip = []
    target_r = []
    target_z = []
    target_e = []
    target_a = []
    target_Tr_down = []
    target_Tr_up = []

    for i in train_data_index:
        shot = number[i]

        u_norm = (u[i] - min_u) / (max_u - min_u)
        ip_norm = (ip[i] - min_ip) / (max_ip - min_ip)
        ic_norm = (ic[i] - min_ic) / (max_ic - min_ic)
        r_norm = (r[i] - min_r) / (max_r - min_r)
        z_norm = (z[i] - min_z) / (max_z - min_z)

        a_norm = (a[i] - min_a) / (max_a - min_a)
        e_norm = (e[i] - min_e) / (max_e - min_e)
        Tr_down_norm = (Tr_down[i] - min_Tr_down) / (max_Tr_down - min_Tr_down)
        Tr_up_norm = (Tr_up[i] - min_Tr_up) / (max_Tr_up - min_Tr_up)
        vloop_norm = (vloop[i] - min_vloop) / (max_vloop - min_vloop)
        TF_norm = (TF[i] - min_TF) / (max_TF - min_TF)

        if smooth_type == 1:
            if shot in shot_step:
                for k in range(data_repeat):
                    for index in range(len(u_norm) - seq -mean_len + 1):
                        input_ip.append(ip_norm[index:index + seq])
                        input_r.append(r_norm[index:index + seq])
                        input_z.append(z_norm[index:index + seq])
                        input_a.append(a_norm[index:index + seq])
                        input_e.append(e_norm[index:index + seq])
                        input_Tr_down.append(Tr_down_norm[index:index + seq])
                        input_Tr_up.append(Tr_up_norm[index:index + seq])

                        input_ic.append(ic_norm[index: index + seq])
                        input_u.append(u_norm[index + 1: index + seq + 1])
                        input_vloop.append(vloop_norm[index: index + seq])
                        input_TF.append(TF_norm[index: index + seq])

                        target_ic.append(ic_norm[index + seq])
                        target_ip.append(np.mean(ip_norm[index + seq: index + seq + mean_len], axis=0))
                        target_r.append(np.mean(r_norm[index + seq: index + seq + mean_len], axis=0))
                        target_z.append(np.mean(z_norm[index + seq: index + seq + mean_len], axis=0))
                        target_a.append(np.mean(a_norm[index + seq: index + seq + mean_len], axis=0))
                        target_e.append(np.mean(e_norm[index + seq: index + seq + mean_len], axis=0))
                        target_Tr_up.append(np.mean(Tr_up_norm[index + seq: index + seq + mean_len], axis=0))
                        target_Tr_down.append(np.mean(Tr_down_norm[index + seq: index + seq + mean_len], axis=0))

            else:
                for index in range(len(u_norm) - seq - mean_len + 1):
                    input_ip.append(ip_norm[index:index + seq])
                    input_r.append(r_norm[index:index + seq])
                    input_z.append(z_norm[index:index + seq])
                    input_a.append(a_norm[index:index + seq])
                    input_e.append(e_norm[index:index + seq])
                    input_Tr_down.append(Tr_down_norm[index:index + seq])
                    input_Tr_up.append(Tr_up_norm[index:index + seq])

                    input_ic.append(ic_norm[index: index + seq])
                    input_u.append(u_norm[index + 1: index + seq + 1])
                    input_vloop.append(vloop_norm[index: index + seq])
                    input_TF.append(TF_norm[index: index + seq])

                    target_ic.append(ic_norm[index + seq])
                    target_ip.append(np.mean(ip_norm[index + seq: index + seq + mean_len], axis=0))
                    target_r.append(np.mean(r_norm[index + seq: index + seq + mean_len], axis=0))
                    target_z.append(np.mean(z_norm[index + seq: index + seq + mean_len], axis=0))
                    target_a.append(np.mean(a_norm[index + seq: index + seq + mean_len], axis=0))
                    target_e.append(np.mean(e_norm[index + seq: index + seq + mean_len], axis=0))
                    target_Tr_up.append(np.mean(Tr_up_norm[index + seq: index + seq + mean_len], axis=0))
                    target_Tr_down.append(np.mean(Tr_down_norm[index + seq: index + seq + mean_len], axis=0))

        elif smooth_type == 2:
            if shot in shot_step:
                for k in range(data_repeat):
                    for index in range(len(u_norm) - seq - mean_len + 1):
                        input_ip.append(ip_norm[index:index + seq])
                        input_r.append(r_norm[index:index + seq])
                        input_z.append(z_norm[index:index + seq])
                        input_a.append(a_norm[index:index + seq])
                        input_e.append(e_norm[index:index + seq])
                        input_Tr_down.append(Tr_down_norm[index:index + seq])
                        input_Tr_up.append(Tr_up_norm[index:index + seq])
                        input_ic.append(ic_norm[index: index + seq])
                        input_vloop.append(vloop_norm[index: index + seq])
                        input_TF.append(TF_norm[index: index + seq])
                        input_u.append(u_norm[index + mean_len: index + seq + mean_len])
                        target_ic.append(ic_norm[index + seq])
                        target_ip.append(np.mean(ip_norm[index + seq + 1 -mean_len: index + seq + 1], axis=0))
                        target_r.append(np.mean(r_norm[index + seq + 1 -mean_len: index + seq + 1], axis=0))
                        target_z.append(np.mean(z_norm[index + seq + 1 -mean_len: index + seq + 1], axis=0))
                        target_a.append(np.mean(a_norm[index + seq + 1 -mean_len: index + seq + 1], axis=0))
                        target_e.append(np.mean(e_norm[index + seq + 1 -mean_len: index + seq + 1], axis=0))
                        target_Tr_up.append(np.mean(Tr_up_norm[index + seq + 1 - mean_len: index + seq + 1], axis=0))
                        target_Tr_down.append(np.mean(Tr_down_norm[index + seq + 1 - mean_len: index + seq + 1], axis=0))
            else:
                for index in range(len(u_norm) - seq - mean_len + 1):
                    input_ip.append(ip_norm[index:index + seq])
                    input_r.append(r_norm[index:index + seq])
                    input_z.append(z_norm[index:index + seq])
                    input_a.append(a_norm[index:index + seq])
                    input_e.append(e_norm[index:index + seq])
                    input_Tr_down.append(Tr_down_norm[index:index + seq])
                    input_Tr_up.append(Tr_up_norm[index:index + seq])
                    input_ic.append(ic_norm[index: index + seq])
                    input_vloop.append(vloop_norm[index: index + seq])
                    input_TF.append(TF_norm[index: index + seq])
                    input_u.append(u_norm[index + mean_len: index + seq + mean_len])
                    target_ic.append(ic_norm[index + seq])
                    target_ip.append(np.mean(ip_norm[index + seq + 1 - mean_len: index + seq + 1], axis=0))
                    target_r.append(np.mean(r_norm[index + seq + 1 - mean_len: index + seq + 1], axis=0))
                    target_z.append(np.mean(z_norm[index + seq + 1 - mean_len: index + seq + 1], axis=0))
                    target_a.append(np.mean(a_norm[index + seq + 1 - mean_len: index + seq + 1], axis=0))
                    target_e.append(np.mean(e_norm[index + seq + 1 - mean_len: index + seq + 1], axis=0))
                    target_Tr_up.append(np.mean(Tr_up_norm[index + seq + 1 - mean_len: index + seq + 1], axis=0))
                    target_Tr_down.append(np.mean(Tr_down_norm[index + seq + 1 - mean_len: index + seq + 1], axis=0))

    input_ic_1 = np.array(input_ic)
    input_ic_2 = torch.from_numpy(input_ic_1.reshape(-1, seq, 18)).type(torch.Tensor)

    input_u_1 = np.array(input_u)
    input_u_2 = torch.from_numpy(input_u_1.reshape(-1, seq, 17)).type(torch.Tensor)

    input_ip_1 = np.array(input_ip)
    input_ip_2 = torch.from_numpy(input_ip_1.reshape(-1, seq, 1)).type(torch.Tensor)

    input_r_1 = np.array(input_r)
    input_r_2 = torch.from_numpy(input_r_1.reshape(-1, seq, 1)).type(torch.Tensor)

    input_z_1 = np.array(input_z)
    input_z_2 = torch.from_numpy(input_z_1.reshape(-1, seq, 1)).type(torch.Tensor)

    input_a_1 = np.array(input_a)
    input_a_2 = torch.from_numpy(input_a_1.reshape(-1, seq, 1)).type(torch.Tensor)

    input_e_1 = np.array(input_e)
    input_e_2 = torch.from_numpy(input_e_1.reshape(-1, seq, 1)).type(torch.Tensor)

    input_Tr_down_1 = np.array(input_Tr_down)
    input_Tr_down_2 = torch.from_numpy(input_Tr_down_1.reshape(-1, seq, 1)).type(torch.Tensor)

    input_Tr_up_1 = np.array(input_Tr_up)
    input_Tr_up_2 = torch.from_numpy(input_Tr_up_1.reshape(-1, seq, 1)).type(torch.Tensor)

    input_vloop_1 = np.array(input_vloop)
    input_vloop_2 = torch.from_numpy(input_vloop_1.reshape(-1, seq, 1)).type(torch.Tensor)

    input_TF_1 = np.array(input_TF)
    input_TF_2 = torch.from_numpy(input_TF_1.reshape(-1, seq, 1)).type(torch.Tensor)

    input_full = torch.cat(
        [input_u_2, input_vloop_2, input_TF_2, input_ic_2, input_a_2, input_e_2, input_Tr_down_2,
         input_Tr_up_2, input_ip_2, input_r_2, input_z_2], dim=2)

    target_ip_1 = np.array(target_ip)
    target_ip_2 = torch.from_numpy(target_ip_1.reshape(-1, 1)).type(torch.Tensor)

    target_ic_1 = np.array(target_ic)
    target_ic_2 = torch.from_numpy(target_ic_1.reshape(-1, 18)).type(torch.Tensor)

    target_r_1 = np.array(target_r)
    target_r_2 = torch.from_numpy(target_r_1.reshape(-1, 1)).type(torch.Tensor)

    target_z_1 = np.array(target_z)
    target_z_2 = torch.from_numpy(target_z_1.reshape(-1, 1)).type(torch.Tensor)

    target_a_1 = np.array(target_a)
    target_a_2 = torch.from_numpy(target_a_1.reshape(-1, 1)).type(torch.Tensor)

    target_e_1 = np.array(target_e)
    target_e_2 = torch.from_numpy(target_e_1.reshape(-1, 1)).type(torch.Tensor)

    target_Tr_up_1 = np.array(target_Tr_up)
    target_Tr_up_2 = torch.from_numpy(target_Tr_up_1.reshape(-1, 1)).type(torch.Tensor)

    target_Tr_down_1 = np.array(target_Tr_down)
    target_Tr_down_2 = torch.from_numpy(target_Tr_down_1.reshape(-1, 1)).type(torch.Tensor)

    target = torch.cat(
        [target_ic_2, target_a_2, target_e_2, target_Tr_down_2, target_Tr_up_2, target_ip_2, target_r_2,
         target_z_2],
        dim=1)
    train_data = (input_full, target)
    return train_data


def data_process_test(test_data_index, seq, seq_interval, mean_len):
    test_data = []
    for i in test_data_index:
        input_u = []
        input_ic = []
        input_ip = []
        input_r = []
        input_z = []
        input_a = []
        input_e = []
        input_Tr_down = []
        input_Tr_up = []
        input_vloop = []
        input_TF = []

        target_ic = []
        target_ip = []
        target_r = []
        target_z = []
        target_e = []
        target_a = []
        target_Tr_down = []
        target_Tr_up = []

        u_norm = (u[i] - min_u) / (max_u - min_u)
        ip_norm = (ip[i] - min_ip) / (max_ip - min_ip)
        ic_norm = (ic[i] - min_ic) / (max_ic - min_ic)
        r_norm = (r[i] - min_r) / (max_r - min_r)
        z_norm = (z[i] - min_z) / (max_z - min_z)

        a_norm = (a[i] - min_a) / (max_a - min_a)
        e_norm = (e[i] - min_e) / (max_e - min_e)
        Tr_down_norm = (Tr_down[i] - min_Tr_down) / (max_Tr_down - min_Tr_down)
        Tr_up_norm = (Tr_up[i] - min_Tr_up) / (max_Tr_up - min_Tr_up)
        vloop_norm = (vloop[i] - min_vloop) / (max_vloop - min_vloop)
        TF_norm = (TF[i] - min_TF) / (max_TF - min_TF)
        for index in range(len(u_norm) - seq - 1):
            input_ip.append(ip_norm[index:index + seq])
            input_r.append(r_norm[index:index + seq])
            input_z.append(z_norm[index:index + seq])
            input_a.append(a_norm[index:index + seq])
            input_e.append(e_norm[index:index + seq])
            input_Tr_down.append(Tr_down_norm[index:index + seq])
            input_Tr_up.append(Tr_up_norm[index:index + seq])

            input_ic.append(ic_norm[index: index + seq])
            input_u.append(u_norm[index+1: index + seq+1])
            input_vloop.append(vloop_norm[index: index + seq])
            input_TF.append(TF_norm[index: index + seq])

            target_ic.append(ic_norm[index + seq])
            target_ip.append(ip_norm[index + seq])
            target_r.append(r_norm[index + seq])
            target_z.append(z_norm[index + seq])
            target_a.append(a_norm[index + seq])
            target_e.append(e_norm[index + seq])
            target_Tr_up.append(Tr_up_norm[index + seq])
            target_Tr_down.append(Tr_down_norm[index + seq])

        input_ic_1 = np.array(input_ic)
        input_ic_2 = torch.from_numpy(input_ic_1.reshape(-1, seq, 18)).type(torch.Tensor)

        input_u_1 = np.array(input_u)
        input_u_2 = torch.from_numpy(input_u_1.reshape(-1, seq, 17)).type(torch.Tensor)

        input_ip_1 = np.array(input_ip)
        input_ip_2 = torch.from_numpy(input_ip_1.reshape(-1, seq, 1)).type(torch.Tensor)

        input_r_1 = np.array(input_r)
        input_r_2 = torch.from_numpy(input_r_1.reshape(-1, seq, 1)).type(torch.Tensor)

        input_z_1 = np.array(input_z)
        input_z_2 = torch.from_numpy(input_z_1.reshape(-1, seq, 1)).type(torch.Tensor)

        input_a_1 = np.array(input_a)
        input_a_2 = torch.from_numpy(input_a_1.reshape(-1, seq, 1)).type(torch.Tensor)

        input_e_1 = np.array(input_e)
        input_e_2 = torch.from_numpy(input_e_1.reshape(-1, seq, 1)).type(torch.Tensor)

        input_Tr_down_1 = np.array(input_Tr_down)
        input_Tr_down_2 = torch.from_numpy(input_Tr_down_1.reshape(-1, seq, 1)).type(torch.Tensor)

        input_Tr_up_1 = np.array(input_Tr_up)
        input_Tr_up_2 = torch.from_numpy(input_Tr_up_1.reshape(-1, seq, 1)).type(torch.Tensor)

        input_vloop_1 = np.array(input_vloop)
        input_vloop_2 = torch.from_numpy(input_vloop_1.reshape(-1, seq, 1)).type(torch.Tensor)

        input_TF_1 = np.array(input_TF)
        input_TF_2 = torch.from_numpy(input_TF_1.reshape(-1, seq, 1)).type(torch.Tensor)

        input_full = torch.cat(
            [input_u_2, input_vloop_2, input_TF_2, input_ic_2, input_a_2, input_e_2, input_Tr_down_2,
             input_Tr_up_2, input_ip_2, input_r_2, input_z_2], dim=2)

        target_ip_1 = np.array(target_ip)
        target_ip_2 = torch.from_numpy(target_ip_1.reshape(-1, 1)).type(torch.Tensor)

        target_ic_1 = np.array(target_ic)
        target_ic_2 = torch.from_numpy(target_ic_1.reshape(-1, 18)).type(torch.Tensor)

        target_r_1 = np.array(target_r)
        target_r_2 = torch.from_numpy(target_r_1.reshape(-1, 1)).type(torch.Tensor)

        target_z_1 = np.array(target_z)
        target_z_2 = torch.from_numpy(target_z_1.reshape(-1, 1)).type(torch.Tensor)

        target_a_1 = np.array(target_a)
        target_a_2 = torch.from_numpy(target_a_1.reshape(-1, 1)).type(torch.Tensor)

        target_e_1 = np.array(target_e)
        target_e_2 = torch.from_numpy(target_e_1.reshape(-1, 1)).type(torch.Tensor)

        target_Tr_up_1 = np.array(target_Tr_up)
        target_Tr_up_2 = torch.from_numpy(target_Tr_up_1.reshape(-1, 1)).type(torch.Tensor)

        target_Tr_down_1 = np.array(target_Tr_down)
        target_Tr_down_2 = torch.from_numpy(target_Tr_down_1.reshape(-1, 1)).type(torch.Tensor)

        target = torch.cat(
            [target_ic_2, target_a_2, target_e_2, target_Tr_down_2, target_Tr_up_2, target_ip_2, target_r_2,
             target_z_2], dim=1)

        test_data.append((input_full, target))
    return test_data

def data_process_train_no_vloop_TF(train_data_index, seq, mean_len, seq_interval, smooth_type):
    input_u = []
    input_ic = []
    input_ip = []
    input_r = []
    input_z = []
    input_a = []
    input_e = []
    input_Tr_down = []
    input_Tr_up = []

    target_ic = []
    target_ip = []
    target_r = []
    target_z = []
    target_e = []
    target_a = []
    target_Tr_down = []
    target_Tr_up = []

    for i in train_data_index:
        shot = number[i]
        u_norm = (u[i] - min_u) / (max_u - min_u)
        ip_norm = (ip[i] - min_ip) / (max_ip - min_ip)
        ic_norm = (ic[i] - min_ic) / (max_ic - min_ic)
        r_norm = (r[i] - min_r) / (max_r - min_r)
        z_norm = (z[i] - min_z) / (max_z - min_z)

        a_norm = (a[i] - min_a) / (max_a - min_a)
        e_norm = (e[i] - min_e) / (max_e - min_e)
        Tr_down_norm = (Tr_down[i] - min_Tr_down) / (max_Tr_down - min_Tr_down)
        Tr_up_norm = (Tr_up[i] - min_Tr_up) / (max_Tr_up - min_Tr_up)

        if smooth_type == 1:
            if shot in shot_step:
                for k in range(data_repeat):
                    for index in range(len(u_norm) - seq -mean_len + 1):
                        input_ip.append(ip_norm[index:index + seq])
                        input_r.append(r_norm[index:index + seq])
                        input_z.append(z_norm[index:index + seq])
                        input_a.append(a_norm[index:index + seq])
                        input_e.append(e_norm[index:index + seq])
                        input_Tr_down.append(Tr_down_norm[index:index + seq])
                        input_Tr_up.append(Tr_up_norm[index:index + seq])

                        input_ic.append(ic_norm[index: index + seq])
                        input_u.append(u_norm[index + 1: index + seq + 1])

                        target_ic.append(ic_norm[index + seq])
                        target_ip.append(np.mean(ip_norm[index + seq: index + seq + mean_len], axis=0))
                        target_r.append(np.mean(r_norm[index + seq: index + seq + mean_len], axis=0))
                        target_z.append(np.mean(z_norm[index + seq: index + seq + mean_len], axis=0))
                        target_a.append(np.mean(a_norm[index + seq: index + seq + mean_len], axis=0))
                        target_e.append(np.mean(e_norm[index + seq: index + seq + mean_len], axis=0))
                        target_Tr_up.append(np.mean(Tr_up_norm[index + seq: index + seq + mean_len], axis=0))
                        target_Tr_down.append(np.mean(Tr_down_norm[index + seq: index + seq + mean_len], axis=0))
            else:
                for index in range(len(u_norm) - seq - mean_len + 1):
                    input_ip.append(ip_norm[index:index + seq])
                    input_r.append(r_norm[index:index + seq])
                    input_z.append(z_norm[index:index + seq])
                    input_a.append(a_norm[index:index + seq])
                    input_e.append(e_norm[index:index + seq])
                    input_Tr_down.append(Tr_down_norm[index:index + seq])
                    input_Tr_up.append(Tr_up_norm[index:index + seq])

                    input_ic.append(ic_norm[index: index + seq])
                    input_u.append(u_norm[index + 1: index + seq + 1])

                    target_ic.append(ic_norm[index + seq])
                    target_ip.append(np.mean(ip_norm[index + seq: index + seq + mean_len], axis=0))
                    target_r.append(np.mean(r_norm[index + seq: index + seq + mean_len], axis=0))
                    target_z.append(np.mean(z_norm[index + seq: index + seq + mean_len], axis=0))
                    target_a.append(np.mean(a_norm[index + seq: index + seq + mean_len], axis=0))
                    target_e.append(np.mean(e_norm[index + seq: index + seq + mean_len], axis=0))
                    target_Tr_up.append(np.mean(Tr_up_norm[index + seq: index + seq + mean_len], axis=0))
                    target_Tr_down.append(np.mean(Tr_down_norm[index + seq: index + seq + mean_len], axis=0))

        elif smooth_type == 2:
            if shot in shot_step:
                for k in range(data_repeat):
                    for index in range(len(u_norm) - seq - mean_len + 1):
                        input_ip.append(ip_norm[index:index + seq])
                        input_r.append(r_norm[index:index + seq])
                        input_z.append(z_norm[index:index + seq])
                        input_a.append(a_norm[index:index + seq])
                        input_e.append(e_norm[index:index + seq])
                        input_Tr_down.append(Tr_down_norm[index:index + seq])
                        input_Tr_up.append(Tr_up_norm[index:index + seq])
                        input_ic.append(ic_norm[index: index + seq])
                        input_u.append(u_norm[index + mean_len: index + seq + mean_len])
                        target_ic.append(ic_norm[index + seq])
                        target_ip.append(np.mean(ip_norm[index + seq + 1 -mean_len: index + seq + 1], axis=0))
                        target_r.append(np.mean(r_norm[index + seq + 1 -mean_len: index + seq + 1], axis=0))
                        target_z.append(np.mean(z_norm[index + seq + 1 -mean_len: index + seq + 1], axis=0))
                        target_a.append(np.mean(a_norm[index + seq + 1 -mean_len: index + seq + 1], axis=0))
                        target_e.append(np.mean(e_norm[index + seq + 1 -mean_len: index + seq + 1], axis=0))
                        target_Tr_up.append(np.mean(Tr_up_norm[index + seq + 1 - mean_len: index + seq + 1], axis=0))
                        target_Tr_down.append(np.mean(Tr_down_norm[index + seq + 1 - mean_len: index + seq + 1], axis=0))
            else:
                for index in range(len(u_norm) - seq - mean_len + 1):
                    input_ip.append(ip_norm[index:index + seq])
                    input_r.append(r_norm[index:index + seq])
                    input_z.append(z_norm[index:index + seq])
                    input_a.append(a_norm[index:index + seq])
                    input_e.append(e_norm[index:index + seq])
                    input_Tr_down.append(Tr_down_norm[index:index + seq])
                    input_Tr_up.append(Tr_up_norm[index:index + seq])
                    input_ic.append(ic_norm[index: index + seq])
                    input_u.append(u_norm[index + mean_len: index + seq + mean_len])
                    target_ic.append(ic_norm[index + seq])
                    target_ip.append(np.mean(ip_norm[index + seq + 1 - mean_len: index + seq + 1], axis=0))
                    target_r.append(np.mean(r_norm[index + seq + 1 - mean_len: index + seq + 1], axis=0))
                    target_z.append(np.mean(z_norm[index + seq + 1 - mean_len: index + seq + 1], axis=0))
                    target_a.append(np.mean(a_norm[index + seq + 1 - mean_len: index + seq + 1], axis=0))
                    target_e.append(np.mean(e_norm[index + seq + 1 - mean_len: index + seq + 1], axis=0))
                    target_Tr_up.append(np.mean(Tr_up_norm[index + seq + 1 - mean_len: index + seq + 1], axis=0))
                    target_Tr_down.append(np.mean(Tr_down_norm[index + seq + 1 - mean_len: index + seq + 1], axis=0))

    input_ic_1 = np.array(input_ic)
    input_ic_2 = torch.from_numpy(input_ic_1.reshape(-1, seq, 18)).type(torch.Tensor)

    input_u_1 = np.array(input_u)
    input_u_2 = torch.from_numpy(input_u_1.reshape(-1, seq, 17)).type(torch.Tensor)

    input_ip_1 = np.array(input_ip)
    input_ip_2 = torch.from_numpy(input_ip_1.reshape(-1, seq, 1)).type(torch.Tensor)

    input_r_1 = np.array(input_r)
    input_r_2 = torch.from_numpy(input_r_1.reshape(-1, seq, 1)).type(torch.Tensor)

    input_z_1 = np.array(input_z)
    input_z_2 = torch.from_numpy(input_z_1.reshape(-1, seq, 1)).type(torch.Tensor)

    input_a_1 = np.array(input_a)
    input_a_2 = torch.from_numpy(input_a_1.reshape(-1, seq, 1)).type(torch.Tensor)

    input_e_1 = np.array(input_e)
    input_e_2 = torch.from_numpy(input_e_1.reshape(-1, seq, 1)).type(torch.Tensor)

    input_Tr_down_1 = np.array(input_Tr_down)
    input_Tr_down_2 = torch.from_numpy(input_Tr_down_1.reshape(-1, seq, 1)).type(torch.Tensor)

    input_Tr_up_1 = np.array(input_Tr_up)
    input_Tr_up_2 = torch.from_numpy(input_Tr_up_1.reshape(-1, seq, 1)).type(torch.Tensor)


    input_full = torch.cat(
        [input_u_2, input_ic_2, input_a_2, input_e_2, input_Tr_down_2,
         input_Tr_up_2, input_ip_2, input_r_2, input_z_2], dim=2)

    target_ip_1 = np.array(target_ip)
    target_ip_2 = torch.from_numpy(target_ip_1.reshape(-1, 1)).type(torch.Tensor)

    target_ic_1 = np.array(target_ic)
    target_ic_2 = torch.from_numpy(target_ic_1.reshape(-1, 18)).type(torch.Tensor)

    target_r_1 = np.array(target_r)
    target_r_2 = torch.from_numpy(target_r_1.reshape(-1, 1)).type(torch.Tensor)

    target_z_1 = np.array(target_z)
    target_z_2 = torch.from_numpy(target_z_1.reshape(-1, 1)).type(torch.Tensor)

    target_a_1 = np.array(target_a)
    target_a_2 = torch.from_numpy(target_a_1.reshape(-1, 1)).type(torch.Tensor)

    target_e_1 = np.array(target_e)
    target_e_2 = torch.from_numpy(target_e_1.reshape(-1, 1)).type(torch.Tensor)

    target_Tr_up_1 = np.array(target_Tr_up)
    target_Tr_up_2 = torch.from_numpy(target_Tr_up_1.reshape(-1, 1)).type(torch.Tensor)

    target_Tr_down_1 = np.array(target_Tr_down)
    target_Tr_down_2 = torch.from_numpy(target_Tr_down_1.reshape(-1, 1)).type(torch.Tensor)

    target = torch.cat(
        [target_ic_2, target_a_2, target_e_2, target_Tr_down_2, target_Tr_up_2, target_ip_2, target_r_2,
         target_z_2],
        dim=1)
    train_data = (input_full, target)
    return train_data

def data_process_test_no_vloop_TF(test_data_index, seq, seq_interval, mean_len):
    test_data = []
    for i in test_data_index:
        input_u = []
        input_ic = []
        input_ip = []
        input_r = []
        input_z = []
        input_a = []
        input_e = []
        input_Tr_down = []
        input_Tr_up = []

        target_ic = []
        target_ip = []
        target_r = []
        target_z = []
        target_e = []
        target_a = []
        target_Tr_down = []
        target_Tr_up = []

        u_norm = (u[i] - min_u) / (max_u - min_u)
        ip_norm = (ip[i] - min_ip) / (max_ip - min_ip)
        ic_norm = (ic[i] - min_ic) / (max_ic - min_ic)
        r_norm = (r[i] - min_r) / (max_r - min_r)
        z_norm = (z[i] - min_z) / (max_z - min_z)

        a_norm = (a[i] - min_a) / (max_a - min_a)
        e_norm = (e[i] - min_e) / (max_e - min_e)
        Tr_down_norm = (Tr_down[i] - min_Tr_down) / (max_Tr_down - min_Tr_down)
        Tr_up_norm = (Tr_up[i] - min_Tr_up) / (max_Tr_up - min_Tr_up)
        for index in range(len(u_norm) - seq - 1):
            input_ip.append(ip_norm[index:index + seq])
            input_r.append(r_norm[index:index + seq])
            input_z.append(z_norm[index:index + seq])
            input_a.append(a_norm[index:index + seq])
            input_e.append(e_norm[index:index + seq])
            input_Tr_down.append(Tr_down_norm[index:index + seq])
            input_Tr_up.append(Tr_up_norm[index:index + seq])

            input_ic.append(ic_norm[index: index + seq])
            input_u.append(u_norm[index+1: index + seq+1])

            target_ic.append(ic_norm[index + seq])
            target_ip.append(ip_norm[index + seq])
            target_r.append(r_norm[index + seq])
            target_z.append(z_norm[index + seq])
            target_a.append(a_norm[index + seq])
            target_e.append(e_norm[index + seq])
            target_Tr_up.append(Tr_up_norm[index + seq])
            target_Tr_down.append(Tr_down_norm[index + seq])

        input_ic_1 = np.array(input_ic)
        input_ic_2 = torch.from_numpy(input_ic_1.reshape(-1, seq, 18)).type(torch.Tensor)

        input_u_1 = np.array(input_u)
        input_u_2 = torch.from_numpy(input_u_1.reshape(-1, seq, 17)).type(torch.Tensor)

        input_ip_1 = np.array(input_ip)
        input_ip_2 = torch.from_numpy(input_ip_1.reshape(-1, seq, 1)).type(torch.Tensor)

        input_r_1 = np.array(input_r)
        input_r_2 = torch.from_numpy(input_r_1.reshape(-1, seq, 1)).type(torch.Tensor)

        input_z_1 = np.array(input_z)
        input_z_2 = torch.from_numpy(input_z_1.reshape(-1, seq, 1)).type(torch.Tensor)

        input_a_1 = np.array(input_a)
        input_a_2 = torch.from_numpy(input_a_1.reshape(-1, seq, 1)).type(torch.Tensor)

        input_e_1 = np.array(input_e)
        input_e_2 = torch.from_numpy(input_e_1.reshape(-1, seq, 1)).type(torch.Tensor)

        input_Tr_down_1 = np.array(input_Tr_down)
        input_Tr_down_2 = torch.from_numpy(input_Tr_down_1.reshape(-1, seq, 1)).type(torch.Tensor)

        input_Tr_up_1 = np.array(input_Tr_up)
        input_Tr_up_2 = torch.from_numpy(input_Tr_up_1.reshape(-1, seq, 1)).type(torch.Tensor)

        input_full = torch.cat(
            [input_u_2, input_ic_2, input_a_2, input_e_2, input_Tr_down_2,
             input_Tr_up_2, input_ip_2, input_r_2, input_z_2], dim=2)

        target_ip_1 = np.array(target_ip)
        target_ip_2 = torch.from_numpy(target_ip_1.reshape(-1, 1)).type(torch.Tensor)

        target_ic_1 = np.array(target_ic)
        target_ic_2 = torch.from_numpy(target_ic_1.reshape(-1, 18)).type(torch.Tensor)

        target_r_1 = np.array(target_r)
        target_r_2 = torch.from_numpy(target_r_1.reshape(-1, 1)).type(torch.Tensor)

        target_z_1 = np.array(target_z)
        target_z_2 = torch.from_numpy(target_z_1.reshape(-1, 1)).type(torch.Tensor)

        target_a_1 = np.array(target_a)
        target_a_2 = torch.from_numpy(target_a_1.reshape(-1, 1)).type(torch.Tensor)

        target_e_1 = np.array(target_e)
        target_e_2 = torch.from_numpy(target_e_1.reshape(-1, 1)).type(torch.Tensor)

        target_Tr_up_1 = np.array(target_Tr_up)
        target_Tr_up_2 = torch.from_numpy(target_Tr_up_1.reshape(-1, 1)).type(torch.Tensor)

        target_Tr_down_1 = np.array(target_Tr_down)
        target_Tr_down_2 = torch.from_numpy(target_Tr_down_1.reshape(-1, 1)).type(torch.Tensor)

        target = torch.cat(
            [target_ic_2, target_a_2, target_e_2, target_Tr_down_2, target_Tr_up_2, target_ip_2, target_r_2,
             target_z_2], dim=1)

        test_data.append((input_full, target))
    return test_data