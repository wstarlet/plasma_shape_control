import numpy as np
import os

def min_max_scale_load(min_max_path):
    max_ic = np.atleast_1d(np.load(os.path.join(min_max_path, 'max_ic_round.npy'), allow_pickle=True))
    min_ic = np.atleast_1d(np.load(os.path.join(min_max_path, 'min_ic_round.npy'), allow_pickle=True))
    max_u = np.atleast_1d(np.load(os.path.join(min_max_path, 'max_u_round.npy'), allow_pickle=True))
    min_u = np.atleast_1d(np.load(os.path.join(min_max_path, 'min_u_round.npy'), allow_pickle=True))
    max_TF = np.atleast_1d(np.load(os.path.join(min_max_path, 'max_TF_round.npy'), allow_pickle=True))
    min_TF = np.atleast_1d(np.load(os.path.join(min_max_path, 'min_TF_round.npy'), allow_pickle=True))
    max_vloop = np.atleast_1d(np.load(os.path.join(min_max_path, 'max_vloop_round.npy'), allow_pickle=True))
    min_vloop = np.atleast_1d(np.load(os.path.join(min_max_path, 'min_vloop_round.npy'), allow_pickle=True))
    max_ip = np.atleast_1d(np.load(os.path.join(min_max_path, 'max_ip_round.npy'), allow_pickle=True))
    min_ip = np.atleast_1d(np.load(os.path.join(min_max_path, 'min_ip_round.npy'), allow_pickle=True))
    max_r = np.atleast_1d(np.load(os.path.join(min_max_path, 'max_r_round.npy'), allow_pickle=True))
    min_r = np.atleast_1d(np.load(os.path.join(min_max_path, 'min_r_round.npy'), allow_pickle=True))
    max_z = np.atleast_1d(np.load(os.path.join(min_max_path, 'max_z_round.npy'), allow_pickle=True))
    min_z = np.atleast_1d(np.load(os.path.join(min_max_path, 'min_z_round.npy'), allow_pickle=True))
    max_a = np.atleast_1d(np.load(os.path.join(min_max_path, 'max_a_round.npy'), allow_pickle=True))
    min_a = np.atleast_1d(np.load(os.path.join(min_max_path, 'min_a_round.npy'), allow_pickle=True))
    max_e = np.atleast_1d(np.load(os.path.join(min_max_path, 'max_e_round.npy'), allow_pickle=True))
    min_e = np.atleast_1d(np.load(os.path.join(min_max_path, 'min_e_round.npy'), allow_pickle=True))
    max_Tr_up = np.atleast_1d(np.load(os.path.join(min_max_path, 'max_Tr_up_round.npy'), allow_pickle=True))
    min_Tr_up = np.atleast_1d(np.load(os.path.join(min_max_path, 'min_Tr_up_round.npy'), allow_pickle=True))
    max_Tr_down = np.atleast_1d(np.load(os.path.join(min_max_path, 'max_Tr_down_round.npy'), allow_pickle=True))
    min_Tr_down = np.atleast_1d(np.load(os.path.join(min_max_path, 'min_Tr_down_round.npy'), allow_pickle=True))

    return max_ic, min_ic, max_u, min_u, max_TF, min_TF, max_vloop, min_vloop, \
           max_ip, min_ip, max_r, min_r, max_z, min_z, max_a, min_a, max_e, min_e, \
           max_Tr_down, min_Tr_down, max_Tr_up, min_Tr_up