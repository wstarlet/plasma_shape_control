import torch
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import argparse
import datetime
import matplotlib.pyplot as plt
import json
import wandb
from utils.utils import make_dir, save_args, plot_rewards
from agent.ppo_agent_beta_lrdc import ReplayBuffer, ppo_agent
from tokamak_env.tokamak_env_update_in44_out25 import env_lstm
from tokamak_env.tokamak_env_update_in44_out25 import shot_base_data_extraction
import sys, os

curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add to system path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# min_max scale
min_max_path = r"data/250331/min_max_scale_1241_11908/"
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

max_s = np.concatenate((max_a, max_e, max_Tr_down, max_Tr_up, max_ip, max_r, max_z), axis=0)
min_s = np.concatenate((min_a, min_e, min_Tr_down, min_Tr_up, min_ip, min_r, min_z), axis=0)

state_sub = max_s-min_s
u_sub = max_u-min_u
print(f'min_s: {min_s}\n'
      f'min_u: {min_u}\n'
      f's_sub: {state_sub}\n '
      f'u_sub:{u_sub}')

base_shot = 11638
start_time = 700
end_time = 1200
control_len = end_time-start_time

project_name = f'train_ppo_{start_time}_{end_time}_'+str(base_shot)
name = str(base_shot)+'_train_reward'
wandb.init(project=project_name, name=name)

target_tensor_norm_7, u_tensor_norm_17, \
lv_TF_tensor_norm_2, ic_tensor_norm_18, start_time_index = shot_base_data_extraction(shot_number=base_shot, start_time=start_time)

t_base_shot = np.load(f'./data/base_shot/ori/{base_shot}/t_shot{base_shot}.npy', allow_pickle=True)
seq_len = 30
target_tensor_norm_7_concat = torch.cat([target_tensor_norm_7[:start_time_index+control_len+1, :],
                                         torch.flip(target_tensor_norm_7[start_time_index:start_time_index + control_len,:],[0])],dim=0)


def save_and_plot_results(s_pred, u_pred, ic_pred, args, base_shot, start_time, end_time, label):
    print(f"\n[INFO] Generating comparison plot for '{label}'...")

    try:
        s_true_full = np.load(f'./data/base_shot/ori/{base_shot}/s_shot{base_shot}.npy', allow_pickle=True)
        t_true_full = np.load(f'./data/base_shot/ori/{base_shot}/t_shot{base_shot}.npy', allow_pickle=True)
    except FileNotFoundError:
        print(f"[ERROR] Real data file for shot {base_shot} not found. Unable to plot and compare.")
        return

    start_index = np.where(t_true_full == start_time)[0][0]
    control_len = len(s_pred)

    t_plot = t_true_full[start_index: start_index + control_len]

    s_true_sliced = s_true_full[start_index: start_index + control_len, :]

    s_init = s_true_full[start_index, :]
    s_pred = np.concatenate((s_init[np.newaxis, :], s_pred[:-1, :]), axis=0)

    err_full = np.abs(s_pred - s_true_sliced)
    err_mean = np.mean(err_full, axis=0)
    a_err, e_err, Tr_down_err, Tr_up_err, ip_err, r_err, z_err = err_mean

    s_true_full_reordered = np.zeros_like(s_true_full)
    s_true_full_reordered[:, 0] = s_true_full[:, 4]  # Ip
    s_true_full_reordered[:, 1:5] = s_true_full[:, 0:4]  # a, k, dl, du
    s_true_full_reordered[:, 5:7] = s_true_full[:, 5:7]  # R, Z

    s_pred_reordered = np.zeros_like(s_pred)
    s_pred_reordered[:, 0] = s_pred[:, 4]
    s_pred_reordered[:, 1:5] = s_pred[:, 0:4]
    s_pred_reordered[:, 5:7] = s_pred[:, 5:7]

    print(f"[INFO] Generating comparison plot for '{label}'...")
    fig, axes = plt.subplots(7, 1, figsize=(8, 10), sharex=True)
    fig.suptitle(f'RL Control vs Target (Shot #{base_shot}, Model: {label.replace("_", " ").title()})', fontsize=16)

    font = 14
    font_err = 13
    font_legend = 11
    y_label_x = -0.1

    ylabels = ['$I_{p} (kA)$', 'a (cm)', '$\kappa$', '$\delta_l$', '$\delta_u$', '$R (cm)$', '$Z (cm)$']
    error_texts = [f'Error: {ip_err:.2f} kA', f'Error: {a_err:.2f} cm', f'Error: {e_err:.4f}',
                   f'Error: {Tr_down_err:.4f}', f'Error: {Tr_up_err:.4f}', f'Error: {r_err:.2f} cm',
                   f'Error: {z_err:.2f} cm']

    for j in range(7):
        ax = axes[j]

        ax.plot(t_true_full, s_true_full_reordered[:, j], label=f'Target (Shot #{base_shot})', color='k',
                linestyle='--', linewidth=1.5)

        ax.plot(t_plot, s_pred_reordered[:, j], label='RL control results', color='r', linestyle='-', linewidth=2.5)

        ax.tick_params(axis='both', labelsize=font, direction='in', top=True, right=True)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(3))
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.yaxis.set_label_coords(x=y_label_x, y=0.5)

        ax.axvline(x=start_time, color='orange', linestyle='-.', linewidth=1.5)
        ax.axvline(x=end_time, color='orange', linestyle='-.', linewidth=1.5)

        ymin = np.min(s_true_full_reordered[:, j])
        ymax = np.max(s_true_full_reordered[:, j])
        yrange = ymax - ymin
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax.fill_between(x=[start_time, end_time], y1=ax.get_ylim()[0], y2=ax.get_ylim()[1], color='orange', alpha=0.1)

        ax.set_ylabel(ylabels[j], fontsize=font)
        ax.text(0.98, 0.95, error_texts[j], transform=ax.transAxes, fontsize=font_err, color='r', ha='right', va='top')

        if j == 0:
            legend = ax.legend(loc='lower left', fontsize=font_legend, ncol=1)
            legend.get_frame().set_edgecolor('black')
            ax.text((start_time + end_time) / 2, 1.05, 'RL Control Phase', transform=ax.get_xaxis_transform(),
                    fontsize=font, weight='bold', color='orange', ha='center')
            ax.yaxis.set_major_locator(plt.MultipleLocator(100))

    axes[-1].set_xlim(t_true_full[0], t_true_full[-1] + 15)
    axes[-1].set_xlabel('Time (ms)', fontsize=font)
    axes[-1].xaxis.set_major_locator(plt.MultipleLocator(500))
    axes[-1].xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    axes[-1].tick_params(axis='x', which='minor', direction='in')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0, bottom=0.08, top=0.92)

    png_filename = os.path.join(args.result_path, f"{base_shot}_RL_control_{label}.png")
    plt.savefig(png_filename)
    print(f"[SUCCESS] Comparison plot for '{label}' successfully saved to: {png_filename}")


def train(args, agent, env):
    agent.actor.train()
    agent.critic.train()
    print('start training!')
    replay_buffer = ReplayBuffer(args)
    running_rewards = []
    ma_rewards = []
    total_steps = 0

    for i in range(args.max_episodes):
        episode_reward = 0
        i_episode = i + 1
        step = 0
        done = False
        state_seq_7, a_in, agent_s = env.reset()  # state_seq_7:1*30*7 action_reset:1*30*17  state_14: 1*14
        while not done:
            a_last = a_in[:, -1, :]  # a tensor (1,17)
            a, a_logprob = agent.choose_action(agent_s, a_last)  # a tensor (1,17)
            a_in = torch.cat([a_in[:, 1:, :], a.unsqueeze(0)], dim=1)  # 1*seq_len*17

            state_14_, ic_pred, reward, done = env.step(state_seq_7, a_in, step) # state_14:1*14  ic_pred: 1*25
            agent_s_ = state_14_
            state_seq_7 = torch.cat([state_seq_7[:, 1:, :], state_14_[:, :-7].unsqueeze(0)], dim=1)  # 1*30*7
            total_steps += 1
            step += 1

            # buffer
            replay_buffer.s.append(agent_s)
            replay_buffer.a.append(a)
            replay_buffer.a_logprob.append(a_logprob)
            replay_buffer.r.append(reward)
            replay_buffer.s_.append(agent_s_)
            replay_buffer.done.append(done)
            replay_buffer.count += 1

            agent_s = agent_s_
            episode_reward += (reward.cpu().item())

            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.clear_memory()
                replay_buffer.count = 0

        running_rewards.append(episode_reward)
        if i == 0:
            max_episode_reward = episode_reward
        else:
            if episode_reward > max_episode_reward:
                max_episode_reward = episode_reward
                agent.save_real_time(path=args.model_path)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * episode_reward)
        else:
            ma_rewards.append(episode_reward)

        print(f'Epoch:{i_episode}/{args.max_episodes}, Reward:{episode_reward:.2f}')
        wandb.log({'epoch_reward': episode_reward})
    print('Finish training!')
    return running_rewards, ma_rewards


def RL_test(args, agent, env):
    agent.actor.eval()
    agent.critic.eval()
    print('start testing!')
    episode_reward = 0
    step = 0
    done = False

    state_seq_7, a_in, agent_s = env.reset()

    while not done:
        a_last = a_in[:, -1, :]  # a tensor (1,17)
        a, a_logprob = agent.choose_action(agent_s, a_last)  # a tensor (1,17)
        a_in = torch.cat([a_in[:, 1:, :], a.unsqueeze(0)], dim=1)  # 1*seq_len*17
        state_14_, ic_pred, reward, done = env.step(state_seq_7, a_in, step)  # state_14:1*14  ic_pred: 1*25
        agent_s_ = state_14_
        state_seq_7 = torch.cat([state_seq_7[:, 1:, :], state_14_[:, :-7].unsqueeze(0)], dim=1)  # 1*30*7
        if step == 0:
            s_test = state_14_[:, :-7].detach()  # 1*7
            u_test = a.detach()  # 1*17
            ic_test = ic_pred.detach()  # 1*18
        elif step >= 1:
            s_test = torch.cat([s_test, state_14_[:, :-7].detach()], dim=0)
            u_test = torch.cat([u_test, a.detach()], dim=0)
            ic_test = torch.cat([ic_test, ic_pred.detach()], dim=0)

        step += 1
        agent_s = agent_s_
        episode_reward += (reward.cpu().item())
    print(f'Reward:{episode_reward:.2f}')
    print('test finished!')
    return s_test.cpu().numpy(), u_test.cpu().numpy(), ic_test.cpu().numpy()

def model_test(args, agent, env):
    print('start model testing!')
    episode_reward = 0

    step = 0
    done = False

    state_seq_7, a_in, agent_s = env.reset()  # state_seq_7:1*30*7 action_reset:1*30*17  state_14: 1*14
    while not done:
        state_14_, ic_pred, reward, done = env.model_test_step(state_seq_7, step)  # state_14:1*14  ic_pred: 1*25
        state_seq_7 = torch.cat([state_seq_7[:, 1:, :], state_14_[:, :-7].unsqueeze(0)], dim=1)  # 1*30*7
        if step == 0:
            s_test = state_14_[:, :-7].detach()  # 1*7
        elif step >= 1:
            s_test = torch.cat([s_test, state_14_[:, :-7].detach()], dim=0)
        step += 1
        episode_reward += (reward.cpu().item())

    s_test_np = s_test[:, -7:].cpu().numpy() * (max_s - min_s) + min_s
    t_test = np.arange(start_time, start_time + len(s_test_np))
    s_true = target_tensor_norm_7.cpu().numpy() * (max_s - min_s) + min_s

    t_end_index = np.where(t_base_shot == end_time)[0][0]
    t_true = t_base_shot[:t_end_index, :]
    s_true = s_true[:t_end_index, :]

    fig, axes = plt.subplots(7, 1, figsize=(12, 14))  # state
    for j in range(7):
        ax = axes[j]
        ax.plot(t_test, s_test_np[:, j], color='r', linestyle='--',linewidth=2)
        ax.plot(t_true, s_true[:, j], color='g', linewidth=2)
    # plt.show()
    plt.savefig(args.temp_result_path + 'model_test.png')
    plt.title(f'Test Reward:{episode_reward:.2f}')
    print(f'Test Reward:{episode_reward:.2f}')
    return s_test_np

def load_parameters(param_file):
    with open(param_file, 'r') as f:
        params = json.load(f)
    return params

if __name__ == '__main__':
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Obtain current time
    parser = argparse.ArgumentParser("Hyperparameters Setting for lstm_ppo")
    # train setting
    parser.add_argument("--max_episodes", type=int, default=10)
    parser.add_argument("--max_episode_steps", type=int, default=control_len)
    parser.add_argument("--max_step", type=int, default=control_len)
    parser.add_argument("--critic_hidden_width", type=int, default=64)  # critic hidden width
    parser.add_argument("--actor_hidden_width", type=int, default=64)  # actor hidden width
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--smoothmax_alpha", type=int, default=0, help="smoothmax alpha parameter")
    parser.add_argument("--batch_size", type=int, default=1500, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=256, help="Minibatch size")
    parser.add_argument("--lr_a", type=float, default=1e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=1e-4, help="Learning rate of critic")
    parser.add_argument("--delta_u", type=float, default=20, help="allowed delta_u")

    # ppo parameters
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.02, help="PPO clip parameter")  # 0.02
    parser.add_argument("--K_epochs", type=int, default=5, help="Update times per batch")
    parser.add_argument("--entropy_coef", type=float, default=0.05, help="policy entropy coef")
    parser.add_argument("--dropout_rate", type=float, default=0, help="dropout")
    parser.add_argument("--delta_e", type=float, default=0.05, help="delta e")

    # tricks
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="advantage normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="reward scaling")
    parser.add_argument("--use_lr_decay", type=bool, default=True)
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=False, help="orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="set Adam epsilon=1e-5")
    parser.add_argument("--use_noise", type=bool, default=False, help="add noise to env output")
    parser.add_argument("--noise_std", type=float, default=0, help="noise std")
    parser.add_argument("--use_ref", type=bool, default=False, help="use joint target or not")

    # reward setting
    parser.add_argument("--reward_mode", type=int, default=1,
                        help="0：no penalty  1：delta_u constraint and ic_constraint")
    # file setting
    parser.add_argument('--result_path', default=curr_path + '/RL_train/' + curr_time + f'_{base_shot}_{start_time}_{end_time}'+ '/results/')
    parser.add_argument('--temp_result_path', default=curr_path + '/RL_train/' + curr_time + f'_{base_shot}_{start_time}_{end_time}' + '/temp_results/')
    parser.add_argument('--model_path', default=curr_path + '/RL_train/' + curr_time + f'_{base_shot}_{start_time}_{end_time}' + '/models/')  # path to save models
    parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")
    parser.add_argument('--dynamics_model_path',default='dynamics/dynamics_model_train_log_250331/in44_out25_dynamics/2025-10-24_15-47-56-lstm_model_5')

    lstm_model_path = parser.parse_args().dynamics_model_path
    saved_params_path_dynamics = os.path.join(lstm_model_path, 'parameters.json')

    if os.path.exists(saved_params_path_dynamics):
        saved_params = load_parameters(saved_params_path_dynamics)
        for key, value in saved_params.items():
            current_value = getattr(parser.parse_args(), key, None)
            if current_value is None:
                parser.set_defaults(**{key: value})
    else:
        print('no dynamics model params. path!')
    args = parser.parse_args()
    args.start_time = start_time
    args.base_shot = base_shot
    args.end_time = end_time
    args.use_init_noise = False

    a_threshold = torch.from_numpy(args.delta_u / (max_u - min_u)).type(torch.Tensor).unsqueeze(0).to(
        device)  # single step delta_u constraint
    diff_u_tensor = torch.from_numpy(max_u - min_u).type(torch.Tensor).to(device)  # 1*17

    args.max_train_steps = args.max_episodes * 1.5 * args.max_episode_steps
    args.device = device
    reward_w = torch.tensor([[1, 1, 1, 1, 1, 1, 1]]).to(
        device)  # a_norm, e_norm, Tr_down_norm, Tr_up_norm, ip_norm, r_norm, z_norm
    args.reward_w = reward_w
    args.u_threshold = a_threshold  # 1*17

    args.diff_u = diff_u_tensor
    # Constraint on the coil current within the zero-crossing deadband
    ic_zero_upper = [1000, 1000, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
    ic_zero_lower = [-1000, -1000, -200, -200, -200, -200, -200, -200, -200, -200, -200, -200, -200, -200, -200, -200,
                     -200, -200]
    ic_zero_upper_norm = (ic_zero_upper - min_ic) / (max_ic - min_ic)
    ic_zero_lower_norm = (ic_zero_lower - min_ic) / (max_ic - min_ic)
    args.ic_zero_upper_norm = ic_zero_upper_norm
    args.ic_zero_lower_norm = ic_zero_lower_norm

    args.state_dim = 14  # 14 a,e,Tr_down,Tr_up,ip,r,z,(a,e,Tr_down,Tr_up,ip,r,z)target
    args.action_dim = 17  # CSA,PF1U,PF1L,PF2U,PF2L...PF8U,PF8L

    args.lstm_in_dim = 44  # lstm_input
    args.lstm_out_dim = 25  # lstm_output

    args.reward_dim = 7
    args.seq_len = seq_len
    args.delta_e_norm = torch.from_numpy(args.delta_e/(max_e-min_e)).type(torch.Tensor).to(device)

    agent = ppo_agent(args)
    env = env_lstm(args)
    make_dir(args.result_path, args.model_path, args.temp_result_path)

    test_s = model_test(args, agent, env)
    # train
    running_rewards, ma_rewards = train(args, agent, env)
    save_args(args)
    agent.save(path=args.model_path)
    plot_rewards(running_rewards, ma_rewards, args, tag='# RL target 6226 train')
    
    # test
    agent.load(path=args.model_path)  # final model
    s_test, u_test, ic_test = RL_test(args, agent, env)
    s_pred = s_test * (max_s - min_s) + min_s
    u_pred = u_test * (max_u - min_u) + min_u
    ic_pred = ic_test * (max_ic - min_ic) + min_ic
    np.save(args.result_path + 'u_pred.npy', u_pred)
    np.save(args.result_path + 's_pred.npy', s_pred)
    np.save(args.result_path + 'ic_pred.npy', ic_pred)
    save_and_plot_results(
        s_pred=s_pred,
        u_pred=u_pred,
        ic_pred=ic_pred,
        args=args,
        base_shot=base_shot,
        start_time=start_time,
        end_time=end_time,
        label='final'
    )
    agent.load_real_time(path=args.model_path)  # best model
    s_test, u_test, ic_test = RL_test(args, agent, env)
    s_pred_real_time = s_test * (max_s - min_s) + min_s
    u_pred_real_time = u_test * (max_u - min_u) + min_u
    ic_pred_real_time = ic_test * (max_ic - min_ic) + min_ic
    np.save(args.result_path + 'u_pred_real_time.npy', u_pred)
    np.save(args.result_path + 's_pred_real_time.npy', s_pred)
    np.save(args.result_path + 'ic_pred_real_time.npy', ic_pred)
    save_and_plot_results(
        s_pred=s_pred_real_time,
        u_pred=u_pred_real_time,
        ic_pred=ic_pred_real_time,
        args=args,
        base_shot=base_shot,
        start_time=start_time,
        end_time=end_time,
        label='best'
    )
    # save onnx_actor
    agent.actor2onnx(path=args.model_path)





