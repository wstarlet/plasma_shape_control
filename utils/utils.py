import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def plot_rewards(rewards, ma_rewards, cfg, tag='train'):
    plt.figure()
    plt.title("learning curve on {}".format(
        cfg.device))
    plt.xlabel('episodes')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    if cfg.save_fig:
        plt.savefig(cfg.result_path+"{}_rewards_curve".format(tag))
    #plt.show()


def make_dir(*paths):

    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_results(rewards, ma_rewards, tag='train', path='./results'):

    np.save(path+'{}_rewards.npy'.format(tag), rewards)
    np.save(path+'{}_ma_rewards.npy'.format(tag), ma_rewards)
    print('Result saved!')


def save_args(args):
    # save parameters
    argsDict = args.__dict__
    with open(args.result_path+'params.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    print("Parameters saved!")