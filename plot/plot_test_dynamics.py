import matplotlib.pyplot as plt
import os


def plot_test_update(test_pred_6, test_label, loss_test_mean, test_number, test_t, save_pic_folder):
    err_mean = loss_test_mean
    number = test_number
    t = test_t
    loss_a = str(round(err_mean[0], 3))
    loss_e = str(round(err_mean[1], 3))
    loss_Tr_down = str(round(err_mean[2], 3))
    loss_Tr_up = str(round(err_mean[3], 3))
    loss_ip = str(round(err_mean[4], 1))
    loss_r = str(round(err_mean[5], 3))
    loss_z = str(round(err_mean[6], 3))
    fig, axes = plt.subplots(7, 1, figsize=(6, 9))
    plt.suptitle(
        '#' + str(number) + ' err_a:' + loss_a + ' err_e:' + loss_e + ' err_Tr_down:' + loss_Tr_down + ' err_Tr_up:'
        + loss_Tr_up + ' err_ip:' + loss_ip + ' err_r:' + loss_r + ' err_z:' + loss_z, fontsize=6)

    for j in range(7):
        ax = axes[j]
        if j != 6:
            ax.set_xticklabels([])

        for i in range(5):
            ax.plot(t[31:], test_pred_6[i, :, 18 + j], color='red', alpha=0.1, linewidth=2.5)  # plot each model's prediction

        ax.fill_between(
            t[31:],
            test_pred_6[:5, :, 18 + j].min(axis=0),
            test_pred_6[:5, :, 18 + j].max(axis=0),
            color='red', alpha=0.1, label='Ensemble')
        # plot avg prediction
        ax.plot(t[31:], test_pred_6[-1, :, 18 + j], label='Prediction', linestyle='--', color='red', linewidth=1.5)

        ax.plot(t[31:], test_label[:, 18 + j], label='Experiment', linestyle='-', color='k', linewidth=1.8)
        ax.legend(loc='lower right', fontsize=7)
        plt.tight_layout()

        if j == 0:
            ax.set_ylabel('a/cm')
        elif j == 1:
            ax.set_ylabel('$\kappa$')

        elif j == 2:
            ax.set_ylabel('$\delta_l$')

        elif j == 3:
            ax.set_ylabel('$\delta_u$')

        elif j == 4:
            ax.set_ylabel('$I_{p}/kA$')

        elif j == 5:
            ax.set_ylabel('$R/cm$')

        elif j == 6:
            ax.set_ylabel('$Z/cm$')
            ax.set_xlabel('Time (ms)')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0, bottom=0.05)
    save_path = os.path.join(save_pic_folder, str(number) + ' prediction.png')
    plt.savefig(save_path)
    plt.close(fig)