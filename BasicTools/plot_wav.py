import numpy as np
import matplotlib.pyplot as plt
import argparse
from . import wav_tools
from .plot_tools import plot_wav_spec


def plot_wav(wav_path, fig_path, is_mix=False, is_stereo=False, label=None,
        is_plot_spec=False, dpi=100):
    if not isinstance(wav_path, list):
        wav_path_all = [wav_path]
        if label is None:
            label_all = ['wav_0']
        else:
            label_all = [label]
    else:
        wav_path_all = wav_path
        if label is None:
            label_all = [f'wav_{i}' for i, _ in enumerate(wav_path_all)]
        else:
            label_all = label
    n_wav = len(wav_path_all)

    if is_mix:
        if is_stereo:
            n_col_fig = 2
        else:
            n_col_fig = 1
    else:
        n_col_fig = n_wav

    if is_plot_spec:
        fig, ax = plt.subplots(2, n_col_fig, tight_layout=True)
        ax = np.reshape(ax, (2, n_col_fig))
        ax_wav_all, ax_spec_all = ax[0, :], ax[1, :]
    else:
        fig, ax = plt.subplots(1, n_col_fig, tight_layout=True)
        if n_wav == 1:
            ax = [ax]
        ax_wav_all = ax
        ax_spec_all = [None for i in range(n_col_fig)]
    
    amp_max = 0
    for wav_i, wav_path in enumerate(wav_path_all):
        wav, fs = wav_tools.read_wav(wav_path)
        amp_max_tmp = np.max(np.abs(wav))
        if amp_max < amp_max_tmp:
            amp_max = amp_max_tmp

        if is_mix:
            if is_stereo:
                ax_wav_tmp = [ax_wav_all[0], ax_wav_all[1]]
                ax_spec_tmp = [ax_spec_all[ 0], ax_spec_all[1]]
            else:
                ax_wav_tmp = [ax_wav_all[0], ax_wav_all[0]]
                ax_spec_tmp = [ax_spec_all[0], ax_spec_all[0]]
        else:
            for ax_i in range(wav_i):
                ax_wav_all[ax_i].set_ylim([-amp_max, amp_max])
            ax_wav_tmp = [ax_wav_all[wav_i], ax_wav_all[wav_i]]
            ax_spec_tmp = [ax_spec_all[wav_i], ax_spec_all[wav_i]]

        for channel_i, channel_name in enumerate(['L', 'R']):
            label_tmp = label_all[wav_i]  # f'{label_all[wav_i]}_{channel_name}'
            plot_wav_spec(wav[:, channel_i], fs, ax_wav_tmp[channel_i], label_tmp, 
                    is_plot_spec, ax_spec_tmp[channel_i], amp_max=amp_max) 
            # ax_wav_tmp[channel_i].set_title(f'{label_tmp}')
    ax_wav_all[-1].legend()
    for ax in ax_wav_all[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)
    for ax in ax_spec_all[1:]:
        if ax is not None:
            plt.setp(ax.get_yticklabels(), visible=False)

    fig.savefig(fig_path, dpi=dpi)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--wav-path', dest='wav_path',type=str,
                        required=True,  action='append', help='path of wav file')
    parser.add_argument('--label', dest='label',type=str,
                        default=None,  action='append', help='path of wav file')
    parser.add_argument('--fig-path', dest='fig_path', type=str, 
                        required=True, help='path of figure to be saved')
    parser.add_argument('--is-mix', dest='is_mix', type=str2bool, 
                        default=False, help='')
    parser.add_argument('--is-stereo', dest='is_stereo', type=str2bool, 
                        default=False, help='')
    parser.add_argument('--is-plot-spec', dest='is_plot_spec', type=str2bool,
                        default=False, help='whether to plot the spectrum')
    parser.add_argument('--dpi', dest='dpi', type=int, default=100, 
                        help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    plot_wav(args.wav_path, args.fig_path, args.is_mix, args.is_stereo, 
            args.label, args.is_plot_spec, args.dpi)

if __name__ == '__main__':
    main()
