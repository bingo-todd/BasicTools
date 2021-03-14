"""
terminal interface for plot_tools.plot_wav
"""


import numpy as np
import matplotlib.pyplot as plt
import argparse
from . import wav_tools
from . import plot_tools


def plot_wav(wav_path, ax=None, fig_path=None, mix_channel=False, dpi=100,
             interactive=False):

    wav, fs = wav_tools.read_wav(wav_path)
    # make wav 2d ndarray
    if len(wav.shape) == 1:
        wav = wav[:, np.newaxis]
    n_channel = wav.shape[1]

    amp_max = np.max(np.abs(wav))

    if mix_channel:
        n_col = 1  # plot all channel in one figure
    else:
        n_col = n_channel

    if ax is None:
        fig, ax = plt.subplots(2, n_col, tight_layout=True)
    else:
        fig = None
        if len(ax.shape) < 2:
            ax = ax[:, np.newaxis]

    if n_col == 1 and n_channel > 1:
        ax = np.repeat(ax, n_channel, axis=1)

    for channel_i in range(n_channel):
        plot_tools.plot_wav(wav=wav[:, channel_i],
                            fs=fs,
                            label=f'channel_{channel_i}',
                            ax_wav=ax[0, channel_i],
                            ax_specgram=ax[1, channel_i],
                            amp_max=amp_max)
    if mix_channel:
        ax[0, channel_i].legend()

    if fig_path is not None:
        fig.savefig(fig_path, dpi=dpi)

    return fig, ax


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--wav-path', dest='wav_path', type=str, nargs='+',
                        required=True,  help='path of wav file')
    parser.add_argument('--fig-path', dest='fig_path', type=str,
                        default=None, help='path of figure to be saved')
    parser.add_argument('--plot-spec', dest='plot_spec', type=str,
                        default='True', choices=['true', 'false'],
                        help='whether to plot the spectrum')
    parser.add_argument('--mix-channel', dest='mix_channel', type=str,
                        default='false', choices=['true', 'false'], help='')
    parser.add_argument('--interactive', dest='interactive', type=str,
                        default='false', choices=['true', 'false'], help='')
    parser.add_argument('--dpi', dest='dpi', type=int, default=100, help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    n_wav = len(args.wav_path)
    if args.mix_channel == 'true':
        fig, ax = plt.subplots(2, n_wav)
        if n_wav == 1:
            ax = ax[:, np.newaxis]
        for wav_i in range(n_wav):
            plot_wav(wav_path=args.wav_path[wav_i],
                     ax=ax[:, wav_i:wav_i+1],
                     mix_channel=True)
    else:
        fig, ax = plt.subplots(2, 2)
        for wav_i in range(n_wav):
            plot_wav(wav_path=args.wav_path[wav_i],
                     ax=ax,
                     mix_channel=False)

    if args.interactive == 'true':
        plt.show()

    if args.fig_path is not None:
        fig.savefig(args.fig_path, dpi=args.dpi)


if __name__ == '__main__':
    main()
