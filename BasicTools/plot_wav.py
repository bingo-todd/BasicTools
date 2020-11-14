import numpy as np
import matplotlib.pyplot as plt
import argparse
from . import wav_tools
from .plot_tools import plot_wav_spec


def plot_wav(wav_path, fig_path, plot_spec=False, mix_channel=False, dpi=100):

    wav, fs = wav_tools.read_wav(wav_path)
    amp_max = np.max(np.abs(wav))
    if len(wav.shape) == 1:
        wav = wav[:, np.newaxis]
    n_channel = wav.shape[1]

    if mix_channel:
        n_col = 1
    else:
        n_col = n_channel

    if plot_spec:
        fig, ax = plt.subplots(3, n_col, tight_layout=True)
        ax_wav, ax_specgram, ax_spec = ax[0], ax[1], ax[2]
    else:
        fig, ax_wav = plt.subplots(1, n_col, tight_layout=True)
        if n_col > 1:
            ax_specgram = [None for i in range(n_col)]
            ax_spec = [None for i in range(n_col)]
        else:
            ax_specgram, ax_spec = None, None

    if n_col == 1:
        ax_wav = [ax_wav for i in range(n_channel)]
        ax_specgram = [ax_specgram for i in range(n_channel)]
        ax_spec = [ax_spec for i in range(n_channel)]

    for channel_i in range(n_channel):
        plot_wav_spec(wav[:, channel_i], fs, ax_wav[channel_i],
                      f'{channel_i}', plot_spec,
                      ax_specgram[channel_i], ax_spec[channel_i],
                      amp_max=amp_max)
        if ax_wav[channel_i] is not None:
            ax_wav[channel_i].legend()
    fig.savefig(fig_path, dpi=dpi)


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--wav-path', dest='wav_path', type=str,
                        required=True,  help='path of wav file')
    parser.add_argument('--fig-path', dest='fig_path', type=str,
                        required=True, help='path of figure to be saved')
    parser.add_argument('--plot-spec', dest='plot_spec', type=str,
                        default='false', choices=['true', 'false'],
                        help='whether to plot the spectrum')
    parser.add_argument('--mix-channel', dest='mix_channel', type=str,
                        default='false', choices=['true', 'false'],
                        help='')
    parser.add_argument('--dpi', dest='dpi', type=int, default=100,
                        help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    plot_wav(wav_path=args.wav_path,
             fig_path=args.fig_path,
             plot_spec=args.plot_spec == 'true',
             mix_channel=args.mix_channel == 'true',
             dpi=args.dpi)


if __name__ == '__main__':
    main()
