import numpy as np
import argparse
import matplotlib.pyplot as plt


from . import wav_tools
from . import fft
from .scale import erb


def plot_spec(wav_path, label, frame_len, frame_shift, spec_type='mean',
              linewidth=2, ax=None):
    wav, fs = wav_tools.read_wav(wav_path)
    if len(wav.shape) == 1:
        wav = wav[:, np.newaxis]
    wav_len, n_channel = wav.shape

    frame_len = np.int(frame_len*fs/1e3)
    frame_shift = np.int(frame_shift*fs/1e3)

    if ax is None:
        fig, ax = plt.subplots(1, n_channel, tight_layout=True, sharex=True,
                               sharey=True, figsize=[6.4+2*(n_channel-1), 4.8])
        if n_channel == 1:
            ax = [ax]
    else:
        fig = None

    for channel_i in range(n_channel):
        stft, t, freqs = fft.cal_stft(wav[:, channel_i], frame_len=frame_len,
                                      frame_shift=frame_shift, fs=fs)
        stft_amp = np.abs(stft)
        if spec_type == 'mean':
            spec_amp = np.mean(stft_amp, axis=0)
            ax[channel_i].plot(
                freqs, spec_amp, label=label, linewidth=linewidth)
        elif spec_type == 'range':
            ax[channel_i].fill_between(freqs, np.max(stft_amp, axis=0),
                                       np.min(stft_amp, axis=0))
            ax[channel_i].plot(
                freqs, spec_amp, label=label, linewidth=linewidth)
        ax[channel_i].set_xlim([freqs[0], freqs[-1]])
        ax[channel_i].xaxis.set_major_formatter(lambda x, pos: f'{x/1000}')
        plt.setp(
            ax[channel_i].get_xticklabels(), rotation=30,
            horizontalalignment='right')
        ax[channel_i].set_xlabel('freq(kHz)')
        ax[channel_i].set_xscale('erb')
        ax[channel_i].set_yscale('log')
        ax[channel_i].set_title(f'channel {channel_i}')
    return fig, ax


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--wav-path', dest='wav_path', required=True, type=str,
                        nargs='+', help='')
    parser.add_argument('--label', dest='label', default=None, type=str,
                        nargs='+', help='')
    parser.add_argument('--frame-len', dest='frame_len', type=int, default=20,
                        help='frame length in ms')
    parser.add_argument('--frame-shift', dest='frame_shift', type=int,
                        default=10, help='frame shift in ms')
    parser.add_argument('--linewidth', dest='linewidth', type=int,
                        default=2, help='')
    parser.add_argument('--fig-path', dest='fig_path', type=str, default=None,
                        help='')
    parser.add_argument('--interactive', dest='interactive', type=str,
                        default='false', choices=['true', 'false'], help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    n_wav = len(args.wav_path)
    if args.label is None:
        label = [None for i in range(n_wav)]
    else:
        label = args.label

    fig, ax = plot_spec(wav_path=args.wav_path[0],
                        label=label[0],
                        frame_len=args.frame_len,
                        frame_shift=args.frame_shift,
                        linewidth=args.linewidth)

    if len(args.wav_path) > 1:
        for wav_path_tmp, label_tmp in zip(args.wav_path[1:], label[1:]):
            plot_spec(wav_path=wav_path_tmp,
                      label=label_tmp,
                      frame_len=args.frame_len,
                      frame_shift=args.frame_shift,
                      linewidth=args.linewidth,
                      ax=ax)
    ax[-1].legend()

    if args.fig_path is not None:
        fig.savefig(args.fig_path)
        print(f'fig is saved to {args.fig_path}')

    if args.interactive == 'true':
        plt.show()


if __name__ == '__main__':
    main()
