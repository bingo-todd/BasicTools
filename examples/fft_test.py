import numpy as np
from BasicTools import wav_tools
from BasicTools import plot_tools

from BasicTools import fft


fig_dir = 'images/fft'


def stft_test(win_f, win_f_name):

    x, fs = wav_tools.read('data/binaural_1.wav')
    frame_len = int(fs*20e-3)
    frame_shift = int(fs*10e-3)
    stft = fft.cal_stft(x, frame_len=frame_len, win_f=win_f)
    n_frame, n_bin, _ = stft.shape

    t = np.arange(n_frame)*frame_shift
    freq = np.arange(n_bin)/frame_len*fs

    x_reconstrut = \
        fft.cal_istft(
            stft, frame_len=frame_len, win_f=win_f)[frame_len:-frame_len]
    x_reconstrut_win_norm, norm_ceofs = \
        fft.cal_istft(
            stft, frame_len=frame_len, win_norm=True, win_f=win_f,
            return_norm_coef=True)
    x_reconstrut_win_norm = x_reconstrut_win_norm[frame_len:-frame_len]

    fig, ax = plot_tools.subplots(1, 1, sharex=True)
    ax.plot(norm_ceofs)
    ax.set_xlim([5000, 6000])
    fig.savefig(f'{fig_dir}/norm_ceofs-{win_f_name}.png')

    fig, ax = plot_tools.subplots(2, 1, sharex=True)
    plot_tools.plot_matrix(
        20*np.log10(np.abs(stft[1:-1, 1:, 0].T)+1e-20), x=t, y=freq,
        vmin=-100, vmax=0, ax=ax[0], fig=fig)
    ax[0].set_yscale('mel')
    ax[0].set_ylabel('Frequency(Hz)')
    ax[0].set_title('L')
    plot_tools.plot_matrix(
        20*np.log10(np.abs(stft[1:-1, 1:, 1].T+1e-20)), x=t, y=freq,
        vmin=-100, vmax=0, ax=ax[1], fig=fig)
    ax[1].set_yscale('mel')
    ax[1].set_xlabel('time(s)')
    ax[1].set_ylabel('Frequency(Hz)')
    ax[1].set_title('R')
    fig.savefig(f'{fig_dir}/stft-{win_f_name}.png')

    fig, ax = plot_tools.subplots(3, 1, sharex=True, sharey=True)
    ax[0].plot(x)
    ax[0].set_title('x')
    ax[1].plot(x_reconstrut)
    ax[1].set_title('x_reconstrut')
    ax[2].plot(x_reconstrut_win_norm)
    ax[2].set_title('x_reconstrut')
    fig.savefig(f'{fig_dir}/x-{win_f_name}.png')


if __name__ == "__main__":
    stft_test(win_f=np.ones, win_f_name='rectangular')
    stft_test(win_f=np.hanning, win_f_name='hannning')
