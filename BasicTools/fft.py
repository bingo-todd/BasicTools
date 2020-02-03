import numpy as np
from . import wav_tools
# from . import auditory_scale


def cal_stft(x, win_f=np.hanning, frame_len=1024, shift_len=None,
             is_plot=False, fs=None, freq_scale='erb'):
    """short-time fast Fourier transform
    Args:
        x: 2d ndarray, [signal_len,n_chann]
        win_f: window function
        frame_len:
        shift_len:
        is_plot:
        fs: sample frquency
        freq_scale:
    Returns:
        stft: stft result with shape of [n_frame,n_bin]
        t: time of each frame (frame center)
        freq: frequency of each bins

    """
    if shift_len is None:
        shift_len = np.int(frame_len/2)

    frames = wav_tools.frame_data(x, frame_len, shift_len)
    window = win_f(frame_len)
    frames = np.multiply(frames, window)
    #
    fft_frames = np.fft.fft(frames)/frame_len
    half_frame_len = np.int(np.floor(frame_len/2.0)+1)
    stft = fft_frames[:, :half_frame_len]
    if fs is None:
        fs = 1
    freq = np.arange(half_frame_len)/frame_len*fs
    t = np.arange(frames.shape[0])*shift_len/fs

    return [stft, t, freq]


def example():
    import wav_tools
    import plot_tools
    # from .auditory_scale import mel
    import savefig
    import matplotlib

    font = {'size': 12}
    matplotlib.rc('font', **font)
    x, fs = wav_tools.wav_read('resource/tar.wav')
    stft, t, freq = cal_stft(x, fs=fs, frame_len=np.int(fs*50e-3))
    ax = plot_tools.imshow(x=t, y=freq, Z=20*np.log10(np.abs(stft)))
    ax.set_yscale('mel')
    ax.set_xlabel('time(s)')
    ax.set_ylabel('Frequency(Hz)')
    fig = ax.get_figure()
    savefig.savefig(fig, fig_name='stft', fig_dir='images/fft')


if __name__ == "__main__":
    example()
