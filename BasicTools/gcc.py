import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm
from . import wav_tools, plot_tools


def _cal_gcc_phat(x1, x2, max_delay, win_f, snr_thd):
    """subfunction of cal_gcc_phat
        gcc-phat, numpy ndarray with shape of [gcc_len]
    """
    n_sample = np.max((x1.shape[0], x2.shape[0]))
    if max_delay is None:
        max_delay = n_sample-1
    #
    window = win_f(n_sample)
    gcc_len = 2*n_sample-1
    x1_fft = np.fft.fft(np.multiply(x1, window), n=gcc_len)
    x2_fft = np.fft.fft(np.multiply(x2, window), n=gcc_len)
    gcc_fft = np.multiply(x1_fft, np.conj(x2_fft))
    # leave out frequency bins with small amplitude
    gcc_fft_amp = np.abs(gcc_fft)

    # clip small value to zeros
    eps = np.max(gcc_fft_amp)*(10**(snr_thd/10))
    gcc_fft[gcc_fft_amp < eps] = 0
    gcc_fft_amp[gcc_fft_amp < eps] = eps

    # phase transform
    gcc_phat_raw = np.real(np.fft.ifft(np.divide(gcc_fft, gcc_fft_amp),
                                       gcc_len))
    #
    gcc_phat = np.concatenate((gcc_phat_raw[-max_delay:],
                               gcc_phat_raw[:max_delay+1]))
    return gcc_phat


def cal_gcc_phat(x1, x2, win_f=np.hanning, max_delay=None,
                 frame_len=None, shift_len=None, snr_thd=-50):
    """Calculate general corss-correlation phase-transform
    Args:
        x1,x2: single-channel data
        win_f: window function, default to hanning
        max_delay: maximal delay in sample of ccf, if not specified, it will
                     be set to signale length. The relation between max_delay
                     and gcc_len: gcc_len=2*max_delay+1
        frame_len: frame length in sample, if not specified, frame_len is
                   set to be signal length
        shift_len: if not specified, set to frame_len/2
        snr_thd: allowed amp range,default to -50 dB
    Returns:
        gcc-phat with shape of [gcc_len] or [n_frame,gcc_len]
    """
    if frame_len is None:
        gcc_phat_result = _cal_gcc_phat(x1, x2, max_delay, win_f, snr_thd)
    else:
        if shift_len is None:
            shift_len = np.int16(frame_len/2)
        # signal length check
        if x1.shape[0] != x2.shape[0]:
            raise Exception('x1,x2 do not have the same length,\
                             x1:{}, x2:{}'.format(x1.shape[0], x2.shape[0]))
        frames_x1 = wav_tools.frame_data(x1, frame_len, shift_len)
        frames_x2 = wav_tools.frame_data(x2, frame_len, shift_len)
        n_frame = frames_x1.shape[0]
        gcc_phat_result = np.asarray([_cal_gcc_phat(frames_x1[frame_i],
                                                    frames_x2[frame_i],
                                                    max_delay, win_f,
                                                    snr_thd)
                                      for frame_i in range(n_frame)])
    return gcc_phat_result


def _cal_ccf(x1, x2, max_delay, win_f):
    """calculate cross-crrelation function in frequency domain
    Args:
        x1,x2: single channel signals
        max_delay: delay range, ccf_len = 2*max_delay+1
    Returns:
        cross-correlation function with shape of [ccf_len]
    """
    n_sample = np.max((x1.shape[0], x2.shape[0]))
    if max_delay is None:
        max_delay = n_sample-1
    # add hanning window before fft
    window = win_f(n_sample)
    ccf_len = 2*n_sample-1
    x1_fft = np.fft.fft(np.multiply(x1, window), ccf_len)
    x2_fft = np.fft.fft(np.multiply(x2, window), ccf_len)
    ccf_unshift = np.real(
                    np.fft.ifft(
                        np.multiply(x1_fft, np.conjugate(x2_fft))))
    ccf = np.concatenate([ccf_unshift[-max_delay:],
                          ccf_unshift[:max_delay+1]],
                         axis=0)
    return ccf


def cal_ccf(x1, x2, max_delay=None, frame_len=None, shift_len=None,
            win_f=np.ones):
    """Calculate cross-correlation function of whole signal or frames
    if frame_len is specified
    Args:
        x1: single-channel signal
        x2: single-channel signal
        max_delay: maximal delay in sample of ccf, if not specified, it will
                     be set to x_len-1. The relation between max_delay
                     and ccf_len: ccf_len=2*max_delay+1
        frame_len: frame length, if not specified, treat whole signal
                   as one frame
        shift_len: if not specified, set to frame_len/2
        win_f: window function, default to np.ones, rectangle windows
    Returns:
        corss-correlation function, with shape of
        - [ccf_len]: ccf of whole signal
        - [n_frame,ccf_len]: ccf of frames
    """
    if frame_len is None:
        ccf = _cal_ccf(x1, x2, max_delay, win_f)
    else:
        if shift_len is None:
            shift_len = np.int16(frame_len/2)
        # signal length check
        if x1.shape[0] != x2.shape[0]:
            raise Exception('x1,x2 do not have the same length,\
                             x1:{}, x2:{}'.format(x1.shape[0], x2.shape[0]))
        frames_x1 = wav_tools.frame_data(x1, frame_len, shift_len)
        frames_x2 = wav_tools.frame_data(x2, frame_len, shift_len)
        n_frame = frames_x1.shape[0]
        ccf = np.asarray([_cal_ccf(frames_x1[i], frames_x2[i],
                                   max_delay, win_f)
                          for i in range(n_frame)])
    return ccf


# ccfs = cal_ccf(wav[:,0],wav[:,1],frame_len=frame_len,max_delay=max_delay)
def test_plot(*pairs):

    n_Z = len(pairs)
    # print(z_min,z_max)

    # norm range for Z1 and Z2

    fig = plt.figure(figsize=[4*n_Z, 4])
    for i, ele in enumerate(pairs):
        z_min = np.min(ele[1])
        z_max = np.max(ele[1])
        ax = fig.add_subplot(1, n_Z, i+1)  # projection='3d')
        plot_tools.imshow(Z=(ele[1]-z_min)/(z_max-z_min),
                          ax=ax, vmin=0, vmax=1)
        ax.set_title(ele[0])
        ax.set_xlabel('Frame')
        ax.set_ylabel('Delay(sample)')

    plt.colorbar(ax.images[0])

    return fig


def test(frame_len=320, max_delay=18):
    """Test the effect of different window function, rect vs hanning
    snr_thd:default,-50
    """
    wav_path = 'resource/binaural_pos4.wav'
    wav, fs = wav_tools.wav_read(wav_path)

    gcc_phat_rect = cal_gcc_phat(wav[:, 0], wav[:, 1], frame_len=frame_len,
                                 win_f=np.ones, max_delay=max_delay)
    gcc_phat_hanning = cal_gcc_phat(wav[:, 0], wav[:, 1], frame_len=frame_len,
                                    win_f=np.hanning, max_delay=max_delay)

    fig = test_plot(['rect', gcc_phat_rect],
                    ['hanning', gcc_phat_hanning])
    plot_tools.savefig(fig, name='diff_window', dir='images/ccf')


def test_ccf():
    x = np.random.normal(0, 1, size=2048)
    # frame_len = 320
    # shift_len = 160
    x_len = x.shape[0]
    ccf_fft = cal_ccf(x, x, max_delay=44)
    print(ccf_fft.shape)
    ccf_ref = np.correlate(x, x, mode='full')[x_len-1-44:x_len+44]
    print(ccf_ref.shape)

    fig, ax = plt.subplots(1, 1)
    ax.plot(ccf_fft, label='ccf_fft')
    print(np.argmax(ccf_fft))

    ax.plot(ccf_ref, label='ccf_ref')
    print(np.argmax(ccf_ref))

    ax.legend()
    plt.show()
    plot_tools.savefig(fig, fig_name='ccf_compare.png',
                       fig_dir='../images/ccf')


if __name__ == '__main__':
    # wav_path = 'resource/binaural_pos4.wav'
    # frame_dur = 20e-3
    # max_delay = 18
    #
    # wav,fs = wav_tools.wav_read(wav_path)
    # frame_len = np.int16(frame_dur*fs)
    #
    # ccfs = cal_ccf(wav[:,0],wav[:,1],frame_len=frame_len,
    #                max_delay=max_delay,win_f=np.hanning)
    # fig,ax = plt.subplots(1,1)
    # ax.plot(ccfs.T)
    # ax.set_xlabel('Delay(sample)')
    # ax.set_title('ccfs')
    # plot_tools.savefig(fig,fig_name='ccf',fig_dir='images/ccf')

    test_ccf()
