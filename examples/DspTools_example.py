import matplotlib.pyplot as plt
import numpy as np
from BasicTools import DspTools
from BasicTools import wav_tools
from BasicTools import plot_tools


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

    gcc_phat_rect = DspTools.cal_gcc_phat(wav[:, 0], wav[:, 1],
                                          frame_len=frame_len,
                                          win_f=np.ones, max_delay=max_delay)
    gcc_phat_hanning = DspTools.cal_gcc_phat(wav[:, 0], wav[:, 1],
                                             frame_len=frame_len,
                                             win_f=np.hanning,
                                             max_delay=max_delay)

    fig = test_plot(['rect', gcc_phat_rect],
                    ['hanning', gcc_phat_hanning])
    plot_tools.savefig(fig, name='diff_window', dir='images/ccf')


def test_ccf():
    x = np.random.normal(0, 1, size=2048)
    # frame_len = 320
    # shift_len = 160
    x_len = x.shape[0]
    ccf_fft = DspTools.cal_ccf(x, x, max_delay=44)
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
    x, fs = wav_tools.read_wav('data/binaural_1.wav')
    x1 = x[100:,0]
    x2 = x[:,0]
    delay = DspTools.cal_delay(x1, x2)
    print(delay)

    ccf = DspTools.cal_ccf(x1, x2)
    plt.plot(ccf)
    print(ccf.shape, x1.shape, x2.shape)
    plt.show()
    plt.savefig('test.png')

