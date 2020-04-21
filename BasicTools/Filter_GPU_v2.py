import os
import numpy as np
import tensorflow as tf


class Filter_GPU:
    def __init__(self):
        return

    def filter(self, x, fcoef):
        if len(x.shape) > 2:
            raise Exception('only 1d array is supported')
        fcoef_conv = np.concatenate((np.flip(fcoef),
                                     np.zeros(fcoef.shape[0])))
        y = tf.nn.conv1d(x[np.newaxis, :, np.newaxis],
                         fcoef_conv[:, np.newaxis, np.newaxis],
                         stride=1,
                         padding='SAME')
        return np.squeeze(np.asarray(y))

    def brir_filter(self, x, brir):
        if brir is None:
            return x.copy()
        y_l = self.filter(x, brir[:, 0])
        y_r = self.filter(x, brir[:, 1])
        return np.asarray((y_l, y_r)).T


def test_filter():
    import matplotlib.pyplot as plt
    import wav_tools
    import plot_tools
    import time
    import scipy.signal as dsp

    rir = np.load('resource/rir.npy')
    wav, fs = wav_tools.wav_read('resource/tar.wav')

    t_start = time.time()
    record_cpu = dsp.lfilter(rir, 1, wav)
    t_elapsed_cpu = time.time() - t_start

    filter_gpu = Filter_GPU(0)
    t_start = time.time()
    record_gpu = filter_gpu.filter(wav, coef=rir)
    t_elapsed_gpu = time.time() - t_start

    fig, ax = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
    ax[0].plot(record_cpu)
    ax[0].set_title(f'cpu {t_elapsed_cpu:.2f} s')

    ax[1].plot(record_gpu)
    ax[1].set_title(f'gpu {t_elapsed_gpu:.2f} s')

    ax[2].plot(record_cpu-record_gpu)
    ax[2].set_title('diff')
    plot_tools.savefig(fig, name='filter_cpu_gpu_diff',
                       dir='../images/Filter_GPU')


def test_brir_filter():
    import matplotlib.pyplot as plt
    import wav_tools
    import plot_tools
    import time

    brir = np.load('resource/brir.npy')
    wav, fs = wav_tools.wav_read('resource/tar.wav')

    t_start = time.time()
    record_cpu = wav_tools.brir_filter(wav, brir)
    t_elapsed_cpu = time.time() - t_start

    filter_gpu = Filter_GPU(gpu_index=0)
    t_start = time.time()
    record_gpu = filter_gpu.brir_filter(wav, brir)
    t_elapsed_gpu = time.time() - t_start

    fig, ax = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
    ax[0].plot(record_cpu)
    ax[0].set_title(f'cpu {t_elapsed_cpu:.2f} s')

    ax[1].plot(record_gpu)
    ax[1].set_title(f'gpu {t_elapsed_gpu:.2f} s')

    ax[2].plot(record_cpu-record_gpu)
    ax[2].set_title('diff')
    plot_tools.savefig(fig, name='brir_filter_cpu_gpu_diff',
                       dir='../images/Filter_GPU')


if __name__ == '__main__':

    test_filter()

    test_brir_filter()
