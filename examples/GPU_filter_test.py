import matplotlib.pyplot as plt
import time
import numpy as np
from BasicTools.GPU_Filter import GPU_Filter
from BasicTools import wav_tools, plot_tools


def test_brir_filter():
    brir = np.load('data/brir.npy')
    wav, fs = wav_tools.read('data/tar.wav')

    t_start = time.time()
    record_cpu = wav_tools.brir_filter(wav, brir)
    t_elapsed_cpu = time.time() - t_start
    print(f'cpu: t {t_elapsed_cpu:.4f} s')

    filter_gpu = GPU_Filter(gpu_id=0)
    t_start = time.time()
    record_gpu = filter_gpu.brir_filter(wav, brir)
    t_elapsed_gpu = time.time() - t_start
    print(f'gpu: t {t_elapsed_gpu:.4f} s')

    fig, ax = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
    ax[0].plot(record_cpu)
    ax[0].set_title(f'cpu {t_elapsed_cpu:.4f} s')

    ax[1].plot(record_gpu)
    ax[1].set_title(f'gpu {t_elapsed_gpu:.4f} s')

    ax[2].plot(record_cpu-record_gpu)
    ax[2].set_title('diff')
    plot_tools.savefig(fig, fig_name='brir_filter_cpu_gpu_diff',
                       fig_dir='images/GPU_Filter')


if __name__ == '__main__':
    test_brir_filter()
