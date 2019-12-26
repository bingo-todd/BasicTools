import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np

class Filter_GPU:
    def __init__(self,gpu_index):
        self._graph = tf.Graph()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = '{}'.format(gpu_index)
        self._sess = tf.compat.v1.Session(graph=self._graph,config=config)
        self._build_model()


    def _build_model(self):
        with self._graph.as_default():
            x = tf.compat.v1.placeholder(dtype=tf.float64,shape=(None,None,1))
            coef = tf.compat.v1.placeholder(dtype=tf.float64,shape=(None,1,1))
            coef_flip_pad = tf.pad(tf.reverse(coef,axis=[0]),
                                   paddings=[[0,tf.shape(coef)[0]-1],[0,0],[0,0]])
            y = tf.nn.convolution(input=x,filter=coef_flip_pad,padding='SAME')

            init = tf.compat.v1.global_variables_initializer()
            self._sess.run(init)

            self._x = x
            self._coef = coef
            self._y = y

    def filter(self,x,coef):
        x_shape = x.shape
        if len(x_shape) == 1:
            x.shape = [1,x_shape[0],1]
        y = self._sess.run(self._y,feed_dict={self._x:x,
                                              self._coef:coef[:,np.newaxis,
                                                              np.newaxis]})
        x.shape = x_shape
        return np.squeeze(y)


    def brir_filter(self,x,brir):
        if brir is None:
            return x.copy()
        y_l = self.filter(x,brir[:,0])
        y_r = self.filter(x,brir[:,1])
        return np.asarray((y_l,y_r)).T



def test_filter():
    import matplotlib.pyplot as plt
    import wav_tools
    import plot_tools
    import time
    import scipy.signal as dsp

    rir = np.load('resource/rir.npy')
    wav,fs = wav_tools.wav_read('resource/tar.wav')

    t_start = time.time()
    record_cpu = dsp.lfilter(rir,1,wav)
    t_elapsed_cpu = time.time() - t_start

    filter_gpu = Filter_GPU(0)
    t_start = time.time()
    record_gpu = filter_gpu.filter(wav,coef=rir)
    t_elapsed_gpu = time.time() - t_start

    fig,ax = plt.subplots(1,3,figsize=(8,3),sharex=True,sharey=True)
    ax[0].plot(record_cpu)
    ax[0].set_title(f'cpu {t_elapsed_cpu:.2f} s')

    ax[1].plot(record_gpu)
    ax[1].set_title(f'gpu {t_elapsed_gpu:.2f} s')

    ax[2].plot(record_cpu-record_gpu)
    ax[2].set_title('diff')
    plot_tools.savefig(fig,name='filter_cpu_gpu_diff',
                       dir='../images/Filter_GPU')

def test_brir_filter():
    import matplotlib.pyplot as plt
    import wav_tools
    import plot_tools
    import time

    brir = np.load('resource/brir.npy')
    wav,fs = wav_tools.wav_read('resource/tar.wav')

    t_start = time.time()
    record_cpu = wav_tools.brir_filter(wav,brir)
    t_elapsed_cpu = time.time() - t_start

    filter_gpu = Filter_GPU(gpu_index=0)
    t_start = time.time()
    record_gpu = filter_gpu.brir_filter(wav,brir)
    t_elapsed_gpu = time.time() - t_start

    fig,ax = plt.subplots(1,3,figsize=(8,3),sharex=True,sharey=True)
    ax[0].plot(record_cpu)
    ax[0].set_title(f'cpu {t_elapsed_cpu:.2f} s')

    ax[1].plot(record_gpu)
    ax[1].set_title(f'gpu {t_elapsed_gpu:.2f} s')

    ax[2].plot(record_cpu-record_gpu)
    ax[2].set_title('diff')
    plot_tools.savefig(fig,name='brir_filter_cpu_gpu_diff',
                       dir='../images/Filter_GPU')


if __name__ == '__main__':

    test_filter()

    test_brir_filter()
