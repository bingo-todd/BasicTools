import tensorflow as tf
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
            coef_flip_pad = tf.pad(tf.reverse(coef,axis=[0]),paddings=[[0,tf.shape(coef)[0]-1],[0,0],[0,0]])
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


if __name__ == '__main__':
    import scipy.signal as dsp
    import matplotlib.pyplot as plt
    import time

    x = np.random.normal(size=2048)
    coef = np.random.normal(size=512)

    t1 = time.time()
    filter_gpu_obj = filter_gpu(gpu_index=0)
    t2 = time.time()
    y_gpu = filter_gpu_obj.filter(x=x,coef=coef)
    t3 = time.time()
    y_scipy = dsp.lfilter(coef,1,x)
    t4 = time.time()

    t_used_gpu_init = t2-t1
    t_used_gpu_filter = t3-t2
    t_used_scipy = t4-t3
    print('time used\n')
    print('\t scipy:{}\n'.format(t_used_scipy))
    print('\t gpu:\n\t init:{} filter:{}'.format(t_used_gpu_init,
                                                 t_used_gpu_filter))
    fig,ax = plt.subplots(1,3,figsize=[8,3],dpi=200)
    ax[0].plot(y_scipy,label='scipy')
    ax[0].plot(y_gpu,label='gpu')
    ax[0].legend()

    xcorr = np.correlate(y_scipy,y_gpu,mode='full')
    print('ccf max pos {}',format(np.argmax(xcorr)))
    ax[1].plot(xcorr)
    ax[1].set_xlim(2047-10,2047+10)
    ax[1].plot(2047,xcorr[2047],'rx')
    ax[1].set_title('CCF')

    ax[2].plot(y_gpu-y_scipy)
    ax[2].set_title('diff')
    plt.tight_layout()
    fig.savefig('images/filter_gpu.png')
