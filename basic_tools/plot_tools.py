import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.image import NonUniformImage
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import datetime
from PIL import Image
import pathlib
from functools import wraps


def line_collector(plot_func):
    @wraps(plot_func)
    def wrapped_plot_func(ax,*args,line_container=None,**kwargs):
        lines = plot_func(ax,*args,**kwargs)
        if line_container is not None:
            line_container.extend(lines)
        return lines
    return wrapped_plot_func


@line_collector
def plot_line(ax,*args,line_container=None,**kwargs):
    return ax.plot(*args,**kwargs)


# @line_collector
# def plot_scatter(ax,*args,line_container=None,**kwargs):
#     return ax.scatter(*args,**kwargs)


@line_collector
def plot_contour(ax,*args,line_container=None,**kwargs):
    contour_set = ax.contour(*args,**kwargs)
    return contour_set.collections

def imshow(ax,Z,x=None,y=None,vmin=None,vmax=None,**kwargs):
    if x is None or y is None:
        y_len,x_len = Z.shape
        x = np.arange(x_len)
        y = np.arange(y_len)

    if vmin is None or vmax is None:
        vmin = np.min(Z)
        vmax = np.max(Z)
    Z_norm = (np.clip(Z,vmin,vmax)-vmin)/(vmax-vmin)

    basic_settings = {'cmap':cm.coolwarm,'interpolation':'bilinear'}
    basic_settings.update(kwargs)

    im = NonUniformImage(ax,**basic_settings,extent=(x[0],x[1],y[0],y[-1]))
    im.set_data(x,y,Z_norm)
    ax.images.append(im)
    ax.set_xlim([x[0],x[-1]])
    ax.set_ylim([y[0],y[-1]])

    return ax


def plot_surf(Z,X=None,Y=None,ax=None,**kwargs):
    m,n = Z.shape
    if X is None and Y is None:
        X,Y = np.meshgrid(np.arange(n),np.arange(m))

    basic_settings = {'cmap':cm.coolwarm}
    basic_settings.update(kwargs)

    if ax is None:
        fig,ax = plt.subplots(1,1,1,projection='3d')
    surf = ax.plot_surface(X,Y,Z,**basic_settings)
    return ax


def plot_bar(*mean_std,legend=None,**kwargs):
    """plot error-bar figure given mean and std values, also support
    matplotlib figure settings
    Args:
    Returns:
        matplotlib figure
    """
    n_set = len(mean_std)
    n_var = mean_std[0][0].shape[0]

    fig,ax = plt.subplots(1,1)
    bar_width = 0.8/n_set
    for i,[mean,std] in enumerate(mean_std):
        x = np.arange(n_var)+ bar_width*(i-np.floor(n_set/2))
        ax.bar(x,mean,yerr=std,width=bar_width)

    ax.set_xticks(range(n_var))
    if legend is not None:
        ax.legend(legend)

    ax.set(**kwargs)
    return fig


def break_plot():
    x = np.random.rand(10)
    x[0] = -100
    fig= plt.figure()
    ax1 = plt.subplot2grid((2,2),(0,1))
    ax1.plot(x)
    ax1.set_ylim((0,1))

    ax2 = plt.subplot2grid((2,2),(1,1))
    ax2.plot(x)
    ax2.set_ylim(-120,-80)

    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = .015  # how big to make the diagonal lines in ax coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom ax
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    savefig(fig,'break_axis')


def plot_wav_spec(wav_all,label_all=None,fs=None,frame_len=1024,
                  shift_len=512,yscale='mel'):
    """plot spectrogram of given len
    Args:
        wav_all: list of 1-channel signal
        label_all: labels of each wav
        fs: sample frequency
        frame_dur:
        yaxis_type: options 'mel'
    """
    import fft
    from auditory_scale import erb
    from auditory_scale import mel

    if  isinstance(wav_all,np.ndarray):
        wav_all = [wav_all]

    n_wav = len(wav_all)
    if label_all is None:
        label_all = ['']*n_wav


    amp_max_overall = np.max([np.max(np.abs(wav)) for wav in wav_all])

    stft_params = {'frame_len':frame_len,'shift_len':shift_len,'fs':fs}
    stft_amp_dB_all =  [20*np.log10(np.abs(fft.cal_stft(wav,**stft_params)[0]))
                                                            for wav in wav_all]
    stft_amp_max_overall = np.max([np.max(stft_amp_dB)
                                    for stft_amp_dB in stft_amp_dB_all])
    stft_amp_min_overall = stft_amp_max_overall-60

    fig,ax = plt.subplots(2,n_wav,figsize=[4*n_wav,6])
    if fs is None:
        t_label = 'sample(n)'
        freq_label = 'normalizeed frequnecy'
    else:
        t_label = 'time(s)'
        freq_label = 'frequnecy(kHz)'

    for wav_i,[wav,wav_name] in enumerate(zip(wav_all,label_all)):
        n_frame,n_bin = stft_amp_dB_all[wav_i].shape
        if fs is None:
            t_tick_wav = np.arange(wav.shape[0])
            t_tick_stft = np.arange(n_frame)
            freq_tick = np.arange(n_bin)/n_bin
        else:
            t_tick_wav = np.arange(wav.shape[0])/fs
            t_tick_stft = (np.arange(n_frame)+1)*shift_len/fs
            freq_tick = np.arange(n_bin)/frame_len*fs

        ax[0,wav_i].plot(t_tick_wav,wav)
        ax[0,wav_i].set_ylim((-amp_max_overall,amp_max_overall))
        ax[0,wav_i].set_title(wav_name)

        imshow(ax[1,wav_i],Z=stft_amp_dB_all[wav_i].T,
                  x=t_tick_stft,y=freq_tick,
                  vmin=stft_amp_min_overall,vmax=stft_amp_max_overall,
                  interpolation='nearest',origin='lower')

        ax[1,wav_i].set_xlabel(t_label)
        if wav_i == 0:
            ax[1,wav_i].set_ylabel('freq_label')

    plt.tight_layout()
    return fig


class Gif:
    # Gif-making class, a encapsulation of matplotlib functions
    def __init__(self):
        self.artists = [] # list of objects of line, image ...
                          # (return of plot)

    def add(self,artist):
        self.artists.append(artist)


    def save(self,fpath,fig,fps=60):
        """save to gif file
        Args:
            fpath: file path of gif
            fig: figure obj that hold artist on
            fps: frame per second
        Returns:
            None
        """
        ani = animation.ArtistAnimation(fig,self.artists,interval=1./fps*1e3)
        writer = animation.FFMpegWriter(fps=fps)
        ani.save(fpath,fps=fps,writer='pillow')


def savefig(fig,name=None,dir='./images'):

    if not os.path.exists(dir):
        os.makedirs(dir)

    # use date as name if name is not defined
    if name is None:
        name = '{0.year}_{0.month}_{0.day}.png'.format(datetime.date.today())

    matplotlib_fig_suffixs = ['.eps', '.pdf', '.pgf', '.png', '.ps', '.raw',
                             '.rgba', '.svg', '.svgz']

    # check whether name has suffix
    stem = pathlib.PurePath(name).stem
    suffix = pathlib.PurePath(name).suffix
    if suffix == '':
        suffix = '.png'

    if suffix == '.jpg': # not support in matplotlib
        fig_path = os.path.join(dir,''.join((stem,'.png')))
        fig.savefig(fig_path)
        Image.open(fig_path).convert('RGB').save(os.path.join(dir,
                                                              ''.join((stem,
                                                                      '.jpg'))))
        os.remove(fig_path)
    else:
        fig_path = os.path.join(dir,''.join((stem,suffix)))
        fig.savefig(fig_path)

    if True:
        print('{}{} is saved in {}'.format(stem,suffix,dir))



""" Test """

def test_bar():
    mean_std_all = [[np.random.normal(size=5),np.random.rand(5)]
                        for i in range(5)]
    fig = plot_bar(*mean_std_all,ylabel='ylabel',
                   xticklabels=['label{}'.format(i) for i in range(4)])
    savefig(fig,name='bar.png',dir='images/plot_tools/')


def test_gif():
    gif = Gif()
    line_container = []
    fig,ax = plt.subplots(1,1)
    for i in range(5):
        # lines = []
        for line_i in range(np.random.randint(1,4)):
            line_tmp = plot_line(ax,np.random.rand(10),line_container=line_container)
            # lines.extend(line_tmp)
        gif.add(line_container)
    gif.save('images/plot_tools/gif_example.gif',fig,fps=10)


def test_imshow():
    import wav_tools
    import fft
    from auditory_scale import mel

    x,fs = wav_tools.wav_read('resource/tar.wav')
    stft,t,freq = fft.cal_stft(x,fs=fs,frame_len=np.int(fs*50e-3))
    ax = imshow(x=t,y=freq,Z=20*np.log10(np.abs(stft)))
    ax.set_yscale('mel')
    fig = ax.get_figure()
    savefig(fig,name='imshow',dir='images/plot_tools')


def test_plot_wav_spec():
    import wav_tools
    x1,fs = wav_tools.wav_read('resource/tar.wav')
    x2,fs = wav_tools.wav_read('resource/inter.wav')

    fig = plot_wav_spec(wav_all=[x1,x2],label_all=['tar','inter'],fs=fs,
                        frame_len=1024,shift_len=512,yscale='mel')
    savefig(fig,name='wav_spec',dir='./images/plot_tools')


if __name__ == "__main__":

    # test_gif()
    # test_bar()
    # test_imshow()
    # break_plot()

    test_plot_wav_spec()
