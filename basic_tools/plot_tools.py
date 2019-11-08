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

def imshow(Z,x=None,y=None,ax=None,vmin=None,vmax=None,**kwargs):
    if x is None or y is None:
        x_len,y_len = Z.shape
        x = np.arange(x_len)
        y = np.arange(y_len)
    if ax is None:
        fig,ax = plt.subplots(1,1)

    if vmin is None or vmax is None:
        vmin = np.min(Z)
        vmax = np.max(Z)
    Z = (Z-vmin)/(vmax-vmin)

    basic_settings = {'cmap':cm.coolwarm,'interpolation':'bilinear'}
    basic_settings.update(kwargs)

    im = NonUniformImage(ax,**basic_settings,extent=(x[0],x[1],y[0],y[-1]))
    im.set_data(x,y,Z.T)
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


if __name__ == "__main__":

    test_gif()
    # test_bar()
    # test_imshow()
