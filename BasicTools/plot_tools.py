import matplotlib as mlp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import numpy as np
import os
import datetime
from PIL import Image
import pathlib
from functools import wraps
from .scale import mel, erb

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['figure.dpi'] = 200


def line_collector(plot_func):
    @wraps(plot_func)
    def wrapped_plot_func(ax, *args, line_container=None, **kwargs):
        result = plot_func(ax, *args, **kwargs)
        if line_container is not None:
            if plot_func.__name__ == 'plot_contour':
                line_container.extend(result.collections)
                line_container.extend(ax.clabel(result, inline=True))
            else:
                line_container.extend(result)
        return result
    return wrapped_plot_func


@line_collector
def plot_line(ax, *args, line_container=None, **kwargs):
    return ax.plot(*args, **kwargs)


def plot_line2(ax, y1, y2, x1=None, x2=None, **kwargs):
    ax_twin = ax.twinx()
    if x1 is None:
        x1 = np.arange(y1.shape[0])
    ax.plot(x1, y1, **kwargs)

    if x2 is None:
        x2 = np.arange(y2.shape[0])
    ax_twin.plot(x1, y1, **kwargs)


# @line_collector
# def plot_scatter(ax,*args,line_container=None,**kwargs):
#     return ax.scatter(*args,**kwargs)


@line_collector
def plot_contour(ax, *args, is_label=False, line_container=None, **kwargs):
    contour_set = ax.contour(*args, **kwargs)
    return contour_set


def imshow(Z, ax=None, x_lim=None, y_lim=None, vmin=None, vmax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = None

    if x_lim is None or y_lim is None:
        x_lim = [0, Z.shape[1]]
        y_lim = [0, Z.shape[0]]

    if vmin is None or vmax is None:
        vmin = np.min(Z)
        vmax = np.max(Z)
    Z_norm = (np.clip(Z, vmin, vmax)-vmin)/(vmax-vmin)

    basic_settings = {'cmap': cm.jet,
                      'aspect': 'auto'}
    basic_settings.update(kwargs)

    ax.imshow(Z_norm, extent=[*x_lim, *y_lim], **basic_settings)
    return fig, ax


def plot_matrix(matrix, ax=None, fig=None, x=None, y=None, xlabel=None,
                ylabel=None, show_value=False, normalize=True, vmin=None,
                vmax=None, aspect='auto', cmap=plt.cm.coolwarm):
    """
    This function prints and plots matrix.
    Normalization can be applied by setting `normalize=True`.
    Args
        X: matrix
        xlabel, ylabel: labels of x-axis and y-axis
        show_value: display the correponding values of each square of images
        normalize: normalize Z to the range of [0, 1]
        vmin, vmax: the min- and max values to clip Z
        cmap: color map
    - normalize: whether normalization
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, tight_layout=True)

    if x is None:
        x_min, x_max = 0, matrix.shape[1]
    else:
        x_min, x_max = x[0], x[-1]
    if y is None:
        y_min, y_max = 0, matrix.shape[0]
    else:
        y_min, y_max = y[0], y[-1]

    im = ax.imshow(matrix, interpolation='nearest', cmap=cmap,
                   vmin=vmin, vmax=vmax, extent=[x_min, x_max, y_min, y_max],
                   aspect=aspect, origin='lower')

    if fig is not None:
        plt.colorbar(im, shrink=0.6)

    # x_axis: colum  y_axis: row
    if show_value:
        fmt = '.2f' if normalize else 'd'
        thresh = matrix.max() / 2.
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(j, i, format(matrix[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if matrix[i, j] > thresh else "black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, ax


def plot_surf(Z, x=None, y=None, ax=None, xlabel=None, ylabel=None,
              zlabel=None, zlim=None, vmin=None, vmax=None, cmap_range=None,
              **kwargs):
    m, n = Z.shape
    if x is None and y is None:
        X, Y = np.meshgrid(np.arange(n), np.arange(m))
    else:
        x, y = np.asarray(x), np.asarray(y)
        if len(np.squeeze(x).shape) < 2:
            X, Y = np.meshgrid(x, y)
        else:
            X, Y = x, y

    if cmap_range is not None:
        norm = mlp.colors.Normalize(vmin=cmap_range[0], vmax=cmap_range[1])
    else:
        norm = None
    basic_settings = {'cmap': cm.coolwarm,
                      'vmin': vmin,
                      'vmax': vmax,
                      'norm': norm}
    basic_settings.update(kwargs)

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'},
                               tight_layout=True)
    else:
        fig = None
    surf = ax.plot_surface(X, Y, Z, **basic_settings)
    ax.set_zlim(zlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if fig is not None:
        fig.colorbar(surf, shrink=0.6)
    return fig, ax


def plot_bar(*mean_std, legend=None, **kwargs):
    """plot error-bar figure given mean and std values, also support
    matplotlib figure settings
    Args:
    Returns:
        matplotlib figure
    """
    n_set = len(mean_std)
    n_var = mean_std[0][0].shape[0]

    fig, ax = plt.subplots(1, 1)
    bar_width = 0.8/n_set
    for i, [mean, std] in enumerate(mean_std):
        x = np.arange(n_var) + bar_width*(i-np.floor(n_set/2))
        ax.bar(x, mean, yerr=std, width=bar_width)

    ax.set_xticks(range(n_var))
    if legend is not None:
        ax.legend(legend)

    ax.set(**kwargs)
    return fig


def break_plot():
    x = np.random.rand(10)
    x[0] = -100
    fig = plt.figure()
    ax1 = plt.subplot2grid((2, 2), (0, 1))
    ax1.plot(x)
    ax1.set_ylim((0, 1))

    ax2 = plt.subplot2grid((2, 2), (1, 1))
    ax2.plot(x)
    ax2.set_ylim(-120, -80)

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

    savefig(fig, 'break_axis')


def plot_wav(wav, fs=None, label=None, ax_wav=None, plot_spec=False,
             ax_specgram=None, frame_len=1024, frame_shift=512, yscale='mel',
             amp_max=None, **plot_params):
    """plot spectrogram of given len
    Args:
    """
    from . import fft

    wav_len = wav.shape[0]
    n_bin = int(frame_len/2)

    if fs is None:
        fs = 1
        t_label = 'sample(n)'
        freq_label = 'normalizeed frequnecy'
    else:
        if wav_len < fs*0.05:
            t_scale = 1000
            t_label = 'time(ms)'
        else:
            t_scale = 1
            t_label = 'time(s)'
        freq_label = 'frequnecy(Hz)'

    fig = None
    # if ax_wav and ax_specgram are not specified
    if plot_spec:
        if ax_wav is None and ax_specgram is None:
            fig, ax = plt.subplots(2, 1)
            ax_wav, ax_specgram = ax
    else:
        if ax_wav is None:
            fig, ax_wav = plt.subplots(1, 1)
            print('here')

    if ax_wav is not None:
        t = np.arange(wav_len)/fs*t_scale
        ax_wav.plot(t, wav, label=label, **plot_params)
        ax_wav.set_xlabel(t_label)
        if amp_max is not None:
            ax_wav.set_ylim((-amp_max, amp_max))
        ax_wav.set_title(label)

    if ax_specgram is not None:
        specgram, t, freqs = fft.cal_stft(wav, frame_len=frame_len,
                                          frame_shift=frame_shift, fs=fs)
        specgram_amp = 20*np.log10(np.abs(specgram))
        max_value = np.max(specgram_amp)
        min_value = max_value-60
        n_frame, n_bin = specgram_amp.shape
        imshow(ax=ax_specgram, Z=specgram_amp.T,
               x_lim=[0, t[-1]], y_lim=[0, freqs[-1]],
               vmin=min_value, vmax=max_value, origin='lower')
        ax_specgram.set_yscale('mel')
        ax_specgram.set_xlabel(t_label)
        ax_specgram.set_ylabel(freq_label)
        # ax_specgram.yaxis.set_major_formatter('{x:.1f}')
        #
    return fig, ax_wav, ax_specgram


def plot_break_axis(x1, x2):
    # how big to make the diagonal lines in axes coordinates
    d = .015
    fig, ax = plt.subplots(2, 1, sharex=True)
    [x1_min, x1_max] = [np.min(x1), np.max(x1)]
    if x1_min == x1_max:
        tmp = x1_max/10.
    else:
        tmp = (x1_max-x1_min)/10
    ax[0].plot(x1)
    ax[0].set_ylim((x1_min-tmp, x1_max+tmp))

    [x2_min, x2_max] = [np.min(x2), np.max(x2)]
    if x2_min == x2_max:
        tmp = x2_max/10.
    else:
        tmp = (x2_max-x2_min)/10
    ax[1].plot(x2)
    ax[1].set_ylim((x2_min-tmp, x2_max+tmp))

    # set spines
    ax[0].spines['bottom'].set_visible(False)
    ax[1].spines['top'].set_visible(False)

    #
    ax[0].xaxis.tick_top()
    ax[1].xaxis.tick_bottom()

    ax[1].tick_params(labeltop=False)

    # draw diagonal marks where axises break
    kwargs = dict(transform=ax[0].transAxes, color='k', clip_on=False)
    ax[0].plot((-d, +d), (-d, +d), **kwargs)
    ax[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax[1].transAxes)
    ax[1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax[1].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    return fig


class GIF:
    # GIF-making class, a encapsulation of matplotlib functions
    def __init__(self):
        self.artists = []  # list of objects of line, image ...

    def add(self, artist):
        self.artists.append(artist)

    def save(self, fig_path, fig, fps=60):
        """save to gif file
        Args:
            fpath: file path of gif
            fig: figure obj that hold artist on
            fps: frame per second
        Returns:
            None
        """
        ani = animation.ArtistAnimation(fig, self.artists, interval=1./fps*1e3)
        # writer = animation.FFMpegWriter(fps=fps)
        ani.save(fig_path, fps=fps, writer='pillow')


def savefig(fig, fig_name=None, fig_dir='./images'):

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # use date as name if name is not defined
    if fig_name is None:
        fig_name = '{0.year}_{0.month}_{0.day}.png'.format(
                                            datetime.date.today())

    # matplotlib_fig_suffixs = ['.eps', '.pdf', '.pgf', '.png', '.ps', '.raw',
    #                           '.rgba', '.svg', '.svgz']

    # check whether name has suffix
    stem = pathlib.PurePath(fig_name).stem
    suffix = pathlib.PurePath(fig_name).suffix
    if suffix == '':
        suffix = '.png'

    if suffix == '.jpg':  # not support in matplotlib
        fig_path = os.path.join(dir, ''.join((stem, '.png')))
        fig.savefig(fig_path)
        Image.open(fig_path).convert('RGB').save(
                                os.path.join(fig_dir, ''.join((stem, '.jpg'))))
        os.remove(fig_path)
    else:
        fig_path = os.path.join(fig_dir, ''.join((stem, suffix)))
        fig.savefig(fig_path)

    if True:
        print('{}{} is saved in {}'.format(stem, suffix, fig_dir))


def test_bar():
    mean_std_all = [[np.random.normal(size=5), np.random.rand(5)]
                    for i in range(5)]
    fig = plot_bar(*mean_std_all, ylabel='ylabel',
                   xticklabels=['label{}'.format(i) for i in range(4)])
    savefig(fig, name='bar.png', dir='images/plot_tools/')


def test_gif():
    gif = GIF()
    line_container = []
    fig, ax = plt.subplots(1, 1)
    for i in range(5):
        # lines = []
        for line_i in range(np.random.randint(1, 4)):
            plot_line(ax, np.random.rand(10),
                      line_container=line_container)
            # lines.extend(line_tmp)
        gif.add(line_container)
    gif.save('images/plot_tools/gif_example.gif', fig, fps=10)


def test_imshow():
    import wav_tools
    import fft
    # from auditory_scale import mel

    x, fs = wav_tools.wav_read('resource/tar.wav')
    stft, t, freq = fft.cal_stft(x, fs=fs, frame_len=np.int(fs*50e-3))
    ax = imshow(x=t, y=freq, Z=20*np.log10(np.abs(stft)))
    ax.set_yscale('mel')
    fig = ax.get_figure()
    savefig(fig, name='imshow', dir='images/plot_tools')


def test_plot_wav_spec():
    import wav_tools
    x1, fs = wav_tools.wav_read('resource/tar.wav')
    x2, fs = wav_tools.wav_read('resource/inter.wav')

    fig = plot_wav(wav_all=[x1, x2], label_all=['tar', 'inter'], fs=fs,
                   frame_len=1024, shift_len=512, yscale='mel')
    savefig(fig, name='wav_spec', dir='./images/plot_tools')


def test_plot_line2():
    y1 = np.random.rand(10)
    y2 = np.random.rand(10)+10
    fig, ax = plt.subplots(1, 1)
    plot_line2(ax, y1, y2)
    savefig(fig, name='plot_line2', dir='images/plot_tools')


if __name__ == "__main__":

    # test_gif()
    # test_bar()
    # test_imshow()
    # break_plot()
    # test_plot_wav_spec()
    test_plot_line2()
