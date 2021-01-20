import numpy as np
import itertools
import matplotlib.pyplot as plt
import argparse


def plot_matrix(Z, xlabel=None, ylabel=None, show_value=False, normalize=True,
                vmin=None, vmax=None, cmap=plt.cm.coolwarm):
    """
    This function prints and plots the confusion matrix.
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

    if vmin is None:
        vmin = np.min(Z)
    if vmax is None:
        vmax = np.max(Z)

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    plt.imshow(Z, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(shrink=0.6)

    # x_axis: colum  y_axis: row
    if show_value:
        fmt = '.2f' if normalize else 'd'
        thresh = Z.max() / 2.
        for i, j in itertools.product(range(Z.shape[0]), range(Z.shape[1])):
            plt.text(j, i, format(Z[i, j], fmt), horizontalalignment="center",
                     color="white" if Z[i, j] > thresh else "black")
        plt.ylabel(xlabel)
        plt.xlabel(ylabel)

    tick_labels = list(map(str, range(Z.shape[0])))
    tick_marks = np.arange(Z.shape[0])
    plt.yticks(tick_marks, tick_labels)  # rotation=45)

    tick_labels = list(map(str, range(Z.shape[1])))
    tick_marks = np.arange(Z.shape[1])
    plt.xticks(tick_marks, tick_labels)

    return fig, ax


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--npy-path', dest='npy_path', type=str, required=True,
                        help='path of the input file')
    parser.add_argument('--vmin', dest='vmin', type=float, default=None,
                        help='')
    parser.add_argument('--vmax', dest='vmax', type=float, default=None,
                        help='')
    parser.add_argument('--xlabel', dest='xlabel', type=str, default=None,
                        help='')
    parser.add_argument('--ylabel', dest='ylabel', type=str, default=None,
                        help='')
    parser.add_argument('--show-value', dest='show_value', type=str,
                        choices=['true', 'false'], default='false',
                        help='')
    parser.add_argument('--normlize', dest='normalize', type=str,
                        choices=['true', 'false'], default='false',
                        help='')
    parser.add_argument('--fig-path', dest='fig_path', type=str, required=True,
                        help='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    Z = np.load(args.npy_path)

    fig, ax = plot_matrix(Z,
                          xlabel=args.xlabel,
                          ylabel=args.ylabel,
                          show_value=args.show_value == 'true',
                          normalize=args.normalize == 'true',
                          vmin=args.vmin,
                          vmax=args.vmax)

    fig.savefig(args.fig_path)
