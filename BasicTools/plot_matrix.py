import numpy as np
import matplotlib.pyplot as plt
import argparse

from .plot_tools import plot_matrix
from .plot_tools import plot_surf


def load_matrix(matrix_path):
    suffix = matrix_path.split('.')[-1]
    if suffix == 'npy':
        matrix = np.load(matrix_path)
    elif suffix == 'txt':
        with open(matrix_path, 'r') as file_obj:
            lines = file_obj.readlines()
        matrix = [[float(item) for item in line.strip().split()]
                  for line in lines if len(line.strip()) > 0]
        matrix = np.asarray(matrix)
    elif suffix == 'dat':
        with open(matrix_path, 'rb') as file_obj:
            lines = file_obj.readlines()
        matrix = [[float(item) for item in line.strip().split()]
                  for line in lines if len(line.strip()) > 0]
        matrix = np.asarray(matrix)

    return matrix


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--matrix-path', dest='matrix_path', type=str,
                        required=True, help='npy, txt, dat')
    parser.add_argument('--type', dest='type', type=str,
                        choices=['image', 'surf'], default='image',
                        help='')
    parser.add_argument('--aspect', dest='aspect', type=str,
                        choices=['equal', 'auto'], default='auto',
                        help='')
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
    parser.add_argument('--interactive', dest='interactive', type=str,
                        choices=['true', 'false'], default='false',
                        help='')
    parser.add_argument('--fig-path', dest='fig_path', type=str, default=None,
                        help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    matrix = load_matrix(args.matrix_path)

    if args.type == 'image':
        fig, ax = plot_matrix(matrix,
                              xlabel=args.xlabel,
                              ylabel=args.ylabel,
                              show_value=args.show_value == 'true',
                              normalize=args.normalize == 'true',
                              vmin=args.vmin,
                              vmax=args.vmax,
                              aspect=args.aspect)
    else:
        fig, ax = plot_surf(Z=matrix,
                            vmin=args.vmin,
                            vmax=args.vmax,
                            xlabel=args.xlabel,
                            ylabel=args.ylabel)

    if args.fig_path is not None:
        fig.savefig(args.fig_path)

    if args.interactive == 'true':
        plt.show()


if __name__ == '__main__':
    main()
