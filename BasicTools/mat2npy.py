import os
import scipy.io as sio
import h5py
import numpy as np
import argparse


def mat2npy(mat_path, npy_path):

    if os.path.exists(npy_path):
        raise Exception(f'{npy_path} already exists')

    try:
        mat_obj = sio.loadmat(mat_path)
        raise Exception('not finished')
    except Exception:
        mat_obj = h5py.File(mat_path, mode='r')
        keys = list(mat_obj.keys())
        if len(keys) > 1:
            raise Exception(f'multiple variales in {mat_path}')
        data = np.asarray(mat_obj[keys[0]]).transpose()
    np.save(npy_path, data)


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--mat-path', dest='mat_path', required=True, type=str,
                        help='')
    parser.add_argument('--npy-path', dest='npy_path', required=True, type=str,
                        help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    mat2npy(mat_path=args.mat_path,
            npy_path=args.npy_path)


if __name__ == '__main__':
    main()
