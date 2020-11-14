import numpy as np
from . import wav_tools


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--tar', dest='tar_path', required=True,
            type=str, help='path of the input file')
    parser.add_argument('--chunksize', dest='chunksize',
            type=int, default=-1, help='chunksize')
    parser.add_argument('--result-path', dest='result_path',
            type=str, default=None, help='path of the output file')
    args = parser.parse_args()
    return args


def main():
    args
