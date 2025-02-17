import numpy as np
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type = str, required = True)
    parser.add_argument('--hidden-size', type = int, default = 16)
    parser.add_argument('--epochs', type = int, default = 200)
    parser.add_argument('--lr', type = float, default = 1e-3)
    args = parser.parse_args()

    data_dir = args.data_dir
    