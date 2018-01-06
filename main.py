#!/usr/bin/env python3
import argparse

from model import create_model

def main():
    # parse arguments
    arg_parser = argparse.ArgumentParser(description='The script is intended for model creation and training')
    arg_parser.add_argument('--plot-model',
                            metavar='IMAGE_FILE',
                            help='Create image plot of this model to this file')
    arg_parser.add_argument('--epochs',
                            type=int,
                            default=10,
                            help='Number of training steps')
    arg_parser.add_argument('--batch-size',
                            type=int,
                            default=64,
                            help='Batch size for training phase')
    arg_parser.add_argument('--learning-rate',
                            type=float,
                            default=0.001,
                            help='Learning rate for training phase')
    args = arg_parser.parse_args()

    model = create_model(1024)

if __name__ == '__main__':
    main()
