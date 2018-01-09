#!/usr/bin/env python3
import os
import argparse
import lzma
import numpy as np
from model import create_model
from keras.optimizers import Adam

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
    arg_parser.add_argument('SAMPLING_RATE',
                            type=int,
                            help='')
    arg_parser.add_argument('TIME_LENGTH',
                            type=float,
                            help='')
    arg_parser.add_argument('MODEL_FILE',
                            help='')
    arg_parser.add_argument('NOISY_DATA_FILE',
                            help='')
    arg_parser.add_argument('CLEAN_DATA_FILE',
                            help='')

    args = arg_parser.parse_args()

    # parse arguments
    time_steps = args.SAMPLING_RATE * args.TIME_LENGTH

    # load data
    arrayx = np.memmap(args.NOISY_AUDIO_FILE, dtype='float32', mode='c')
    arrayx = arrayx.reshape(arrayx.shape[0] + (1,))

    arrayy = np.memmap(args.CLEAN_DATA_FILE, dtype='float32', mode='c')
    arrayy = arrayy.reshape(arrayx.shape[0] + (1,))

    # load or create model
    model = create_model(time_steps)

    if os.path.exists(args.MODEL_FILE):
        model.load_weights(args.MODEL_FILE)
    else:
        model.save_weights(args.MODEL_FILE)

    model.compile(
        loss='mse',
        optimizer=Adam(lr=args.learning_rate)
    )

    # train model
    model.fit(
        arrayx,
        arrayy,
        batch_size=args.batch_size
    )

    # save model
    model.save_weights(args.MODEL_FILE)

if __name__ == '__main__':
    main()
