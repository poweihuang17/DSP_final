#!/usr/bin/env python3
import os
import argparse
import lzma
import numpy as np
from model import create_model
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.utils.vis_utils import plot_model

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def main():
    # parse arguments
    arg_parser = argparse.ArgumentParser(description='The script is intended for model creation and training')
    arg_parser.add_argument('--plot-model',
                            metavar='IMAGE_FILE',
                            help='Create image plot of this model to this file')
    arg_parser.add_argument('--epochs',
                            type=int,
                            default=1,
                            help='Number of training epochs')
    arg_parser.add_argument('--batch-size',
                            type=int,
                            default=10,
                            help='Batch size for training phase')
    arg_parser.add_argument('--learning-rate',
                            type=float,
                            default=0.001,
                            help='Learning rate for training phase')
    arg_parser.add_argument('SAMPLING_RATE',
                            type=int,
                            help='')
    arg_parser.add_argument('DURATION',
                            type=float,
                            help='')
    arg_parser.add_argument('MODEL_FILE',
                            help='')
    arg_parser.add_argument('NOISY_TRAIN_FILE',
                            help='')
    arg_parser.add_argument('CLEAN_TRAIN_FILE',
                            help='')
    arg_parser.add_argument('NOISY_TEST_FILE',
                            help='')
    arg_parser.add_argument('PREDICTION_OUTFILE',
                            help='')
    arg_parser.add_argument('LOG_OUTFILE',
                            help='')

    args = arg_parser.parse_args()

    # parse arguments
    time_steps = int(args.SAMPLING_RATE * args.DURATION)

    # load data
    arrayx = np.memmap(args.NOISY_TRAIN_FILE, dtype='float32', mode='c')
    arrayx = arrayx.reshape((-1, time_steps, 1))

    arrayy = np.memmap(args.CLEAN_TRAIN_FILE, dtype='float32', mode='c')
    arrayy = arrayy.reshape((-1, time_steps, 1))

    array_test = np.memmap(args.NOISY_TEST_FILE, dtype='float32', mode='c')
    array_test = array_test.reshape((-1, time_steps, 1))

    # load or create model
    model = create_model(time_steps)

    if os.path.exists(args.MODEL_FILE):
        model.load_weights(args.MODEL_FILE)
    else:
        model.save_weights(args.MODEL_FILE)

    model.compile(
        loss='mse',
        optimizer=Adam(lr=args.learning_rate),
        metrics=['accuracy'],
    )

    # plot model
    if args.plot_model is not None:
        plot_model(model, to_file=args.plot_model, show_shapes=True)

    # train model
    history_callback = LossHistory()

    model.fit(
        arrayx,
        arrayy,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[history_callback],
    )

    # save model
    if args.epochs != 0:
        model.save_weights(args.MODEL_FILE)

    # save history
    with open(args.LOG_OUTFILE, 'w') as file_history:
        file_history.write('loss\taccuracy\n')

        for loss in history_callback.losses:
            file_history.write('%f\n' % loss)

    # run prediction
    array_prediction = model.predict(
        array_test,
        batch_size=args.batch_size,
    )
    array_output = np.memmap(args.PREDICTION_OUTFILE, dtype='float32', mode='w+', shape=array_prediction.shape)
    np.copyto(array_output, array_prediction)

if __name__ == '__main__':
    main()
