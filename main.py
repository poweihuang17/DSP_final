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
    arg_parser.add_argument('MODEL_FILE',
                            help='')
    args = arg_parser.parse_args()

    # load or create model
    model = create_model(88200)

    if os.path.exists(args.MODEL_FILE):
        model.load_weights(args.MODEL_FILE)
    else:
        model.save_weights(args.MODEL_FILE)

    model.compile(
        loss='mse',
        optimizer=Adam(lr=args.learning_rate)
    )

    # train model
    #with lzma.open("../dataset_clean.bin.xz","rb") as f:
    #with open("../dataset_clean.bin") as f:
    #    file_content = f.read()
    #    with lzma.open("../dataset_noisy.bin.xz","rb") as g:
    #        file_content2=g.read()
    arrayx = np.memmap("../dataset_noisy.bin", dtype='float32', mode='c')
    arrayx = arrayx.reshape((-1, 22050 * 4,1))

    arrayy = np.memmap("../dataset_clean.bin", dtype='float32', mode='c')
    arrayy = arrayy.reshape((-1, 22050 * 4,1))
    
            #x = np.random.rand(1, 30, 1)
            #y = np.random.rand(1, 30, 1)
            
    model.fit(arrayx, arrayy)

if __name__ == '__main__':
    main()
