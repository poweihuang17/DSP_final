#!/usr/bin/env python3
import os
import argparse
import numpy as np
import scipy.io.wavfile

def main():
    # parse arguments
    arg_parser = argparse.ArgumentParser(description='The script is intended for model creation and training')
    arg_parser.add_argument('SAMPLE_RATE',
                            type=int,
                            help='')
    arg_parser.add_argument('DURATION',
                            type=float,
                            help='')
    arg_parser.add_argument('PREDICTION_FILE',
                            help='')
    arg_parser.add_argument('FILENAME_FILE',
                            help='')
    arg_parser.add_argument('OUTPUT_DIR',
                            help='')

    args = arg_parser.parse_args()

    # parse arguments
    time_steps = int(args.SAMPLE_RATE * args.DURATION)

    # load data
    array_prediction = np.memmap(args.PREDICTION_FILE, dtype='float32', mode='c')
    array_prediction = array_prediction.reshape((-1, time_steps))

    with open(args.FILENAME_FILE, 'r') as file_filenames:
        filenames = [line[:-1] for line in file_filenames]

    assert len(filenames) == array_prediction.shape[0]

    # convert to wav files
    for fname, array_audio in zip(filenames, array_prediction):
        wav_path = os.path.join(args.OUTPUT_DIR, fname)
        scipy.io.wavfile.write(wav_path, args.SAMPLE_RATE, array_audio)

if __name__ == '__main__':
    main()
