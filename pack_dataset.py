#!/usr/bin/env python3
import os
import sys
import pickle
import argparse

import numpy as np
import scipy.io.wavfile

def main():
    # parse arguments
    arg_parser = argparse.ArgumentParser(description='')
    arg_parser.add_argument('RESAMPLE_RATE',
                            type=int,
                            help='')
    arg_parser.add_argument('MAX_DURATION',
                            type=float,
                            help='')
    arg_parser.add_argument('CLEAN_DATA_DIR',
                            help='')
    arg_parser.add_argument('NOISY_DATA_DIR',
                            help='')
    arg_parser.add_argument('OUTFILE',
                            help='')
    args = arg_parser.parse_args()

    assert args.RESAMPLE_RATE > 0 and args.MAX_DURATION > 0

    # some calculations
    max_steps = int(args.RESAMPLE_RATE * args.MAX_DURATION)

    # scan file paths
    clean_wav_paths = { filename: os.path.join(args.CLEAN_DATA_DIR, filename)
                        for filename in os.listdir(args.CLEAN_DATA_DIR) if filename[-4:] == '.wav' }
    noisy_wav_paths = { filename: os.path.join(args.NOISY_DATA_DIR, filename)
                        for filename in os.listdir(args.NOISY_DATA_DIR) if filename[-4:] == '.wav' }

    paired_filenames = clean_wav_paths.keys() and noisy_wav_paths.keys()
    unpaired_filenames = clean_wav_paths.keys() ^ noisy_wav_paths.keys()

    if len(unpaired_filenames) != 0:
        print('warning: these files are not paired in clean or noisy data: %s' % ', '.join(unpaired_filenames),
              file=sys.stderr)

    # utility function
    def sanitize_audio(audio):
        audio = audio[:max_steps]

        padding_steps = max(0, max_steps - audio.shape[0])
        audio = np.pad(audio, [(0, padding_steps)], 'constant', constant_values=0.0)

        return audio / (2**16 - 1)

    # load files
    clean_array = np.zeros((0, max_steps), dtype='float64')
    noisy_array = np.zeros((0, max_steps), dtype='float32')

    for filename in clean_wav_paths:
        print('processing %s' % filename)

        clean_path = clean_wav_paths[filename]
        noisy_path = noisy_wav_paths[filename]

        clean_sample_rate, clean_audio = scipy.io.wavfile.read(clean_path)
        noisy_sample_rate, noisy_audio = scipy.io.wavfile.read(noisy_path)

        if clean_sample_rate != noisy_sample_rate:
            print('warning: the clean and noisy data for file %s differ in sample rate (%d vs. %s), skip this file' % (filename, clean_sample_rate, noisy_sample_rate),
                  file=sys.stderr)
            continue

        if clean_audio.shape != noisy_audio.shape:
            print('warning: the clean and noisy data for file %s differ duration, skip this file' % (filename,),
                  file=sys.stderr)
            continue

        sanitized_clean_audio = sanitize_audio(clean_audio)
        sanitized_noisy_audio = sanitize_audio(noisy_audio)

        clean_array = np.append(clean_array, sanitized_clean_audio.reshape(1, -1), axis=0)
        noisy_array = np.append(noisy_array, sanitized_noisy_audio.reshape(1, -1), axis=0)

    with open(args.OUTFILE, 'wb') as file_output:
        pickle.dump((clean_array, noisy_array), file_output)

if __name__ == '__main__':
    main()
