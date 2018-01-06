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
    arg_parser.add_argument('SAMPLE_RATE',
                            type=int,
                            help='')
    arg_parser.add_argument('MAX_DURATION',
                            type=float,
                            help='')
    arg_parser.add_argument('CLEAN_DATA_DIR',
                            help='')
    arg_parser.add_argument('NOISY_DATA_DIR',
                            help='')
    arg_parser.add_argument('OUTFILE_PREFIX',
                            help='')
    args = arg_parser.parse_args()

    assert args.SAMPLE_RATE > 0 and args.MAX_DURATION > 0

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
    def fill_array(array, filenames, path_dict):
        assert array.shape[0] == len(filenames)

        for ind, fname in enumerate(filenames):
            wav_path = path_dict[fname]
            sample_rate, audio = scipy.io.wavfile.read(wav_path)

            # verify
            if sample_rate != args.SAMPLE_RATE:
                print('warning: the sample rate of file %s (%d) differ from that specified in argument (%d), skip this file' % (fname, sample_rate, args.SAMPLE_RATE),
                      file=sys.stderr)
                continue

            # sanitize audio
            audio = audio[:max_steps]

            padding_steps = max(0, max_steps - audio.shape[0])
            audio = np.pad(audio, [(0, padding_steps)], 'constant', constant_values=0.0)

            audio = audio / (2**16 - 1)

            array[ind] = audio

    # make mmapped files
    n_samples = len(paired_filenames)
    max_steps = int(args.SAMPLE_RATE * args.MAX_DURATION)

    saved_filenames = list(paired_filenames)
    saved_filenames.sort()

    path_clean = '%s_clean.bin' % args.OUTFILE_PREFIX
    path_noisy = '%s_noisy.bin' % args.OUTFILE_PREFIX

    array_clean = np.memmap(path_clean, dtype='float32', mode='w+', shape=(n_samples, max_steps))
    array_noisy = np.memmap(path_noisy, dtype='float32', mode='w+', shape=(n_samples, max_steps))

    print('creating %s' % path_clean)
    fill_array(array_clean, saved_filenames, clean_wav_paths)


    print('creating %s' % path_noisy)
    fill_array(array_noisy, saved_filenames, noisy_wav_paths)

    # make filename list
    path_filenames = '%s_filenames.txt' % args.OUTFILE_PREFIX
    print('creating %s' % path_filenames)

    with open(path_filenames, 'w+') as file_fnames:
        file_fnames.write('\n'.join(saved_filenames))
        file_fnames.write('\n')

if __name__ == '__main__':
    main()
