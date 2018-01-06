#!/usr/bin/env bash

PRINT_USAGE () {
    echo -ne "Usage: $0 TARGET_SAMPLE_RATE SOURCE_DIR TARGET_DIR

Options:
    TARGET_SAMPLE_RATE  the sample rate to convert to
    SOURCE_DIR          the directory containing wav files
    TARGET_DIR          the directory where resampled wav files are stored in
"
}

# parse arguments
[ $# -eq 3 ] ||
    { PRINT_USAGE >&2; exit 1; }

SAMPLE_RATE=$1
SOURCE_DIR=$2
TARGET_DIR=$3

# sanity check
command -v sox &>/dev/null &&
    command -v parallel &>/dev/null ||
        { echo 'error: sox or GNU parallel is not installed'; exit 1; }

# preparing
mkdir -p $TARGET_DIR

# convert file
for src in $SOURCE_DIR/*.wav; do
    filename=$(basename $src)
    dst=$TARGET_DIR/$filename
    echo sox $src $dst channels 1 rate $SAMPLE_RATE
done | parallel
