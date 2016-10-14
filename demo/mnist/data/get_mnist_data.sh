#!/usr/bin/env sh
# This scripts downloads the mnist data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
rm -rf "$DIR/raw_data"
mkdir "$DIR/raw_data"
cd "$DIR/raw_data"

echo "Downloading..."

for fname in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
do
    if [ ! -e $fname ]; then
        wget --no-check-certificate http://yann.lecun.com/exdb/mnist/${fname}.gz
        gunzip ${fname}.gz
    fi
done

cd $DIR
rm -f *.list
python generate_list.py

