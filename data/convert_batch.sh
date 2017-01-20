#!/bin/sh

# brew install imagemagick

EXT=pgm
SRC=$1
DEST=$2
for fl in $(find $SRC -name "*.png")
do
    fn=$(basename ${fl%.*})
    echo "converting $fn"
    convert $fl $DEST/$fn.$EXT
done

echo "converted" $(ls $DEST/ | wc -l) "images"
