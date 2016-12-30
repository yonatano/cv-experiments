#!/bin/sh

# brew install imagemagick

EXT=pgm
SRC="tiny-imagenet-200/val/images"
DEST="tiny_imagenet_$EXT"
mkdir -p $DEST;
for fl in $(find $SRC -name "*.JPEG")
do
    fn=$(basename ${fl%.*})
    echo "converting $fn"
    convert $fl $DEST/$fn.$EXT
done

echo "converted" $(ls $DEST/ | wc -l) "images"
