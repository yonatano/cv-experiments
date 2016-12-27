#!/bin/sh

# brew install imagemagick

EXT=ppm
SRC=imgs
DEST=imgs_$EXT
mkdir -p $DEST;
for fl in $(ls $SRC/*.png)
do
    fn=$(basename ${fl%.*})
    convert $fl $DEST/$fn.$EXT 
done

echo "converted" $(ls $DEST/ | wc -l) "images"
