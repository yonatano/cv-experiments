#!/bin/sh

# brew install imagemagick

EXT=ppm
DEST=imgs_$EXT
mkdir -p $DEST;
for fl in $(ls imgs/*.png)
do
    fn=$(basename ${fl%.*})
    convert $fl $DEST/$fn.$EXT 
done

echo "converted" $(ls $DEST/ | wc -l) "images"
