"""
Load images from CarBlur dataset 
"""
import os
import sys
import numpy as np
from PIL import Image

def isoverlapping(startx1, starty1, endx1, endy1, 
                  startx2, starty2, endx2, endy2):
    xszone = abs(endx1 - startx1)
    yszone = abs(endy1 - starty1)
    xsztwo = abs(endx2 - startx2)
    ysztwo = abs(endy2 - starty2)
    condv = abs(endy2 - endy1) < ysztwo or abs(endy1 - endy2) < yszone
    condh = abs(endx2 - endx1) < xsztwo or abs(endx1 - endx2) < xszone
    return condv or condh

# pass a window over the bounding box, skipping by step pixels 
# with each iteration.
def compute_subwindows(im, patchsz, step, startx, starty, endx, endy):
    subwindows = []
    while startx < (endx - patchsz):
        while starty < (endy - patchsz):
            subwindows.append( (startx, starty, startx + patchsz, starty + patchsz) )
            starty += step
        startx += step 
    return subwindows

# randomly sample a number of windows not overlapping with the bounding box
def compute_randwindows(im, patchsz, numwindows, startx, starty, endx, endy):
    subwindows = []
    while len(subwindows) < numwindows:
        randx = np.random.randint(0, im.width - patchsz)
        randy = np.random.randint(0, im.height - patchsz)
        if not isoverlapping(randx, randy, randx + patchsz, randy + patchsz,
                             startx, starty, endx, endy):
            subwindows.append( (randx, randy, randx + patchsz, randy + patchsz) )
    return subwindows

if __name__ == "__main__":
    data_dir = sys.argv[1] # dataset directory
    save_dir = sys.argv[2]
    patchsz = int(sys.argv[3]) # size of patch to generate data for
    os.makedirs(save_dir + "/pos")
    os.makedirs(save_dir + "/neg")

    ground = data_dir + "/groundtruth_rect.txt"
    images = data_dir + "/img/"
    step = 1

    coords = [] # (x, y, width, height)
    with open(ground) as ground_truth:
        dat = ground_truth.read()
        for d in dat.split('\r\n'):
            x, y, w, h = [int(c) for c in d.split("\t")]
            coords.append( (x, y, x + w, y + h) )

    for i, f in enumerate(os.listdir(images)):
        c = coords[i]
        im = Image.open(images+f)
        pos = compute_subwindows(im, patchsz, step, c[0], c[1], c[2], c[3])
        neg = compute_randwindows(im, patchsz, len(pos), c[0], c[1], c[2], c[3])

        print "%s: (%s pos %s neg)" % (f, len(pos), len(neg))

        for k, w in enumerate(pos):
            im.crop(w).save(save_dir + "/pos/%s_%s.jpg" % (i, k))

        for k, w in enumerate(neg):
            im.crop(w).save(save_dir + "/neg/%s_%s.jpg" % (i, k))