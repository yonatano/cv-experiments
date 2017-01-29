"""
Load images from CarBlur dataset 
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

def isoverlapping(startx1, starty1, endx1, endy1, 
                  startx2, starty2, endx2, endy2):
    cond1 = startx1 > endx2
    cond2 = endx1 < startx2
    cond3 = starty1 > endy2
    cond4 = endy1 < starty2
    return not (cond1 or cond2 or cond3 or cond4)

# pass a window over the bounding box, skipping by step pixels 
# with each iteration.
def compute_subwindows(im, patchsz, step, startx, starty, endx, endy):
    currx = startx 
    curry = starty 
    subwindows = []
    while currx <= (endx - patchsz + step):
        while curry <= (endy - patchsz + step):
            subwindows.append( (currx, curry, currx + patchsz, curry + patchsz) )
            curry += step
        curry = starty
        currx += step
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
    step = 5

    coords = [] # (x, y, width, height)
    with open(ground) as ground_truth:
        dat = ground_truth.read()
        for d in dat.split('\r\n'):
            x, y, w, h = [int(c) for c in d.split("\t")]
            coords.append( (x, y, x + w, y + h) )

    for i, f in enumerate(os.listdir(images)):
        c = coords[i]
        im = Image.open(images+f)
        iw, ih = im.size
        draw = ImageDraw.Draw(im) 
        pos = compute_subwindows(im, patchsz, step, c[0], c[1], c[2], c[3])
        # neg = compute_randwindows(im, patchsz, len(pos), c[0], c[1], c[2], c[3])
        neg = compute_subwindows(im, patchsz, step, 0, 0, iw, ih)
        neg = [p for p in neg if not isoverlapping(p[0], p[1], p[2], p[3], 
                                                   c[0], c[1], c[2], c[3])]

        print "%s: (%s pos %s neg)" % (f, len(pos), len(neg))

        for k, w in enumerate(pos):
            im.crop(w).save(save_dir + "/pos/%s_%s.png" % (i, k))
            draw.rectangle([(w[0], w[1]), (w[2], w[3])], outline="red", fill=None)

        for k, w in enumerate(neg):
            im.crop(w).save(save_dir + "/neg/%s_%s.png" % (i, k))
            draw.rectangle([(w[0], w[1]), (w[2], w[3])], outline="blue", fill=None)

        im.show()
        break

    
