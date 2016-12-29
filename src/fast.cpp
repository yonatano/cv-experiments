/*
fast.cpp:
extractCircle
    - midpoint circle algorithm (Bresenham circle)
segmentTestCriterion(img, x, y, n, thresh)
    - optimization for n > 12 -- consider pixels 1,5,9,13
extractCorners(img, n, thresh) -- have DEFAULTS for these

main.cpp / util.cpp: 
loadImg
generateTrainingData -- loop over images and generate data, etc
something like:
    fastDetector = DecisionTree(data)

*/

#include "fast.h"
#include "utils.h"

using namespace std;
using namespace arma;

// https://en.wikipedia.org/wiki/Midpoint_circle_algorithm
// returns a vector of Points representing the generated circle.
vector<point> computeCircle(int cx, int cy, int r) {
    // fill out the points along the circle for one octant, and reflect them
    // to complete the circle.
    vector<point> circle;

    if (r < 0) {
        return circle;
    }

    int x = r;
    int y = 0;
    int err = 0;
    while (x >= y) {

        // reflections to all octants
        for (int i = -1; i <= 1; i += 2) {
            for (int j = -1; j <= 1; j += 2) {
                Point p(cx + i * x, cy + j * y);
                if (indexInVector(circle, p) == -1) {
                    circle.push_back(p);
                }
            }
        }
        for (int i = -1; i <= 1; i += 2) {
            for (int j = -1; j <= 1; j += 2) {
                Point p(cx + i * y, cy + j * x);
                if (indexInVector(circle, p) == -1) {
                    circle.push_back(p);
                }
            }
        }

        // adjust parameters to generate the remaining points in the octant
        if (err <= 0) {
            y += 1;
            err += 2 * y + 1;
        }
        if (err > 0) {
            x -= 1;
            err -= 2 * x + 1;
        }
    }

    return circle;
}

// circumference: 2pi*r = numPx
// iterate over possible radii until we generate a circle with the
// required number of pixels
vector<Point> computeCircleOfSize(int cx, int cy, int numPx) {
    float r = float(numPx) / (2 * M_PI);
    int rMin = int(floor(r)) - 1;
    int rMax = int(ceil(r)) + 1;

    vector<Point> circle; 
    for (int i = rMin; i < rMax + 1; i++) {
        circle = computeCircle(cx, cy, i);
        if (circle.size() >= numPx)
            return circle;
    }
    return circle;
}


vector<Point> shiftPointCenter(vector<Point> points, int cx, int cy) {
    for (int i = 0; i < points.size(); i++) {
        points[i].x += cx;
        points[i].y += cy;
    }
    return points;
}

int relativeBrightness(int pxOne, int pxTwo, int thresh) {
    if (pxOne > pxTwo + thresh)
        return REL_BRIGHTER;
    if (pxOne < pxTwo - thresh)
        return REL_DARKER;
    return REL_EQUAL;
}

// computeSegmentTestCriterion() 
// returns a bool indicating whether or not pixel at x,y of img 
// is a corner by checking if there exist N contiguous pixels in a circle of R 
// that are all brighter or darker by some threshold T,
bool isCornerWithSegmentTestCriterion(Mat<int> img, int cx, int cy, 
                                      int n, int numPx, int thresh) {
    vector<Point> circle = computeCircleOfSize(cx, cy, numPx);
    return isCornerWithSegmentTestCriterion(img, circle, cx, cy, n, thresh);
}

// overloaded method, allows for circle to be precomputed for faster processing
// over many pixels for a fixed circle size.
bool isCornerWithSegmentTestCriterion(Mat<int> img, vector<Point> circle, 
                          int cx, int cy, int n, int thresh) {
    int cmag = img(cx, cy);
    if (circle.size() == 16 && n == 12) { // perform simpler negative test
        int relup = relativeBrightness(cmag, img(circle[IDX_UP].x, circle[IDX_UP].y), thresh);
        int reldn = relativeBrightness(cmag, img(circle[IDX_DOWN].x, circle[IDX_DOWN].y), thresh);
        int rellf = relativeBrightness(cmag, img(circle[IDX_LEFT].x, circle[IDX_LEFT].y), thresh);
        int relri = relativeBrightness(cmag, img(circle[IDX_RIGHT].x, circle[IDX_RIGHT].y), thresh);
        bool condOne    = (relup == relri == reldn) && relup != REL_EQUAL;
        bool condTwo    = (relri == reldn == rellf) && relri != REL_EQUAL;
        bool condThree  = (reldn == relri == relup) && reldn != REL_EQUAL;
        bool condFour   = (rellf == relup == relri) && rellf != REL_EQUAL;
        if (!condOne && !condTwo && !condThree && !condFour) { // rule out the possibility of N contiguous like pixels
            return false;
        }
    }

    vector<int> relBrightness;
    for (int i = 0; i < circle.size(); i++) {
        int rmag = img(circle[i].x, circle[i].y);
        relBrightness[i] = relativeBrightness(cmag, rmag, thresh);
    }

    int streakLen = 0;
    int streakVal = 0;
    for (int i = 0; i < relBrightness.size(); i++) {
        int reli = relBrightness[i];
        if (reli != REL_EQUAL) {
            if (reli == streakVal) {
                streakLen += 1;
            } else {
                streakLen = 1;
            }
        }
        streakVal = reli;
        if (streakVal >= n)
            return true;
    }
    return false;
}