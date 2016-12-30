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

    // aggregate the points in each octant and combine them so they're ordered
    // contiguously.
    map<int, function<Point(int, int)> > octantFn = {
        { 0, [](int x, int y) { return Point(x, y); } },
        { 1, [](int x, int y) { return Point(y, x); } },
        { 2, [](int x, int y) { return Point(-y, x); } },
        { 3, [](int x, int y) { return Point(-x, y); } },
        { 4, [](int x, int y) { return Point(-x, -y); } },
        { 5, [](int x, int y) { return Point(-y, -x); } },
        { 6, [](int x, int y) { return Point(y, -x); } },
        { 7, [](int x, int y) { return Point(x, -y); } },
    };
    map<int, vector<Point> > octants;

    int x = r;
    int y = 0;
    int err = 0;
    while (x >= y) {
        for (int o = 0; o < 8; o++) {
            Point p = (octantFn[o])(x, y);
            if (indexInVector(octants[o], p) == -1) {
                octants[o].push_back(p);
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

    // points ordered counter-clockwise starting from (r, 0)
    for (int o = 0; o < 8; o++) {
        for (auto v = octants[o].begin(); v != octants[o].end(); v++) {
            circle.push_back(*v);
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

int relativeBrightness(int center, int relative, float thresh) {
    if (relative > center * (1 + thresh))
        return REL_BRIGHTER;
    if (relative < center * (1 - thresh))
        return REL_DARKER;
    return REL_EQUAL;
}

vector<int> relativeBrightnessForCircle(Mat<int> img, int cmag, vector<Point> circle, float thresh) {
    vector<int> relBrightness;
    for (int i = 0; i < circle.size(); i++) {
        int rmag = img(circle[i].y, circle[i].x);
        relBrightness.push_back( relativeBrightness(cmag, rmag, thresh) );
    }
    return relBrightness;
}
// computeSegmentTestCriterion() 
// returns a bool indicating whether or not pixel at x,y of img 
// is a corner by checking if there exist N contiguous pixels in a circle of R 
// that are all brighter or darker by some threshold T,
bool isCornerWithSegmentTestCriterion(Mat<int> img, int cx, int cy, 
                                      int n, int numPx, float thresh) {
    vector<Point> circle = computeCircleOfSize(cx, cy, numPx);
    return isCornerWithSegmentTestCriterion(img, cx, cy, circle, n, thresh);
}

// overloaded method, allows for circle to be precomputed for faster processing
// over many pixels for a fixed circle size.
bool isCornerWithSegmentTestCriterion(Mat<int> img, int cx, int cy, 
                                      vector<Point> circle, int n, float thresh) {
    int cmag = img(cy, cx);
    if (circle.size() == DEFAULT_CIRCLESZ && n == DEFAULT_PX_COUNT_REQ) { // perform simpler negative test
        int relup = relativeBrightness(cmag, img(circle[IDX_UP].y, circle[IDX_UP].x), thresh);
        int reldn = relativeBrightness(cmag, img(circle[IDX_DOWN].y, circle[IDX_DOWN].x), thresh);
        int rellf = relativeBrightness(cmag, img(circle[IDX_LEFT].y, circle[IDX_LEFT].x), thresh);
        int relri = relativeBrightness(cmag, img(circle[IDX_RIGHT].y, circle[IDX_RIGHT].x), thresh);
        bool condOne    = (relup == relri == reldn) && relup != REL_EQUAL;
        bool condTwo    = (relri == reldn == rellf) && relri != REL_EQUAL;
        bool condThree  = (reldn == relri == relup) && reldn != REL_EQUAL;
        bool condFour   = (rellf == relup == relri) && rellf != REL_EQUAL;
        if (!condOne && !condTwo && !condThree && !condFour) { // rule out the possibility of N contiguous like pixels
            return false;
        }
    }

    vector<int> relBrightness = relativeBrightnessForCircle(img, cmag, circle, thresh);
    vector<int> cp(relBrightness);
    relBrightness.insert(relBrightness.end(), cp.begin(), cp.end()); // for wrap-around check

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
        if (streakLen >= n)
            return true;
    }
    return false;
}