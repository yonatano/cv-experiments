#ifndef FAST_H
#define FAST_H

#include <iostream>
#include <math.h>
#include <vector>
#include <armadillo>

using namespace std;
using namespace arma;

const int IDX_UP    = 0;
const int IDX_DOWN  = 8;
const int IDX_RIGHT = 4;
const int IDX_LEFT  = 12;

const int DEFAULT_MAG_THRESHOLD = 0;
const int DEFAULT_PX_COUNT_REQ  = 12;
const int DEFAULT_CIRCLESZ      = 16;

// XXX: change to enum
const int REL_EQUAL     = 0;
const int REL_BRIGHTER  = 1;
const int REL_DARKER    = 2;

typedef struct point {
    int x;
    int y;

    point(int x, int y) {
        this->x = x;
        this->y = y;
    }

    bool operator==(const point& rhs) const {
        return (this->x == rhs.x && this->y == rhs.y);
    }

} Point;

vector<Point> computeCircle(int cx, int cy, int r);
vector<Point> computeCircleOfSize(int cx, int cy, int numPx);
vector<Point> shiftPointCenter(vector<int> points, int cx, int cy);
bool isCornerWithSegmentTestCriterion(Mat<int> img, int cx, int cy, int n, int csz, int thresh);
bool isCornerWithSegmentTestCriterion(Mat<int> img, vector<Point> circle, int cx, int cy, int n, int thresh);

#endif