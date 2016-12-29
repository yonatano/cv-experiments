#ifndef FAST_H
#define FAST_H

#include <iostream>
#include <math.h>
#include <vector>
#include <armadillo>

using namespace std;
using namespace arma;

const int IDX_UP    = 4;
const int IDX_DOWN  = 12;
const int IDX_RIGHT = 0;
const int IDX_LEFT  = 8;

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


    friend ostream &operator<<( ostream &output, const point p) {
        output << "(" << p.x << "," << p.y << ")";
        return output;
    }

} Point;

vector<Point> computeCircle(int cx, int cy, int r);
vector<Point> computeCircleOfSize(int cx, int cy, int numPx);
vector<Point> shiftPointCenter(vector<Point> points, int cx, int cy);
int relativeBrightness(int magOne, int magTwo, int thresh);
vector<int> relativeBrightnessForCircle(Mat<int> img, int cmag, vector<Point> circle, int thresh);
bool isCornerWithSegmentTestCriterion(Mat<int> img, int cx, int cy, int n, int csz, int thresh);
bool isCornerWithSegmentTestCriterion(Mat<int> img, int cx, int cy, vector<Point> circle, int n, int thresh);

#endif