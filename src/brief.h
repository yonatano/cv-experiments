#ifndef BRIEF_H
#define BRIEF_H

#include <math.h>
#include <armadillo>
#include "distributions.h"
#include "fast.h"
#include "utils.h"

using namespace std;
using namespace arma;

typedef bitset<64> brief64;
typedef bitset<128> brief128;
typedef bitset<256> brief256;
typedef bitset<512> brief512;

typedef struct patch {
    Point center;
    int startx;
    int starty;
    int endx;
    int endy;

    patch(Mat<int> img, Point center, int sidelength) {
        int startx = center.x - sidelength / 2;
        int starty = center.y - sidelength / 2;
        int endx = center.x + sidelength / 2;
        int endy = center.y + sidelength / 2;
        this->center = center;
        this->startx = max(0, min(int(img.n_cols), startx));
        this->starty = max(0, min(int(img.n_rows), starty));
        this->endx = max(0, min(int(img.n_cols), endx));
        this->endy = max(0, min(int(img.n_rows), endy));
    }

    Mat<int> sub(Mat<int> img) {
        return img.submat(this->starty, this->startx, this->endy, this->endx);
    }

    int area() {
        return (this->endx - this->startx) * (this->endy - this->starty);
    }

    int size() {
        return (this->endx - this->startx);
    }

    Point clip(Point p) {
        int clx = max(this->startx, min(this->endx, p.x));
        int cly = max(this->starty, min(this->endy, p.y));
        return Point(clx, cly);
    }

    Point fromlocal(Point p) {
        int lx = p.x + this->startx;
        int ly = p.y + this->starty;
        return Point(lx, ly);
    }

    Point tolocal(Point p) {
        int lx = p.x - this->startx;
        int ly = p.y - this->starty;
        return Point(lx, ly);
    }

} Patch;

brief512 generateBRIEFDescriptor(Mat<int>& img, Patch& p, int size, vector<Point>& pts);
brief512 generateBRIEFDescriptor(Mat<int>& img, Patch& p, vector<Point>& pairs, vector<Point>& pts);
void sampleWithGaussianStrategy(Mat<int>& img, Patch& p, Point& pt1, Point& pt2);
void sampleWithLocalizedGaussianStrategy(Mat<int>& img, Patch& p, Point& pt1, Point& pt2);

#endif