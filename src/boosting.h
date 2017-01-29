#ifndef BOOSTING_H
#define BOOSTING_H

#include <math.h>
#include <limits.h>
#include <armadillo>
#include "fast.h"
#include "brief.h"

using namespace std;
using namespace arma;

// class WeakLearner {
// public:
//     WeakLearner() {}
//     virtual ~WeakLearner() {}
//     virtual int predict(Mat<int> &img) = 0;
// };

class BRIEFTestStub {
public:
    Point p1;
    Point p2;

    BRIEFTestStub() {}
    ~BRIEFTestStub() {}

    BRIEFTestStub(Point p1, Point p2) {
        this->p1 = p1;
        this->p2 = p2;
    }

    int predict(Mat<int>& img) {
        return img(p1.y, p1.x) < img(p2.y, p2.x);
    }

    int predict(Mat<int>& img, Patch p) {
        Point pt1 = p.fromlocal(p1);
        Point pt2 = p.fromlocal(p2);
        return img(pt1.y, pt1.x) < img(pt2.y, pt2.x);
    }
};

void AdaBoost(vector<Mat<int> >& dataX,
              vector<int>& dataY, 
              vector<BRIEFTestStub>& learners, 
              vector<float>& samplewt, 
              vector<float>& learnerwt,
              vector<int>& ensemble);

int predictEnsemble(Mat<int>& x,
                    vector<BRIEFTestStub>& ensemble, 
                    vector<float>& learnerwt);

#endif