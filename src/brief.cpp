#include "brief.h"

using namespace std;
using namespace arma;

/*

- sample point from bivariate normal and clip 
- construct BRIEF descriptor, returns uint
    - BRIEF16(), BRIEF32() BRIEF64() (nd/8)
    - BRIEF(k) generates 8k pairs of points, performs test, sums them 2^i-1
    - perform test for two points
*/

// X, Y ~ Gaussian(0, 1/25 * S^2)
void sampleWithGaussianStrategy(Mat<int>& img, Patch& p, Point& pt1, Point& pt2) {
    float var = 1 / 25.0 * pow(p.size() / 2.0, 2);
    int x, y;
    // sample point 1
    x = int(sampleNormalDist(p.center.x, var));
    y = int(sampleNormalDist(p.center.y, var));
    pt1 = p.clip( Point(x, y) );
    // sample point 2
    x = int(sampleNormalDist(p.center.x, var));
    y = int(sampleNormalDist(p.center.y, var));
    pt2 = p.clip( Point(x, y) );
}

// X ~ Gaussian(0, 1/25 * S^2) Y ~ Gaussian(Xi, 1/100 * S^2)
void sampleWithLocalizedGaussianStrategy(Mat<int>& img, Patch& p, Point& pt1, Point& pt2) {
    float var1 = 1 / 25.0 * pow(p.size() / 2.0, 2);
    float var2 = 1 / 100.0 * pow(p.size() / 2.0, 2);
    int x, y;
    // sample point 1
    x = int(sampleNormalDist(p.center.x, var1));
    y = int(sampleNormalDist(p.center.y, var1));
    pt1 = p.clip( Point(x, y) );
    // sample point 2
    x = int(sampleNormalDist(x, var2));
    y = int(sampleNormalDist(y, var2));
    pt2 = p.clip( Point(x, y) );
}

uint64_t generateBRIEFDescriptor(Mat<int>& img, Patch& p, int size, vector<Point>& pts) {
    uint64_t descriptor = 0;
    Point pt1;
    Point pt2;
    Mat<int> sub = p.sub(img);
    smoothImageWithGaussian(sub);
    for (int i = 0; i < size; i++) {
        sampleWithGaussianStrategy(img, p, pt1, pt2);
        pts.push_back(pt1);
        pts.push_back(pt2);

        pt1 = p.tolocal(pt1);
        pt2 = p.tolocal(pt2);
        bool test = (sub(pt1.y, pt1.x) < sub(pt2.y, pt2.x));
        descriptor += test;
        descriptor = (descriptor << 1);
    }
    return descriptor;
}

uint64_t generateBRIEFDescriptor(Mat<int>& img, Patch& p, int size, vector<Point>& pairs, vector<Point>& pts) {
    uint64_t descriptor = 0;
    Point pt1;
    Point pt2;
    Mat<int> sub = p.sub(img);
    smoothImageWithGaussian(sub);
    for (int i = 0; i < size; i += 2) {
        pt1 = pairs[i];
        pt2 = pairs[i+1];
        pts.push_back( p.fromlocal(pt1) );
        pts.push_back( p.fromlocal(pt2) );

        bool test = (sub(pt1.y, pt1.x) < sub(pt2.y, pt2.x));
        descriptor += test;
        descriptor = (descriptor << 1);
    }
    return descriptor;
}
