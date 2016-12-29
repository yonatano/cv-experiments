#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <armadillo>
#include <Magick++.h>

#include "fast.h"
#include "decisiontree.h"
#include "utils.h"

using namespace std;
using namespace arma;
using namespace Magick;

void displayCube(Cube<int> im) {

}

void testID3() {
    // load training data
    string trainFile = "data/test_data.csv";
    cout << "DATA:" << endl;
    map<string, vector<string> > dat = loadCSV(trainFile);
    printVectorMap(dat);
    cout << endl;

    // report value-spaces for features
    cout << "Labels:" << endl;
    for (auto m = dat.begin(); m != dat.end(); m++)  {
        string key = m->first;
        vector<string> vals = m->second;
        vector<string> ftrs = uniqueElems(vals);
        cout << key << ": {";
        for (int i = 0; i < ftrs.size(); i++) {
            cout << ftrs[i] << ", ";
        }
        cout << "}" << endl;
    }
    cout << endl;

    // label-encode samples
    map<string, vector<int> > enc = labelEncodeData(dat);
    
    // data -> matrices
    Mat<int> X;
    Col<int> Y;
    dataToMatrix(enc, "outcome", X, Y);

    // generate tree
    DecisionTree tree;
    tree.fitWithID3(X, Y);
    cout << "ID3 RESULT:" << endl;
    tree.print();

    // evaluate predictions
    Col<int> Yp = tree.predict(X);
    cout << "CONFUSION MATRIX:" << endl;
    printConfusionMatrix(Y, Yp);
}

int main(int argc, char **argv) {
    InitializeMagick(*argv);

    string fileName = "data/imgs_ppm/blue-bottle-coffee.ppm";
    if (argc > 1) {
        fileName = argv[1];
        cout << "loading image: " << fileName << endl;
    }    
    Cube<int> img;
    img.load(fileName, ppm_binary);
    displayCube(img);

    int numPx = 16;
    vector<Point> circle = computeCircleOfSize(10, 10, numPx);
    for (auto v = circle.begin(); v != circle.end(); v++) {
        Point p = *v;
        cout << "(" << p.x << ", " << p.y << ")" << ", ";  
    }
    cout << circle.size() << " points" << endl;
}