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

    string fileName = "data/imgs_pgm/blue-bottle-coffee.pgm";
    if (argc > 1) {
        fileName = argv[1];
        cout << "loading image: " << fileName << endl;
    }
    Mat<int> img;
    img.load(fileName, pgm_binary);
    cout << "loaded image: (" << fileName << ") " << img.n_rows << "x" << img.n_cols << endl;

    // generate training data for corner detector
    stringstream trainingData;
    for (int i = 0; i < DEFAULT_CIRCLESZ; i++) { // write csv header
        trainingData << "R" << i << ',';
    }
    trainingData << "Y" << '\n';

    int npos;
    int nneg;

    vector<Point> circle = computeCircleOfSize(0, 0, DEFAULT_CIRCLESZ);
    int startx = 4; // radius size of 16px circle
    int starty = 4;
    int endx = (img.n_cols - 1) - 4;
    int endy = (img.n_rows - 1) - 4;

    for (int cy = starty; cy <= endy; cy++) {
        for (int cx = startx; cx <= endx; cx++) {
            int cmag = img(cy, cx);
            vector<int> relBrightness = relativeBrightnessForCircle(img, 
                                                                    cmag, 
                                                                    shiftPointCenter(circle, cx, cy), 
                                                                    DEFAULT_MAG_THRESHOLD);
            bool isCorner = isCornerWithSegmentTestCriterion(img, 
                                                             cx, 
                                                             cy, 
                                                             shiftPointCenter(circle, cx, cy),
                                                             DEFAULT_PX_COUNT_REQ,
                                                             DEFAULT_MAG_THRESHOLD);
            if (isCorner) {
                npos++;
            } else {
                nneg++;
            }

            for (auto v = relBrightness.begin(); v != relBrightness.end(); v++) {
                trainingData << *v << ',';
            }
            trainingData << isCorner;
            trainingData << '\n';
        }
    }

    ofstream outFile;
    outFile.open("train.csv");
    outFile << trainingData.str();

    cout << "POS: " << npos << " NEG: " << nneg << endl;
}