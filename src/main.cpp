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

void generateTrainingData() {
    string fileName = "data/imgs_pgm/blue-bottle-coffee.pgm";
    Mat<int> img;
    img.load(fileName, pgm_binary);
    cout << "loaded image: (" << fileName << ") " << img.n_rows << "x" << img.n_cols << endl;

    // generate training data for corner detector
    stringstream trainingData;
    for (int i = 0; i < DEFAULT_CIRCLESZ; i++) { // write csv header
        trainingData << "R" << i << ',';
    }
    trainingData << "Y" << '\n';

    int npos = 0;
    int nneg = 0;

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
    cout << "POS: " << npos << " NEG: " << nneg << endl;

    ofstream outFile;
    outFile.open("train.csv");
    outFile << trainingData.str();
    outFile.close();
}

void testID3() {
    // load training data
    string trainFile = "data/test_data.csv";
    cout << "DATA:" << endl;
    map<string, vector<string> > dat = loadCSVAsString(trainFile);
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

    // generate training data
    // cout << "generating training data..." << endl;
    // generateTrainingData();

    map<string, vector<int> > data = loadCSVAsInt("train.csv");
    map<string, vector<int> > train;
    map<string, vector<int> > test;
    splitTrainingData(data, train, test, data["Y"].size(), 0.8);
    cout << "loaded " << data["Y"].size() << " samples (" << train["Y"].size() << "/" << test["Y"].size() << " split)" << endl;

    // data -> matrix
    Mat<int> Xtrain;
    Col<int> Ytrain;
    Mat<int> Xtest;
    Col<int> Ytest;
    dataToMatrix(train, "Y", Xtrain, Ytrain);
    dataToMatrix(test, "Y", Xtest, Ytest);
    cout << "Xtrain: " << Xtrain.n_rows << "x" << Xtrain.n_cols << endl;
    cout << "Ytrain: " << Ytrain.n_rows << "x" << Ytrain.n_cols << endl;
    cout << "Xtest: " << Xtest.n_rows << "x" << Xtest.n_cols << endl;
    cout << "Ytest: " << Ytest.n_rows << "x" << Ytest.n_cols << endl;
    cout << endl;

    // train a corner classifier
    cout << "generating decision tree..." << endl;
    DecisionTree tree;
    tree.fitWithID3(Xtrain, Ytrain);

    // evaluate
    cout << "confusion matrix:" << endl;
    Col<int> Yp = tree.predict(Xtest);
    printConfusionMatrix(Ytest, Yp);
    cout << endl;

    // sanity-check
    cout << "sanity-check... using X ~ Unif(0, 1) for each prediction:" << endl;
    Col<int> Ybad(Yp.n_rows, 1);
    Col<float> Yrand(Yp.n_rows, 1);
    Yrand.randu();
    for (int i = 0; i < Yrand.n_rows; i++) {
        Ybad(i) = round(Yrand(i) * 1);
    }
    printConfusionMatrix(Ytest, Ybad);
}