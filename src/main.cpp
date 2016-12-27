#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <armadillo>

#include "decisiontree.h"
#include "utils.h"

using namespace std;
using namespace arma;


int main() {
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