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
    cout << "Data:" << endl;
    map<string, vector<string> > dat = loadCSV(trainFile);
    printVectorMap(dat);
    cout << endl;

    // possible values for features
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
    cout << "Label-Encoded data:" << endl;
    map<string, vector<int> > enc = labelEncodeData(dat);
    printVectorMap(enc);
    cout << endl;

    // matricize data
    Mat<int> X;
    Col<int> Y;
    dataToMatrix(enc, "outcome", X, Y);
    cout << "Data Matrices:" << endl;
    cout << "X:" << endl;
    X.print();
    cout << endl;
    cout << "Y:" << endl;
    Y.print();
    cout << endl;
    
    X.rows(find( Y == 1 )).print();

    // generate decision tree
    DecisionTree tree;
    tree.fitWithID3(X, Y);
    // tree.print();
    Col<int> Yp = tree.predict(X);
    Yp.print();
}
