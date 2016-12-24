#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <armadillo>

// #include "decisiontree.h"

using namespace std;
using namespace arma;

map<string, vector<string> > loadCSV(string fileName) {
    map<string, vector<string> > data;
    ifstream file(fileName);

    // read column names
    vector<string> columns;
    string colstemp;
    getline(file, colstemp, '\n');
    stringstream colss(colstemp);
    for (string c; getline(colss, c, ',');) {
        columns.push_back(c);
    }
    for (string line; getline(file, line, '\n');) {
        stringstream ss(line);
        int idx = 0;
        for (string entry; getline(ss, entry, ',');) {
            data[columns[idx]].push_back(entry);
            idx++;
        }
    }

    return data;
} 

void printCSV(map<string, vector<string> > dat) {
    for (auto m = dat.begin(); m != dat.end(); m++) {
        string col = m->first;
        vector<string> vals = m->second;
        cout << col << " (" << vals.size() << "):\t";
        for (auto v = vals.begin(); v != vals.end(); v++) {
            cout << *v << ", ";
        }
        cout << endl;
    }
}



int main() {
    // 0. Load features
    /*
        F1   F2   F3  =  Y
        a    a    2      0
        b    b    1      1
    */

    // 2. Aggregate the feature space for every feature
    /*
        F1: {a, b}
        F2: {a, b}
        F3: {1, 2}
        Y: {0, 1}
    */

    // 3. Turn every entry into an index into its feature-space list
    /*
        F1   F2   F3  =  Y
        0    0    1      0
        1    1    0      1
    */

    // 4. ID3(X, Y, X{f->[k]}, Y[k])

    string trainFile = "data/test_data.csv";
    map<string, vector<string> > dat = loadCSV(trainFile);
    printCSV(dat);
}
