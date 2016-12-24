#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <armadillo>

// #include "decisiontree.h"

using namespace std;
using namespace arma;

// forward-declaration headers
map<string, vector<string> > loadCSV(string fileName);
void printCSV(map<string, vector<string> >);
void labelEncodeData(map<string, vector<string> > dat);
vector<string> uniqueElems(vector<string> v);
int indexInVector(vector<int> v, int s);

// loads CSV data into a map from column_names -> [data] given a fileName
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

// prints a structure generated with loadCSV
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

// replaces all categorical entries with a corresponding list index 
void labelEncodeData(map<string, vector<string> > dat) {
    for (auto m = dat.begin(); m != dat.end(); m++)  {
        string key = m->first;
        vector<string> ftrs = uniqueElems(m->second);

    }
}

// generates a matrix from a data map. y is the key corresponding to the desired
// predictions.
// returns X, a matrix containing the features. y, a column vector containing
// the desired predictions.
void dataToMatrix(map<string, vector<string> > dat, string y) {
    // xsz: # of features, length(dat.keys())
    // ysz: # of outcomes, length(uniqueElems(dat[y]))
    // n: # of data samples, (dat[0].size())
    // X: mat(n, xzs)
    // Y: mat(n, 1)
    // numelems = 
    // mat data = ()
}

/* Util Functions */
vector<string> uniqueElems(vector<string> v) {
    set<string> s(v.begin(), v.end());
    vector<string> vec(s.begin(), s.end());
    return vec;
}

int indexInVector(vector<int> v, int s) {
    int pos = find(v.begin(), v.end(), s) - v.begin();
    return pos;
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

    // test 
    vector<int> v = {0, 1, 2, 3, 4};
    cout << indexInVector(v, 3) << endl;

}
