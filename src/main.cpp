#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <armadillo>

#include "decisiontree.h"

using namespace std;
using namespace arma;

// forward-declaration headers
map<string, vector<string> > loadCSV(string fileName);
map<string, vector<int> > labelEncodeData(map<string, vector<string> > dat);
vector<string> uniqueElems(vector<string> v);
int indexInVector(vector<string> v, string s);

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

// replaces all categorical entries with a corresponding list index
map<string, vector<int> > labelEncodeData(map<string, vector<string> > dat) {
    map<string, vector<int> > encoded; 
    for (auto m = dat.begin(); m != dat.end(); m++)  {
        string key = m->first;
        vector<string> vals = m->second;
        vector<string> ftrs = uniqueElems(vals);
        vector<int> encodedFtrs(vals.size());
        for (int i = 0; i < vals.size(); i++) {
            encodedFtrs[i] = indexInVector(ftrs, vals[i]);
        }
        encoded[key] = encodedFtrs;
    }
    return encoded;
}

// generates a matrix from a data map. y is the key corresponding to the desired
// predictions.
// returns X, a matrix containing the features. y, a column vector containing
// the desired predictions.
void dataToMatrix(map<string, vector<int> > dat, string y, Mat<int>& X, Mat<int>& Y) {
    // Ysz: outcome dimensionality
    // Xsz: input dimensionality
    // N: number of samples 
    int Ysz = 1;    
    int Xsz = dat.size() - Ysz;
    int N = dat[y].size();

    X.resize(N, Xsz);
    Y.resize(N, Ysz);

    int idx = 0;
    for (auto m = dat.begin(); m != dat.end(); m++) { // each map value corresponds to a column of data
        string key = m->first;
        vector<int> vals = m->second;
        if (y.compare(key)) {
            Col<int> col(vals);
            X.col(idx) = col;
            idx++;
        }
    }
    Col<int> col(dat[y]);
    Y.col(0) = col;
}

/* Util Functions */
vector<string> uniqueElems(vector<string> v) {
    set<string> s(v.begin(), v.end());
    vector<string> vec(s.begin(), s.end());
    return vec;
}

int indexInVector(vector<string> v, string s) {
    int pos = find(v.begin(), v.end(), s) - v.begin();
    if (pos >= v.size()) {
        return -1;
    }
    return pos;
}

template<class T>
void printVectorMap(map<string, vector<T> > dat) {
    for (auto m = dat.begin(); m != dat.end(); m++) {
        string col = m->first;
        vector<T> vals = m->second;
        cout << col << " (" << vals.size() << "):\t";
        for (auto v = vals.begin(); v != vals.end(); v++) {
            cout << *v << ", ";
        }
        cout << endl;
    }
}

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
    Mat<int> Y;
    dataToMatrix(enc, "outcome", X, Y);
    cout << "Data Matrices:" << endl;
    cout << "X:" << endl;
    X.print();
    cout << endl;
    cout << "Y:" << endl;
    Y.print();
    cout << endl;
}
