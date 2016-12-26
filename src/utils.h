#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <armadillo>

using namespace std;
using namespace arma;

vector<string> uniqueElems(vector<string> v);
map<string, vector<string> > loadCSV(string fileName);
map<string, vector<int> > labelEncodeData(map<string, vector<string> > dat);
void dataToMatrix(map<string, vector<int> > dat, string y, Mat<int>& X, Col<int>& Y);

template<class T>
inline
int indexInVector(vector<T> v, T s) {
    int pos = find(v.begin(), v.end(), s) - v.begin();
    if (pos >= v.size()) {
        return -1;
    }
    return pos;
}

template<class T>
inline
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

#endif