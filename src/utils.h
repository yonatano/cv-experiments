#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>
#include <armadillo>

using namespace std;
using namespace arma;

map<string, vector<string> > loadCSV(string fileName);
map<string, vector<int> > labelEncodeData(map<string, vector<string> > dat);
void dataToMatrix(map<string, vector<int> > dat, string y, Mat<int>& X, Col<int>& Y);
void printConfusionMatrix(Col<int> Y, Col<int> Yp);
bool isInBounds(int w, int h, int x, int y);

template<class T>
inline
vector<T> uniqueElems(vector<T> v) {
    set<T> s(v.begin(), v.end());
    vector<T> vec(s.begin(), s.end());
    return vec;
}

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