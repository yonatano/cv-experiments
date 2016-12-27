
#include "utils.h"

using namespace std;
using namespace arma;

/* Util Functions */
vector<string> uniqueElems(vector<string> v) {
    set<string> s(v.begin(), v.end());
    vector<string> vec(s.begin(), s.end());
    return vec;
}

/* Data handling */
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
void dataToMatrix(map<string, vector<int> > dat, string y, Mat<int>& X, Col<int>& Y) {
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
    Y = col;
}

void printConfusionMatrix(Col<int> Y, Col<int> Yp) {
    Col<int> outcomes = unique(Y);
    int matsz = outcomes.n_rows;
    Mat<int> conf(matsz, matsz);
    for (int i = 0; i < matsz; i++) {
        for (int j = 0; j < matsz; j++) {
            conf(i,j) = uvec(find( Y == i && Yp == j )).n_rows;
        }
    }
    conf.print();
}