
#include "utils.h"

using namespace std;
using namespace arma;

/* Data handling */

// loads CSV data into a map from column_names -> [data] given a fileName
map<string, vector<string> > loadCSVAsString(string fileName) {
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
    int step = 1000000;
    int lineNum = 0;
    for (string line; getline(file, line, '\n');) {
        ++lineNum;
        if (lineNum >= step){
            cout << "read 1M lines" << endl;
            step += 1000000;
        }

        stringstream ss(line);
        int idx = 0;
        for (string entry; getline(ss, entry, ',');) {
            data[columns[idx]].push_back(entry);
            idx++;
        }
    }
    return data;
}

map<string, vector<int> > loadCSVAsInt(string fileName) {
    map<string, vector<int> > datInt;
    map<string, vector<string> > dat = loadCSVAsString(fileName);
    for (auto m = dat.begin(); m != dat.end(); m++) {
        string k = m->first;
        vector<string> old = m->second;
        vector<int> new_;
        for (auto v = old.begin(); v != old.end(); v++) {
            new_.push_back(stoi(*v));
        } 
        datInt[k] = new_;
    }
    return datInt;
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
    conf.ones();

    for (int i = 0; i < Y.n_rows; i++) {
        int a = Y[i];
        int b = Yp[i];
        conf(a, b) += 1;
    }

    conf.print();
}

bool isInBounds(int w, int h, int x, int y) {
    return (0 <= x < w) && (0 <= y < h);
}

/* Image functions */
void smoothImageWithGaussian(Mat<int>& img) {
    // XXX: implement function to generate gaussian kernel
    // for now use weights for Sigma=2, window size 9x9
    mat k({0.319466, 0.361069, 0.319466});
    mat f = conv_to<mat>::from(img);
    f = conv2(f, k, "same");
    f = conv2(f, k.t(), "same");
    img = conv_to< Mat<int> >::from( f ); // convert to, from 2D convolution
}