//
// This file implements ID3 for generating a decision tree
//
#include "decisiontree.h"
#include "utils.h"

using namespace std;
using namespace arma;

DecisionTree::DecisionTree() {
    this->root = new Node();
}

void DecisionTree::fitWithID3(Mat<int> X, Col<int> Y) {
    vector<int> usedFtrs;
    ID3(X, Y, this->root, "", X.n_cols + 1, usedFtrs);
}

int DecisionTree::predict(Row<int> X) {
    return evaluate(X, this->root);
}

Col<int> DecisionTree::predict(Mat<int> X) {
    Col<int> Y(X.n_rows, 1);
    for (int i = 0; i < X.n_rows; i++) {
        Row<int> Xi = X.row(i);
        Y[i] = predict(Xi);
    }
    return Y;
}

void DecisionTree::print() {
    printTree(this->root, "");
}

void ID3(Mat<int> X, Col<int> Y, Node* root, string tabs, int maxDepth, vector<int> usedFtrs) {

    if (maxDepth == 0) {
        return;
    }

    // compute the entropy of outcomes
    double yEnt = computeEntropy(Y, unique(Y));
    if (is_zero(yEnt)) {    // all samples belong to the same class -- this branch 
                            // should result in a leaf-node
        Node *n = new Node();
        n->datum = Y[0];
        (root->children)->push_back(n);
        return;
    }

    // test all features to determine optimal split
    // select the split yielding lowest entropy
    int bestIdx = 0;
    double bestEnt = double(INT_MAX);
    for (int i = 0; i < X.n_cols; i++) {

        if (indexInVector(usedFtrs, i) != -1) { // don't re-split on used features
            continue;
        }

        // compute entropy over feature space of feature i
        double fEnt = 0.0;
        Col<int> c = X.col(i);
        Col<int> possVals = unique(c);
        for (int j = 0; j < possVals.size(); j++) { 
            // entropy of corresponding Y for each k in feature space
            Col<int> yElems = Y.elem( find(c == possVals[j]) );
            fEnt += computeEntropy(yElems, unique(yElems));
        }
        if (fEnt < bestEnt) {
            bestIdx = i;
            bestEnt = fEnt;
        }
    }
    Col<int> bestFtr = X.col(bestIdx);
    Col<int> bestFtrPoss = unique(bestFtr);
    usedFtrs.push_back(bestIdx);
    for (int j = 0; j < bestFtrPoss.size(); j++) {
        Node *n = new Node();
        int target = bestFtrPoss[j];
        n->ftrIdx = bestIdx;
        n->target = target;
        (root->children)->push_back(n);

        // recursively fill out the tree
        Mat<int> Xf = X.rows( find(bestFtr == bestFtrPoss[j]) );
        Col<int> Yf = Y.elem( find(bestFtr == bestFtrPoss[j]) );
        ID3(Xf, Yf, n, tabs + "\t", maxDepth - 1, usedFtrs);
    }
}

int evaluate(Row<int> X, Node* root) {
    vector<Node*> vec{};
    for (auto ch = root->children->begin(); ch != root->children->end(); ch++) {
        vec.push_back(*ch);
    }
    int curr = 0;
    while (true) {
        Node* n = vec[curr];
        if (n->isLeaf()) {
            return n->datum;
        }
        int ftrIdx = n->ftrIdx;
        int target = n->target;
        if (X[ftrIdx] == target) { 
            curr = vec.size() - 1; // no need to explore alternative values
            for (auto ch = n->children->begin(); ch != n->children->end(); ch++) {
                vec.push_back(*ch);
            }
        }
        curr++;
    }
    return -1;
}

void printTree(Node* root, string indent) {
    int rootsz = (root->children)->size();
    int halfidx = rootsz / 2;
    for (int i = 0; i < halfidx; i++) {
        printTree((*root->children)[i], indent+"\t\t");
    }
    cout << indent << *root << endl;
    for (int i = halfidx; i < rootsz; i++) {
        printTree((*root->children)[i], indent+"\t\t");
    }
}

double computeEntropy(Col<int> X, Col<int> possValues) {
    // - ∑ Pklog2(Pk) for k ∈ {X}
    double e = 0.0;
    for (int i = 0; i < possValues.size(); i++) {
        int k = uvec( find(X == possValues[i]) ).size();
        double p = float(k) / X.size();
        e += p * log2(p);
    }
    return -e;
}

bool is_zero(double X) {
    return abs(X - 0.0) < EPS;
}