//
// This file implements ID3 for generating a decision tree
//
#include "decisiontree.h"
#include <armadillo>

typedef struct node {
    std::vector<node*> children;
} Node;

class DecisionTree {
public:
    Node* root;
    void fitWithID3(Mat<int> X, Mat<int> Y, vector<int> possFtrs, vector<int> possOutcomes);
    int predict(Mat<int> X);
};

void DecisionTree::fitWithID3(Mat<int> X, Mat<int> Y, vector<int> possFtrs, vector<int> possOutcomes) {
    ID3(X, Y, possFtrs, possOutcomes, this->root);
}

void DecisionTree::predict(Mat<int> X) {
    return 0;
}

void ID3(Mat<int> X, Mat<int> Y, vector<int> possFtrs, vector<int> possOutcomes, Node& root) {

}

float computeEntropy(data y, int numOutcomes) {

}


// keep entries of X where Xi[key] == value
// XXX: switch to use std::copy_if
std::vector<data> filter(std::vector<data> x, int key, int value) {
    std::vector<data> v;
    for (data f : v) {
        if (f[key] == value)
            v.push_back(f);
    }
    return v;
}