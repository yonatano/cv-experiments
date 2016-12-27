#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

const double EPS = 1e-6;

typedef struct node {
    int datum;
    int ftrIdx;
    int target;
    vector<node*>* children;
    
    node() {
        this->datum = 0;
        this->ftrIdx = 0;
        this->target = 0;
        this->children = new vector<node*>();
    }

    bool isLeaf() {
        return this->children->size() == 0;
    }

    friend ostream &operator<<( ostream &output, const node n) {
        if (n.children->size() == 0) {
            output << n.datum;
        } else {
            output << "{f[" << n.ftrIdx << "]=" << n.target << "}";
        }
        return output;
    }
} Node;

class DecisionTree {
public:
    Node* root;
    void fitWithID3(Mat<int> X, Col<int> Y);
    int predict(Row<int> X);
    Col<int> predict(Mat<int> X);
    void print();
    DecisionTree();
};

void ID3(Mat<int> X, Col<int> Y, Node* root, string tabs, int maxDepth, vector<int> usedFtrs);
double computeEntropy(Col<int> X, Col<int> possValues);
bool is_zero(double X);
void printTree(Node* root, string indent);
int evaluate(Row<int> X, Node* root);

#endif
