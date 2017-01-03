#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <time.h>
#include <dirent.h>
#include <armadillo>
#include <Magick++.h>

#include "fast.h"
#include "decisiontree.h"
#include "utils.h"
#include "detect.h"
#include "distributions.h"

using namespace std;
using namespace arma;
using namespace Magick;

void saveImageWithKeypoints(string fileName, vector<Point> keypoints) {
    string saveName = fileName + ".keypoints.png";
    Image image;
    image.read(fileName);
    for (auto p = keypoints.begin(); p != keypoints.end(); p++) {
        image.pixelColor((*p).x, (*p).y, "green");
    }
    image.write(saveName);
}

vector<string> loadImageNet(int num) {
    vector<string> images;
    const char* dir = "data/tiny_imagenet_pgm/";
    DIR* dirData = opendir(dir);
    struct dirent* f;
    while (( f = readdir( dirData )) != NULL ) {
        if (num == 0)
            break;
        string fname(f->d_name);
        if (fname.find(".pgm") != string::npos) {
            images.push_back(string(dir) + fname);
            num--;
        }
    }
    cout << "read " << images.size() << " images" << endl;
    return images;
}

vector<Point> detectKeypointsForImage(Mat<int>& img, int skip) {
    vector<Point> keypoints;
    vector<Point> circle = computeCircleOfSize(0, 0, DEFAULT_CIRCLESZ);
    int startx = 4;
    int starty = 4;
    int endx = (img.n_cols - 1) - 4;
    int endy = (img.n_rows - 1) - 4;
    cout << "X: " << startx << " -> " << endx << endl;
    cout << "Y: " << starty << " -> " << endy << endl;
    for (int cy = starty; cy <= endy; cy += skip) {
        for (int cx = startx; cx <= endx; cx += skip) {
            int cmag = img(cy, cx);
            vector<int> relBrightness = relativeBrightnessForCircle(img, 
                                                                    cmag, 
                                                                    shiftPointCenter(circle, cx, cy), 
                                                                    DEFAULT_MAG_THRESHOLD);
            Row<int> X(relBrightness);
            if (isKeypoint(X)) {
                keypoints.push_back( Point(cx, cy) );
            }
        }
    }
    return keypoints;
}

void computeKeypointTrainingDataForImage(Mat<int> img,
                                         vector<Point> circle,
                                         vector<vector<int> >& relBrightnessVec,
                                         vector<bool>& isCornerVec) {
    int startx = 4; // radius size of 16px circle
    int starty = 4;
    int endx = (img.n_cols - 1) - 4;
    int endy = (img.n_rows - 1) - 4;
    for (int cy = starty; cy <= endy; cy++) {
        for (int cx = startx; cx <= endx; cx++) {
            int cmag = img(cy, cx);
            vector<int> relBrightness = relativeBrightnessForCircle(img, 
                                                        cmag, 
                                                        shiftPointCenter(circle, cx, cy), 
                                                        DEFAULT_MAG_THRESHOLD);
            bool isCorner = isCornerWithSegmentTestCriterion(img, 
                                                        cx, 
                                                        cy, 
                                                        shiftPointCenter(circle, cx, cy),
                                                        DEFAULT_PX_COUNT_REQ,
                                                        DEFAULT_MAG_THRESHOLD);
            relBrightnessVec.push_back(relBrightness);
            isCornerVec.push_back(isCorner);
        }
    }
}

void generateTrainingData(vector<string> imageFiles, string outName) {
    stringstream outData;

    for (int i = 0; i < DEFAULT_CIRCLESZ; i++) { // write csv header
        outData << "R" << i << ',';
    }
    outData << "Y" << '\n';

    vector<Point> circle = computeCircleOfSize(0, 0, DEFAULT_CIRCLESZ);

    // perform segment test over all pixels in each image
    for (auto f = imageFiles.begin(); f != imageFiles.end(); f++) { 
        Mat<int> img;
        vector<vector<int> > relBrightnessVec;
        vector<bool> isCornerVec;
        img.load(*f, pgm_binary);
        computeKeypointTrainingDataForImage(img,
                                            circle,
                                            relBrightnessVec,
                                            isCornerVec);
        for (int i = 0; i < isCornerVec.size(); i++) {
            for (auto rel = relBrightnessVec[i].begin(); 
                rel != relBrightnessVec[i].end(); rel++) {
                outData << *rel << ',';
            }
            outData << isCornerVec[i] << '\n';
        }
    }
    ofstream outFile;
    outFile.open(outName);
    outFile << outData.str();
    outFile.close();
}

int main(int argc, char **argv) {
    InitializeMagick(*argv);

    /*

    // generate training data
    cout << "loading ImageNet..." << endl;
    vector<string> images = loadImageNet(100);
    cout << "generating training data..." << endl;
    generateTrainingData(images, "train.csv");

    cout << "loading training data..." << endl;
    map<string, vector<int> > data = loadCSVAsInt("train.csv");
    map<string, vector<int> > train;
    map<string, vector<int> > test;
    splitTrainingData(data, train, test, data["Y"].size(), 0.8);
    cout << "loaded " << data["Y"].size() << " samples (" << train["Y"].size() << "/" << test["Y"].size() << " split)" << endl;

    // data -> matrix
    Mat<int> Xtrain;
    Col<int> Ytrain;
    Mat<int> Xtest;
    Col<int> Ytest;
    dataToMatrix(train, "Y", Xtrain, Ytrain);
    dataToMatrix(test, "Y", Xtest, Ytest);
    cout << "Xtrain: " << Xtrain.n_rows << "x" << Xtrain.n_cols << endl;
    cout << "Ytrain: " << Ytrain.n_rows << "x" << Ytrain.n_cols << endl;
    cout << "Xtest: " << Xtest.n_rows << "x" << Xtest.n_cols << endl;
    cout << "Ytest: " << Ytest.n_rows << "x" << Ytest.n_cols << endl;
    cout << endl;

    // train a corner classifier
    cout << "generating decision tree..." << endl;
    DecisionTree tree;
    tree.fitWithID3(Xtrain, Ytrain);

    // evaluate
    cout << "confusion matrix:" << endl;
    Col<int> Yp = tree.predict(Xtest);
    printConfusionMatrix(Ytest, Yp);
    cout << endl;

    // stringstream treeDump;
    // tree.dumpConditionals(treeDump);
    // ofstream treeFile;
    // treeFile.open("tree-dump.txt");
    // treeFile << treeDump.str();
    // treeFile.close();

    */
    string selfdir = "/Users/yonatanoren/Code/c++/projects/computervision/orb/";

    string testpng = selfdir+"data/test-imgs/skiing_large.png";
    string testpgm = selfdir+"data/test-imgs/skiing_large.pgm";

    Mat<int> img; // load image into matrix 
    img.load(testpgm, pgm_binary);
    cout << "loaded image: " << testpng << endl;
    vector<Point> keypoints = detectKeypointsForImage(img, 5);
    cout << "detected " << keypoints.size() << " keypoints." << endl;

    saveImageWithKeypoints(testpng, keypoints);


}