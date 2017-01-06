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
#include "brief.h"

using namespace std;
using namespace arma;
using namespace Magick;

void drawKeypoints(Image& image, vector<Point> keypoints, string color) {
    for (auto p = keypoints.begin(); p != keypoints.end(); p++) {
        image.pixelColor((*p).x, (*p).y, color);
    }
}

void drawPatch(Image& image, Patch p, string id) {
    image.draw( DrawableLine(p.startx, p.starty, p.endx, p.starty) );
    image.draw( DrawableLine(p.startx, p.starty, p.startx, p.endy) );
    image.draw( DrawableLine(p.startx, p.endy, p.endx, p.endy) );
    image.draw( DrawableLine(p.endx, p.starty, p.endx, p.endy) );
    image.draw( DrawableText(p.startx + 10, p.starty + 10, id) );
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
    string testpng = selfdir+"data/test-imgs/apple-store.png";
    string testpgm = selfdir+"data/test-imgs/apple-store.pgm";

    Image image;
    image.read(testpng);
    image.strokeWidth(1);

    // load image into matrix 
    Mat<int> img; 
    img.load(testpgm, pgm_binary);

    // detect interest points
    int skip = 1;
    vector<Point> keypoints = detectKeypointsForImage(img, skip);
    cout << "detected " << keypoints.size() << " keypoints." << endl;

    // draw interest points
    drawKeypoints(image, keypoints, "green");

    // generate a descriptor for a patch
    int patchsz = 25;
    Patch p1(img, Point(50, 50), patchsz);
    Patch p2(img, Point(200, 200), patchsz);
    Patch p3(img, Point(201, 200), patchsz);
    
    // use the same local coordinates for each descriptor
    int descsz = 512;
    Point pt1;
    Point pt2;
    vector<Point> pairs;
    for (int i = 0; i < descsz; i++) {    
        sampleWithGaussianStrategy(img, p1, pt1, pt2);
        pairs.push_back( p1.tolocal(pt1) );
        pairs.push_back( p1.tolocal(pt2) );
    }

    vector<Point> descriptorPts;
    
    uint64_t d1 = generateBRIEFDescriptor(img, p1, descsz, pairs, descriptorPts);
    image.strokeColor("blue");
    drawPatch(image, p1, "1");
    drawKeypoints(image, descriptorPts, "red");

    uint64_t d2 = generateBRIEFDescriptor(img, p2, descsz, pairs, descriptorPts);
    image.strokeColor("blue");
    drawPatch(image, p2, "2");
    drawKeypoints(image, descriptorPts, "red");

    uint64_t d3 = generateBRIEFDescriptor(img, p3, descsz, pairs, descriptorPts);
    image.strokeColor("blue");
    drawPatch(image, p3, "3");
    drawKeypoints(image, descriptorPts, "red");

    cout << "patch 1 & patch 2 dist: " << (d1^d2) << endl;
    cout << "patch 1 & patch 3 dist: " << (d1^d3) << endl;
    cout << "patch 2 & patch 3 dist: " << (d2^d3) << endl;

    image.write(testpng + ".keypoints.png");
}