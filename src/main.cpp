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

vector<string> listFiles(string dir, string ext) {
    vector<string> files;
    const char* cdir = dir.c_str();
    DIR* dirData = opendir(cdir);
    struct dirent* f;
    while (( f = readdir( dirData )) != NULL ) {
        string fname(f->d_name);
        if (fname.find(ext) != string::npos) {
            files.push_back(string(dir) + fname);
        }
    }
    return files;
}

vector<string> loadImageNet() {
    return listFiles("data/tiny_imagenet_pgm/", "pgm");
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

void generateCornerTrainingData(vector<string> imageFiles, string outName) {
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

void trainCornerDetector() {
     // generate training data
    cout << "loading ImageNet..." << endl;
    vector<string> images = loadImageNet();
    cout << "generating training data..." << endl;
    generateCornerTrainingData(images, "train.csv");

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
}

int kNN(brief256& x, vector<brief256>& databaseX, vector<int>& databaseY, int k) {
    // indices of databaseX sorted by distances to x
    vector<size_t> dists(databaseX.size());
    for (int i = 0; i < databaseX.size(); i++) {
        dists[i] = (databaseX[i]^x).count();
    }
    vector<size_t> distidx(databaseX.size());
    iota(distidx.begin(), distidx.end(), static_cast<size_t>(0));
    sort(distidx.begin(), distidx.end(), [&](size_t a, size_t b) { 
        return dists[a] < dists[b];
    });

    vector<int> predictions;
    for (int i = 0; i < k; i++) {
        predictions.push_back( databaseY[distidx[i]] );
        // cout << " " << distidx[i] << "/" << (databaseX[distidx[i]]^x).count() << "/" << databaseY[distidx[i]];
    }
    // cout << endl;

    int streakVal = 0;
    int streakLen = 0;
    for (int i = 0; i < predictions.size(); i++) {
        int n = count(predictions.begin(), predictions.end(), predictions[i]);
        // cout << "count (" << predictions[i] << "): " << n << endl;
        if (n > streakLen) {
            streakVal = predictions[i];
            streakLen = n;
        }
    }
    // cout << " = " << streakVal << endl;
    return streakVal;
}

int main(int argc, char **argv) {
    InitializeMagick(*argv);

    string dir = "/Users/yonatanoren/Code/c++/projects/computervision/orb/";

    /* Generate training data */
    stringstream outstream;

    // load each example and generate its BRIEF descriptor
    vector<string> pos = listFiles(dir+"data/test-cars/pospgm/", "pgm");
    vector<string> neg = listFiles(dir+"data/test-cars/negpgm/", "pgm");
    cout << "loaded, POS: " << pos.size() << " NEG: " << neg.size() << endl;

    Mat<int> img;
    vector<Point> pts;
    
    // initialize a sample patch for use with all descriptors
    int patchsz = 25;
    Patch sample(Point(0, 0), Point(patchsz-1, patchsz-1));
    
    // use the same local coordinates for each descriptor
    int descsz = 64;
    Point pt1;
    Point pt2;
    brief256 desc;
    vector<Point> pairs;
    for (int i = 0; i < descsz; i++) {
        sampleWithGaussianStrategy(sample, pt1, pt2);
        pairs.push_back( sample.tolocal(pt1) );
        pairs.push_back( sample.tolocal(pt2) );
    }

    // generate examples
    vector<brief256> databaseX;
    vector<int> databaseY;

    cout << "generate positive descriptors" << endl;
    for (int i = 0; i < pos.size(); i++) {
        img.load(pos[i], pgm_binary);
        desc = generateBRIEFDescriptor(img, sample, pairs, pts);
        databaseX.push_back( desc );
        databaseY.push_back( 1 );
    }

    cout << "generate negative descriptors" << endl;
    for (int i = 0; i < neg.size(); i++) {
        img.load(pos[i], pgm_binary);
        desc = generateBRIEFDescriptor(img, sample, pairs, pts);
        databaseX.push_back( desc );
        databaseY.push_back( 0 );
    }

    // train-test split
    vector<brief256> Xtrain;
    vector<brief256> Xtest;
    vector<int> Ytrain;
    vector<int> Ytest;

    vector<int> randidxs;
    for (int i = 0; i < databaseX.size(); i++) { randidxs.push_back(i); }
    random_shuffle(randidxs.begin(), randidxs.end());

    int splitidx = databaseX.size() * 0.8;
    for (int i = 0; i < splitidx; i++) {
        Xtrain.push_back( databaseX[randidxs[i]] );
        Ytrain.push_back( databaseY[randidxs[i]] );
    }
    for (int i = splitidx; i < databaseX.size(); i++) {
        Xtest.push_back( databaseX[randidxs[i]] );
        Ytest.push_back( databaseY[randidxs[i]] );
    }
    
    // evaluate
    int curr = 0;
    Col<int> Y(Ytest);
    Col<int> Yp(Ytest.size());
    for (int i = 0; i < Xtest.size(); i++) {
        int p = kNN(Xtest[i], Xtrain, Ytrain, 5);
        Yp[i] = p;
        curr++;
        if (curr > 1000) {
            curr = 0;
            cout << "classified 1000" << endl;
        }
    }
    cout << "confusion matrix:" << endl;
    printConfusionMatrix(Ytest, Yp);
    cout << endl;

}