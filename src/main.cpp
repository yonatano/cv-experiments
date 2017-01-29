#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
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
#include "boosting.h"

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

int kNN(brief64& x, vector<brief64>& databaseX, vector<int>& databaseY, int k) {
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

Cube<float> generateNaiveBayesMatrix(vector<brief64> databaseX, vector<int> databaseY, int numClasses, int numFtrs, int ftrsz) {
    // For each bit in {0,1}^256, compute Pr[Xi=0,1 | Pos] and Pr[Xi=0,1 | Neg]
    
    // for each class Ck, we generate a numFtrs x ftrVals sized matrix
    // entry i,j contains an integer indicating the number of observed samples
    // in class Ck of feature i having value j
    Cube<int> classFtrCounts(numFtrs, ftrsz, numClasses, fill::zeros);
    Cube<float> classFtrProbabilities(numFtrs, ftrsz, numClasses, fill::zeros);
    for (int i = 0; i < databaseX.size(); i++) {
        brief64 x = databaseX[i];
        int y = databaseY[i];
        for (int idx = 0; idx < x.size(); idx++) {
            int bitatidx = x[idx];
            classFtrCounts(idx, bitatidx, y) += 1;
        }
    }

    // The probability of feature i taking on value j given class k is:
    // with M = classFtrMatrices[k]
    // M[i, j] / sum(M[i, :])
    for (int k = 0; k < numClasses; k++) {
        Mat<int> m = classFtrCounts.slice(k);
        Col<int> ftrTotals = sum(m, 1);
        for (int i = 0; i < numFtrs; i++) {
            for (int j = 0; j < ftrsz; j++) {
                classFtrProbabilities(i, j, k) = m(i, j) / float(ftrTotals[i]);
            }
        }
    }
    return classFtrProbabilities;
}

// y = argmax Pr[Ck] Π Pr[Xi | Ck]
int naiveBayesPrediction(brief64& x, Cube<float>& ftrProbabilities, float classProbabilities[]) {
    int label = 0;
    float score = 0.0;
    for (int k = 0; k < ftrProbabilities.n_slices; k++) {
        float classScore = classProbabilities[k];
        // iterate over the entries of x and compute a probability for class k
        //cout << "class " << k << " pr: [0.5*";
        for (int idx = 0; idx < x.size(); idx++) {
            int bit = x[idx];
            classScore *= ftrProbabilities(idx, bit, k); // Pr[Xidx = bit | Ck]
            //cout << ftrProbabilities(idx, bit, k) << "*";
        }
        //cout << "] = " << classScore << endl;
        if (classScore >= score) {
            score = classScore;
            label = k;
        }
    }
    return label;
}

template<size_t asz, size_t bsz>
void truncateBitset(bitset<asz>& a, bitset<bsz>& b) {
    for (int idx = 0; idx < bsz; idx++)
        b[idx] = a[idx];
}

void testScene() {

    string dir = "/Users/yonatanoren/Code/c++/projects/computervision/orb/";

    // points for display 
    vector<Point> pts;

    // initialize a sample patch for use with all descriptors
    Patch sample(Point(0, 0), Point(25 - 1, 25 - 1));

    /* Load an image of a scene */
    string imgv = dir+"data/BlurCar2/img/0001.jpg";
    string imgf = dir+"data/BlurCar2/pgm/0001.jpg.pgm";

    Image image;
    image.read(imgv);

    Mat<int> img;
    img.load(imgf, pgm_binary);

    // detect keypoints
    vector<Point> keypoints = detectKeypointsForImage(img, 1);
    cout << "detected " << keypoints.size() << " keypoints" << endl;
    drawKeypoints(image, keypoints, "red");

    // smoothImageWithGaussian(img); // smooth over for better performance

    // build the ensemble
    vector<Point> enspts;
    vector<BRIEFTestStub> ensemble;
    vector<float> learnerwt;

    ifstream file("ensemble.csv"); // x1, y1, x2, y2, wt 
    for (string line; getline(file, line, '\n');) {
        vector<string> entries;
        stringstream l(line);
        for (string entry; getline(l, entry, ',');) {
            entries.push_back(entry);
        }
        int x1 = stoi(entries[0]);
        int y1 = stoi(entries[1]);
        int x2 = stoi(entries[2]);
        int y2 = stoi(entries[3]);
        float wt = stof(entries[4]);

        Point p1(x1, y1);
        Point p2(x2, y2);
        enspts.push_back(p1);
        enspts.push_back(p2);
        ensemble.push_back( BRIEFTestStub(p1, p2) );
        learnerwt.push_back( wt );
        cout << "read: " << p1 << " " << p2 << " " << wt << endl;
    }

    // for each keypoint, extract the patch around it and classify
    for (auto kp = keypoints.begin(); kp != keypoints.end(); kp++) {
        Point c = *kp;
        Patch p(img, c, 25);

        // only consider patches where all points in-bound
        int invalid = 0;
        for (auto pti = enspts.begin(); pti != enspts.end(); pti++) {
            Point pt = *pti;
            if (!p.inBounds( p.fromlocal(pt) )) {
                invalid = 1;
                break;
            }
        }
        if (invalid == 1) { continue; }

        // draw
        vector<Point> classifypts;
        for (auto pti = enspts.begin(); pti != enspts.end(); pti++) {
            Point pt = *pti;
            classifypts.push_back( p.fromlocal(pt) );
        }
        drawKeypoints(image, classifypts, "blue");

        // classify
        Mat<int> sub = p.sub(img);
        int res = predictEnsemble(sub, ensemble, learnerwt);
    
        if (res == 1) {
            drawPatch(image, p, to_string(res));    
        }
        
    }
    drawKeypoints(image, keypoints, "red");
    image.write("test-car-detect.png");

}

int main(int argc, char **argv) {
    InitializeMagick(*argv);

    testScene();

    return 0;

    string dir = "/Users/yonatanoren/Code/c++/projects/computervision/orb/";

    /* Generate training data */
    stringstream outstream;

    // load each example and generate its BRIEF descriptor
    vector<string> pos = listFiles(dir+"data/test-cars-2/25/pospgm/", "pgm");
    vector<string> neg = listFiles(dir+"data/test-cars-2/25/negpgm/", "pgm");
    cout << "loaded, POS: " << pos.size() << " NEG: " << neg.size() << endl;

    vector<Mat<int> > databaseX;
    vector<int> databaseY;
    int counter = 0;
    int total = 0;
    for (auto fn = pos.begin(); fn < pos.end(); fn++) {
        Mat<int> img;
        img.load(*fn, pgm_binary);
        databaseX.push_back( img );

        if (counter >= 10000) {
            counter = 0;
            cout << "loaded " << total << "/" << pos.size() + neg.size() << endl;
        }
        counter++;
        total++;
    }
    for (auto fn = neg.begin(); fn < neg.end(); fn++) {
        Mat<int> img;
        img.load(*fn, pgm_binary);
        databaseX.push_back( img );

        if (counter >= 10000) {
            counter = 0;
            cout << "loaded " << total << "/" << pos.size() + neg.size() << endl;
        }
        counter++;
        total++;
    }
    for (int i = 0; i < pos.size(); i++) { databaseY.push_back(1); }
    for (int i = 0; i < neg.size(); i++) { databaseY.push_back(0); }

    int numSamples = databaseX.size();
    cout << "loaded " << numSamples << " samples" << endl;

    // randomize
    vector<int> randidxs;
    for (int i = 0; i < numSamples; i++) { randidxs.push_back(i); }
    random_shuffle(randidxs.begin(), randidxs.end());
    
    for (int idx = 0; idx < randidxs.size(); idx++) {
        databaseX.push_back( databaseX[randidxs[idx]] );
        databaseY.push_back( databaseY[randidxs[idx]] );
    }
    databaseX.erase(databaseX.begin(), databaseX.begin()+numSamples);
    databaseY.erase(databaseY.begin(), databaseY.begin()+numSamples);

    cout << "randomized datasets" << endl;

    // points vector for displaying on images
    vector<Point> pts;

    /* generate BRIEF descriptors */
    const int BRIEFSZ = 64;
    vector<brief64> BRIEFX;

    // initialize a sample 25x25 patch for use with all descriptors
    Patch sample(Point(0, 0), Point(25 - 1, 25 - 1));

    // use the same local coordinates for each descriptor
    Point pt1;
    Point pt2;
    vector<Point> pairs;
    for (int i = 0; i < BRIEFSZ; i++) {
        sampleWithGaussianStrategy(sample, pt1, pt2);
        pairs.push_back( sample.tolocal(pt1) );
        pairs.push_back( sample.tolocal(pt2) );
    }
    for (int i = 0; i < databaseX.size(); i++) {
        Mat<int> img(databaseX[i]); // copy so we can smooth
        brief64 desc;
        brief512 d = generateBRIEFDescriptor(img, sample, pairs, pts);
        truncateBitset(d, desc);
        BRIEFX.push_back( desc );
    }

    /* generate LBPs */
    const int LBPSZ = 12;
    const int CIRCSZ = 4;
    vector<bitset<LBPSZ> > LBPX;
    for (int i = 0; i < databaseX.size(); i++) {
        Mat<int> img(databaseX[i]);
        smoothImageWithGaussian(img);

        bitset<LBPSZ> lbp;
        int cmag = img(sample.center.y, sample.center.x);
        vector<Point> circle = computeCircle(sample.center.x, sample.center.y, CIRCSZ);
        vector<int> relBrightness = relativeBrightnessForCircle(img,
                                                                cmag,
                                                                circle, 
                                                                DEFAULT_MAG_THRESHOLD);
        for (int idx = 0; idx < LBPSZ; idx++) {
            lbp[idx] = relBrightness[idx];
        }
        LBPX.push_back( lbp );
    }
    
    auto dataX = BRIEFX; // which descriptor type to use

    /* train-test split */
    vector<brief64> Xtrain;
    vector<brief64> Xtest;
    vector<int> Ytrain;
    vector<int> Ytest;
    random_shuffle(randidxs.begin(), randidxs.end());

    int splitidx = numSamples * 0.8;
    for (int i = 0; i < splitidx; i++) {
        Xtrain.push_back( dataX[randidxs[i]] );
        Ytrain.push_back( databaseY[randidxs[i]] );
    }
    for (int i = splitidx; i < numSamples; i++) {
        Xtest.push_back( dataX[randidxs[i]] );
        Ytest.push_back( databaseY[randidxs[i]] );
    }
    cout << "Train/Test: " << Xtrain.size() << "/" << Xtest.size() << endl;

    /* Generate Parameters for Naive Bayes Classifier */
    const int numClasses = 2; // pos or neg
    const int numFtrs = BRIEFSZ; // each bit is a feature
    const int ftrsz = 2; // each bit element of {0, 1}
    float classProbabilities[] = { 
        pos.size() / float(pos.size() + neg.size()), 
        neg.size() / float(pos.size() + neg.size())};
    Cube<float> classFtrProbabilities = generateNaiveBayesMatrix(Xtrain, 
                                                                 Ytrain, 
                                                                 numClasses,
                                                                 numFtrs, 
                                                                 ftrsz);


    /* Generate Parameters for kNN Classifier */
    const int k = 1;
    const int knnsz = 1000;
    vector<brief64> knnx(Xtrain.begin(), Xtrain.begin()+knnsz);
    vector<int> knny(Ytrain.begin(), Ytrain.begin()+knnsz);

    cout << "adaboost..." << endl;
    /* Use AdaBoost to generate an ensemble for classification */
    // we turn each BRIEF point pair into a weak learner
    int N = 1000;     // # of weak learners to pre-generate
    int iters = 200;  // # of boosting iterations / ensemble size
    vector<BRIEFTestStub> learners;
    vector<float> samplewt;
    vector<float> learnerwt;
    vector<int> ensembleidx;

    for (int i = 0; i < numSamples; i++) { samplewt.push_back(1); }

    // generate weak learners
    Point p1;
    Point p2;
    Patch p(Point(0, 0), Point(24, 24));
    for (int i = 0; i < N; i++) {
        sampleWithGaussianStrategy(p, p1, p2);
        learners.push_back( BRIEFTestStub(p1, p2) );
    }

    stringstream ss;

    // Boosting procedure
    for (int i = 0; i < iters; i++) {
        cout << "STEP: " << i << endl;
        AdaBoost(databaseX, databaseY, learners, samplewt, learnerwt, ensembleidx);

        cout << "learner weights: ";
        for (auto li = learnerwt.begin(); li != learnerwt.end(); li++) {
            cout << *li << ",";
        }
        cout << endl;

        // evaluate ensemble
        int correct = 0;
        vector<BRIEFTestStub> ensemble;
        for (int idx = 0; idx < ensembleidx.size(); idx++) {
            ensemble.push_back( learners[ensembleidx[idx]] );
        }
        for (int idx = 0; idx < numSamples; idx++) {
            int pred = predictEnsemble(databaseX[idx], ensemble, learnerwt);
            if (pred == databaseY[idx]) {
                correct++;
            }
        }
        float acc = correct / float(numSamples);
        cout << "EVAL: " << acc << endl;
        cout << endl;

        if (i % 10 == 0) {
            // save points and learner weights
            ss << "ITERATION: " << i << " ACC: " << acc << '\n';
            for (int idx = 0; idx < ensemble.size(); idx++) {
                BRIEFTestStub ftr = learners[ensembleidx[idx]];
                ss << ftr.p1 << "," << ftr.p2 << "," << learnerwt[idx] << '\n';
            }
            ss << '\n';
            ofstream outFile;
            outFile.open(dir+"boosting_weights.txt");
            outFile << ss.str();
            outFile.close();
            ss.clear();
        }
    }



    // /* Evaluate */
    // cout << "classifying..." << endl;
    // Col<int> Y(Ytest);
    // Col<int> Yp(Ytest.size());
    // for (int i = 0; i < Xtest.size(); i++) {
    //     Yp[i] = naiveBayesPrediction(Xtest[i], classFtrProbabilities, classProbabilities);
    // }
    // cout << "Accuracy: " << computeAccuracy(Ytest, Yp) << endl;
    // cout << "F1-Score: " << computeF1Score(Ytest, Yp) << endl;
    // cout << "confusion matrix:" << endl;
    // printConfusionMatrix(Ytest, Yp);
    // cout << endl;    
}