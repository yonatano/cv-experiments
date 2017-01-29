#include "boosting.h"

void AdaBoost(vector<Mat<int> >& dataX,
              vector<int>& dataY, 
              vector<BRIEFTestStub>& learners, 
              vector<float>& samplewt, 
              vector<float>& learnerwt,
              vector<int>& ensemble) {
    int numSamples = dataX.size();
    int numLearners = learners.size();
    vector<float> sampledist(numSamples); // compute distribution over samples
    float sumwt = 0.0;
    for (int i = 0; i < samplewt.size(); i++) { sumwt += samplewt[i]; }
    for (int i = 0; i < samplewt.size(); i++) {
        sampledist[i] = samplewt[i] / sumwt;
    }

    int bestidx; // score all WLs
    float besterr = numeric_limits<float>::infinity();
    for (int lidx = 0; lidx < numLearners; lidx++) { 
        auto wl = learners[lidx];
        float err = 0.0;
        for (int si = 0; si < numSamples; si++) {
            err += sampledist[si] * abs(wl.predict(dataX[si]) - dataY[si]);
        }
        if (err < besterr) {
            bestidx = lidx;
            besterr = err;
        }
        // cout << "Scoring WL (" << lidx << ") [" << wl.p1 << " , " << wl.p2 << "]: " << err << endl;
    }
    cout << "Winner: " << bestidx << " Err: " << besterr << endl;

    auto wl = learners[bestidx];
    float bt = besterr / (1 - besterr);
    for (int wi = 0; wi < numSamples; wi++) { // update sample weights
        int ei = (wl.predict(dataX[wi]) == dataY[wi]) ? 0 : 1;
        samplewt[wi] = samplewt[wi] * pow(bt, 1 - ei);
    }

    ensemble.push_back( bestidx );
    learnerwt.push_back( log( 1 / bt) );
}

int predictEnsemble(Mat<int>& x,
                    vector<BRIEFTestStub>& ensemble, 
                    vector<float>& learnerwt) {
    float wtsum = 0.0;
    float prsum = 0.0;
    for (int idx = 0; idx < ensemble.size(); idx++) {
        auto wl = ensemble[idx];
        wtsum += learnerwt[idx];
        prsum += learnerwt[idx] * wl.predict(x);
    }
    return (prsum >= 0.5 * wtsum) ? 1 : 0;
}



