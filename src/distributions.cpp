// sample uniform(a, b)
// sample exponential distribution w/ L = ?
// sample normal distribution w/ Mu/S^2
#include "distributions.h"

using namespace std;

knuth_lcg rng{getSeed()};

// generate seed for rng via random_device
uint64_t getSeed() {
    random_device rdev;
    uint64_t seed = (uint64_t(rdev()) << 32) | rdev();
    return seed;
}

// density function of exponential distribution
float exponentialDensity(float x, float l) {
    return l * exp(-l * x);
}

// density function of normal distribution
float normalDensity(float x, float mu, float var) {
    return 1.0 / (2 * var * M_PI) * exp( pow(-(x - mu), 2) / (2 * var) );
}

float stdExponentialDensity(float x) {
    return exponentialDensity(x, 1);
}

float stdNormalDensity(float x) {
    return normalDensity(x, 0, 1);
}

// U ~ Unif(0, 1)
float sampleStdUniformDist() {
    float r = static_cast <float> (rng());
    float max = static_cast <float> (rng.max());
    float min = static_cast <float> (rng.min());
    return (r - min) / (max - min);
}

// X ~ Exponential(λ)
// Use the inverse method to sample exponential r.v.s 
// form uniform.
// X = - 1/λ * ln(U)
float sampleExponentialDist(float l) {
    float u = sampleStdUniformDist();
    return -1/l * log(u);
}

// X ~ N(0, 1)
// We use the acceptance-rejection method to generate normal r.v.s
// G ~ Exponential(1), g(x) = e^(-x)
// U ~ Unif(0, 1) <= f(x)/Cg(x)
// An upper bound C for f(x)/g(x) is 1/√(2*pi) * e^(1/2)
float sampleStdNormalDist() {
    float c = 1.0 / pow(2 * M_PI, 0.5) * exp(0.5);
    while (true) {
        float y = sampleExponentialDist(1);
        float u = sampleStdUniformDist();
        float fcg = stdExponentialDensity(y) / (c * stdNormalDensity(y));
        if (u <= fcg) {
            return y;
        }
    }
    return 0.0;
}

// X ~ N(m, v)
float sampleNormalDist(float mean, float variance) {
    return sampleStdNormalDist() * variance + mean;
}