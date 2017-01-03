#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

#include <stdlib.h>
#include <math.h>
#include <random>

using namespace std;

typedef linear_congruential_engine<uint64_t, 6364136223846793005U, 1442695040888963407U, 0U> knuth_lcg;  /* Knuth's preferred 64-bit LCG */

uint64_t getSeed();
float sampleStdUniform();
float sampleExponentialDist(float l);
float sampleNormalDist(float mean, float variance);

#endif