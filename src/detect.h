#ifndef DETECT_H
#define DETECT_H

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>
#include <armadillo>

using namespace std;
using namespace arma;

bool isKeypoint(Row<int>& X);
int predictDescriptor(Row<int>& X);

#endif