#ifndef __UTIL_H_
#define __UTIL_H_

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <set>
#include <cmath>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <limits.h>

#define DEBUG 1
#define EPS 1.0e-9

// optimize algorithm
#define GD 0
#define SGD 1

// eps value
#define LR_EPS 1.0e-2
#define SGD_EPS 1.0e-3

// NOR METHOD TYPE
#define MIN_MAX_NOR_TYPE 0
#define SQUARE_NOR_TYPE 1

namespace suml {
namespace util {

void split(const std::string &line, char tag, std::vector<std::string> &vctSplitRes) ;

}
}
#endif
