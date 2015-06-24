#ifndef __UTIL_H_
#define __UTIL_H_

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <queue>
#include <set>
#include <cmath>
#include <fstream>
#include <stdio.h>
#include <string.h>
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
#define LR_EPS 1.0e-4
#define SGD_EPS 1.0e-10

// NOR METHOD TYPE
#define DEFAULT -1
#define MIN_MAX_NOR_TYPE 0
#define SQUARE_NOR_TYPE 1

// regularization
#define REG_L1 0
#define REG_L2 1

namespace suml {
namespace util {

int rand_int(int num, int sed);

void split(const std::string &line, char tag, std::vector<std::string> &vctSplitRes) ;


float sigmoid(float val);

int32_t partition(std::vector<int32_t> &sort_index,
            const std::map<int32_t, float> &sort_value,
            int32_t start,
            int32_t end) ;

void quick_sort(std::vector<int32_t> &sort_index,
            const std::map<int32_t, float> &sort_value,
            int32_t start,
			int32_t end) ;
}
}
#endif
