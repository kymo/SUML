
#include "Util.h"

namespace suml {

namespace util {


int random_int(int n, int seed) {
    long long multiplier = 0x5DEECE66DL, mask = (1L << 48) - 1, addend = 0xBL;
    if (n <= 1) return 0;
    if ((n & -n) == n) {
        return (int) ((n * (long) ((int) ((seed = (seed * multiplier + addend) & mask) >> 17))) >> 31);
    }
    int bits, val;
    do {
        bits = (int) ((seed = (seed * multiplier + addend) & mask) >> 17);
        val = bits % n;
    } while (bits - val + (n - 1) < 0);
    return val;
}
void 
split(const std::string &line, char tag, std::vector<std::string> &vctSplitRes) {
    std::string tempStr = "";
    for (size_t i = 0; i < line.length(); i ++) {
        if (line[i] != tag) {
            tempStr += line[i];
        } else {
            vctSplitRes.push_back(tempStr);
            tempStr = "";
        }
    }
    if ("" != tempStr) {
        vctSplitRes.push_back(tempStr);
    }
}


float sigmoid(float x) {
    return 1.0 / (1 + exp(-x));
}

int32_t partition(std::vector<int32_t> &sort_index,
            const std::map<int32_t, float> &sort_value,
            int32_t start,
            int32_t end) {
    
    std::map<int32_t, float>::const_iterator ret = sort_value.find(sort_index[start]);
    
    float val = ret->second;
    int32_t i = sort_index[start];
    
    while (start < end) {
        while (start < end && end >= 0 && 
                sort_value.find(sort_index[end])->second >= val) {
            end --;
        }
        sort_index[start] = sort_index[end];
        while (start < end && sort_value.find(sort_index[start])->second <= val) {
            start ++;
        }
        sort_index[end] = sort_index[start];
    }
    sort_index[start] = i;
    return start;
}


void quick_sort(std::vector<int32_t> &sort_index,
            const std::map<int32_t, float> &sort_value,
            int32_t start,
            int32_t end) {
    if (start < end) {
        
        int32_t mid = partition(sort_index, sort_value, start, end);
        quick_sort(sort_index, sort_value, start, mid - 1);
        quick_sort(sort_index, sort_value, mid + 1, end);
    }
}

}

}
