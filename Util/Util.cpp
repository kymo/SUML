
#include "Util.h"

namespace suml {

namespace util {

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


}

}
