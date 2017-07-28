
#include "RegRF.h"


namespace suml {

namespace rf {

void RandomForestRegressor::build_tree(suml::basic::Tree<float> * & tree,
        std::vector<std::vector<float> > &feature,
        std::vector<float> &label) {

    tree = new suml::tree::RegressionTree(get_tree_depth(),get_tree_node_cnt(), get_multi_thread_on(), true);
    tree->getMinSampleCnt() = get_min_sample_cnt();
    tree->setData(feature, label);
    tree->train();
}

float RandomForestRegressor::f_predict(const std::vector<float> &feature) {

    float tot = 0.0;
    for (int32_t i = 0; i < get_tree_num(); ++i) {
        float label = get_trees()[i]->predict(feature);
        tot += label;
    }
    return tot / get_tree_num();
}

}
}

