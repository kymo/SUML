
#include "ClaRF.h"


namespace suml {

namespace rf {

void RandomForestClassifier::build_tree(suml::basic::Tree<float>* &tree,
        std::vector<std::vector<float> > &feature,
        std::vector<float> &label) {

    tree = new suml::tree::ClassificationTree(get_tree_depth(),get_tree_node_cnt(), get_multi_thread_on(), get_label_cnt(), true);
    tree->getMinSampleCnt() = get_min_sample_cnt();
    tree->setData(feature, label);
    tree->train();
}

float RandomForestClassifier::f_predict(const std::vector<float> &feature) {
    
    std::map<int32_t, int32_t> label_cnt_map;
    int max_cnt = 0, pre_label;
    for (int32_t i = 0; i < get_tree_num(); ++i) {
        
        int32_t label = (int32_t)get_trees()[i]->predict(feature);
        if (label_cnt_map.find(label) == label_cnt_map.end()) {
            label_cnt_map[label] = 1;
        } else {
            label_cnt_map[label] += 1;
        }
        
        if (max_cnt < label_cnt_map[label]) {
            max_cnt = label_cnt_map[label];
            pre_label = label;
        }

    }
    
    return pre_label;

}

}
}

