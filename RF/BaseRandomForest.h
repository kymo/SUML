#ifndef __BASE_RF_H_
#define __BASE_RF_H_

#include "Util.h"
#include "Model.h"
#include "BaseTree.h"
#include "RegTree.h"
#include "ClassifyTree.h"

namespace suml {
namespace rf {

template <class T>
class RandomForest : public suml::model::Model<T> {

private:
    std::vector<suml::basic::Tree<T>* > _trees;
    int32_t _label_cnt;
    int32_t _tree_num;
    int32_t _tree_depth;
    int32_t _tree_node_cnt;
    int32_t _min_sample_cnt;
    bool _multi_thread_on;

public:

    RandomForest(){}

    RandomForest(int tree_cnt, int node_cnt, int depth, int min_sample_cnt, int label_cnt, bool mutil_thread_on) : 
        _tree_num(tree_cnt), 
        _tree_node_cnt(node_cnt),
        _tree_depth(depth),
        _min_sample_cnt(min_sample_cnt),
        _label_cnt (label_cnt),
        _multi_thread_on(mutil_thread_on){
    
    }

    std::vector<suml::basic::Tree<T>* >& get_trees() {
        return _trees;
    }
    

    int32_t& get_label_cnt() {
        return _label_cnt;
    }

    int32_t& get_tree_num() {
        return _tree_num;
    }

    int32_t& get_tree_depth() {
        return _tree_depth;
    }

    int32_t &get_tree_node_cnt() {
        return _tree_node_cnt;
    }

    int32_t &get_min_sample_cnt() {
        return _min_sample_cnt;
    }
    
    bool& get_multi_thread_on() {
        return _multi_thread_on;
    }
    
    virtual void build_tree(suml::basic::Tree<T>* &tree,
            std::vector<std::vector<float> > &feature,
            std::vector<T> &label) {}

    virtual T f_predict(const std::vector<float> &feature) {
    }
    void train(int32_t opt_type);
    void load_model(const char* file_name);
    void dump_model(const char* file_name);
    T predict(const std::vector<float> &feature);
};

template <class T>
void RandomForest<T>::train(int32_t opt_type) {
    
    for (int32_t i = 0; i < _tree_num; ++i ) {
    
        //sample the data
        
        std::vector<int32_t> new_sample_index;
        for (int ins = 0; ins < this->_sample_size; ++ins) {
            new_sample_index.push_back(ins);
        }
        srand( (unsigned)time(NULL));
        for (int ins = 0; ins < this->_sample_size; ++ins ) {
            int new_ins = rand() % this->_sample_size;
            std::swap(new_sample_index[new_ins], new_sample_index[ins]);
        }

        std::vector<std::vector<float> > new_sample_feature;
        std::vector<T> new_sample_label;
        for (int ins = 0; ins < this->_sample_size * 0.8; ++ ins) {
            new_sample_feature.push_back(this->_feature[ins]);
            new_sample_label.push_back(this->_label[ins]);
        }
        
        suml::basic::Tree<float> *tree;
        build_tree(tree, new_sample_feature, new_sample_label);
        _trees.push_back(tree);

    }
}

template <class T>
void RandomForest<T>::load_model(const char* file_name) {}

template <class T>
void RandomForest<T>::dump_model(const char* file_name) {}

template <class T>
T RandomForest<T>::predict(const std::vector<float> &feature) {
    return f_predict(feature);
}

}
}
#endif
