#ifndef __RF_REG_H_
#define __RF_REG_H_

#include "BaseRandomForest.h"

namespace suml {
namespace rf {

class RandomForestRegressor : public suml::rf::RandomForest<float> {

public:
	
	RandomForestRegressor() {}
	RandomForestRegressor(int tree_cnt, int node_cnt, int depth, int min_sample_cnt, bool mutil_thread_on) : suml::rf::RandomForest<float>(tree_cnt, node_cnt, depth, min_sample_cnt, 0, mutil_thread_on) {}
	
	void build_tree(suml::basic::Tree<float>* &tree,
			std::vector<std::vector<float> > &feature,
			std::vector<float> &label);

	float forest_predict(const std::vector<float> &feature);
};

class RandomForestClassifier : public suml::rf::RandomForest<float> {
	
public:
	RandomForestClassifier() {}
	RandomForestClassifier(int tree_cnt, int node_cnt, int depth, int min_sample_cnt, int label_cnt, bool mutil_thread_on) : suml::rf::RandomForest<float>(tree_cnt, node_cnt, depth, min_sample_cnt, label_cnt, mutil_thread_on) {}
	
	void build_tree(suml::basic::Tree<float>* &tree,
			std::vector<std::vector<float> > &feature,
			std::vector<float> &label);

	float forest_predict(const std::vector<float> &feature);
};

}
}
#endif
