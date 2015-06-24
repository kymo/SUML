#ifndef RF_H_
#define RF_H_

#include "Util.h"
#include "Model.h"

namespace suml {
namespace rf {

template <class T>
class RandomForest : public suml::model::Model<T> {

private:
	std::vector<suml::basic::Tree<T>* > _trees;

	int32_t _tree_num;
	int32_t _tree_depth;
	int32_t _tree_node_cnt;
	int32_t _min_sample_cnt;
	bool _multi_thread_on;

public:
	RandomForest(){}

	RandomForest(int tree_cnt, int node_cnt) : tree_cnt_(tree_cnt), node_cnt_(node_cnt){}
    
	virutal void add(CART *tree) {}

	virtual int size() {}

	void train();
	void load_model(const char* file_name);
	void dump_model(const char* file_name);
	T predict(const std::vector<float> &feature);
};

class RandomForestClassifier : public RandomForest<int32_t> {

publ ic:
	

};

}
}
#endif
