#ifndef __REg_TREE_H__
#define __REG_TREE_H__

#include "BaseTree.h"

namespace suml {
namespace tree {

class RegressionTree : public suml::basic::Tree<float> {

public:
    
	RegressionTree() {} 
    
	RegressionTree(int32_t maxNodeCnt, 
			int32_t maxDepth, 
			bool isMultiThreadOn,
			bool ensemble) : suml::basic::Tree<float>(maxNodeCnt, maxDepth, isMultiThreadOn, 0, ensemble) {
		
		std::cout << "regression tree" << std::endl;
	
	}
	
	void optSplitPos(int &nOptFeatureIndex,
                float &nOptFeatureVal,
                std::vector<int32_t> &vCurrentIndex,
                std::vector<int32_t> &vFeatureIndex); 
	
	void optSplitPosMultiThread(int &nOptFeatureIndex,
			float &nOptFeatureVal,
			std::vector<int32_t> &vCurrentIndex,
			std::vector<int32_t> &vFeatureIndex);

	void splitData(struct suml::basic::Node<float>* &node,
		const int &nOptFeatureIndex,
		const float &fOptFeatureVal,
		const std::vector<int32_t> &vTempCurrentIndex,
   		std::vector<int32_t> &vLeftIndex,
		std::vector<int32_t> &vRightIndex);

	friend void* selectFeatureFunc(void* param);

    float predict(const std::vector<float> &testFeatureX);

};

}
}
#endif
