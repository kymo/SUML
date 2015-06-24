#ifndef __ClASSIFY_TREE_H__
#define __CLASSIFY_TREE_H__

#include "BaseTree.h"
#include "Feature.h"

namespace suml {
namespace tree {

class ClassificationTree : public suml::basic::Tree<float> {
public:

	ClassificationTree() {} 
    
	ClassificationTree(int32_t maxNodeCnt, 
			int32_t maxDepth, 
			bool isMultiThreadOn,
			int32_t labelCnt,
			bool ensemble) : suml::basic::Tree<float>(maxNodeCnt, maxDepth, isMultiThreadOn, labelCnt, ensemble) {}
	
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

	friend void* selectFeatureFuncC(void* param);

    float predict(const std::vector<float> &testFeatureX);

};

}
}

#endif
