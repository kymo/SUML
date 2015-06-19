// gbdt head file
// definition for gradient boosting decision  tree
// apply regression and classification

#ifndef __GBDT_H__
#define __GBDT_H__

#include "tree.h"
#include "tree.cpp"


namespace gbdt{

template <class T>

class GradientBoostingTree {

public:
    std::vector<Tree<T> *> trees;
    
	// define for the model
    int32_t m_nTreeNum;                // number of trees
    int32_t m_nTreeDepth;				// max tree node
	int32_t m_nNodeCnt;				//	 max node count
	int32_t m_nMinSampleCnt;		// minimum sample count in single node

	float m_fLearningRate;             // learning rate

	std::vector<T> m_vGradient;
	std::vector<T> m_vTempGradient;


	bool m_bMultiThreadOn;

    GradientBoostingTree() {}
	GradientBoostingTree(int32_t treeNum, int32_t treeDepth, int32_t nodeCnt, int32_t minSampleCnt, float learningRate, bool isMultiThreadOn) :
		m_nTreeNum(treeNum),
		m_nTreeDepth(treeDepth),
		m_nNodeCnt(nodeCnt),
		m_nMinSampleCnt(minSampleCnt),
		m_fLearningRate(learningRate),
   		m_bMultiThreadOn(isMultiThreadOn){}


    virtual void train(std::vector<std::vector<float> > &trainingX,
		std::vector<T> &trainingY) {}
    virtual T predict(const std::vector<T> &feature){}
};

template <class T>
class GradientBoostingRegressionTree : public GradientBoostingTree<T> {
public:
    GradientBoostingRegressionTree(){}
	GradientBoostingRegressionTree(int32_t treeNum, int32_t treeDepth, int32_t nodeCnt, int32_t minSampleCnt, float fLearningRate, bool isMultiThreadOn) : GradientBoostingTree<T>(treeNum, treeDepth, nodeCnt, minSampleCnt, fLearningRate, isMultiThreadOn) {}
	
	void train(std::vector<std::vector<float> > &trainingX, std::vector<T> &trainingY);
	T predict(const std::vector<T> &feature);

};

template <class T>
class GradientBoostingClassificationTree {
public:
    GradientBoostingClassificationTree() {}
    void train();
	T predict(const std::vector<float> &feature);
};

};

#endif
