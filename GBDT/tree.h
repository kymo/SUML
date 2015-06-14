
// define for tree

#ifndef __TREE_H__
#define __TREE_H__

#include "util.h"
#include "pthread.h"


namespace gbdt {


template <class T>
struct Node {

	std::vector<int32_t> m_vCurrentNodeTampleIndexVec;
	std::vector<int32_t> m_vFeatureIndexVec;
    
	int32_t m_nCurrentOptSplitIndex;
    float m_fCurrentOptSplitValue;
    int32_t index, level;
	struct Node *m_oLeft;
    struct Node *m_oRight;
    T label;
    
	Node() {}
	Node(std::vector<int32_t> currentIndexVec,
				std::vector<int32_t> featureIndexVec,
				int _level ,int _index): m_oLeft(NULL), m_oRight(NULL) {
		level = _level;
		index = _index;
		m_vFeatureIndexVec = featureIndexVec;
		m_vCurrentNodeTampleIndexVec = currentIndexVec;
	}
	/*
	Node(int nCurrentOptSplitIndex, float fCurrentOptSplitValue, std::vector<int32_t> vCurrentNodeTampleIndexVec, T _label) : m_oLeft(NULL), m_oRight(NULL) {
        m_nCurrentOptSplitIndex = nCurrentOptSplitIndex;
        m_fCurrentOptSplitValue = fCurrentOptSplitValue;
		m_vCurrentNodeTampleIndexVec = vCurrentNodeTampleIndexVec;
		label = _label;
	}
	*/
};

template <class T>
class Tree { 
public:
    struct Node<T> *m_oTreeRootNode;
    int32_t m_nMaxNodeCnt;
    int32_t m_nMaxDepth;
	bool	m_bMultiThreadOn;
	std::vector<std::vector<float> > m_vTrainingX;
	std::vector<T> m_vTrainingY;

	std::vector<T>& getTrainingY();
	std::vector<std::vector<float> >& getTrainingX();
	int32_t &getMaxNodeCnt();
	int32_t &getMaxDepth();
	struct Node<T>* &getTreeRootNode();

		
	
	Tree(){}
	Tree(int32_t maxNodeCnt, int32_t maxDepth, bool multiThreadOn) : m_nMaxNodeCnt(maxNodeCnt), m_nMaxDepth(maxDepth), m_bMultiThreadOn(multiThreadOn) {}
    ~Tree() {}
    

    // split the current sample in current Node with the best split index
    // of the feature, and get the index and the value, the function needs
    // to be implemented by child class(Regression & Classification)
    // arg:
    //      nOptFeatureIndex: save the optimized split index
    //      nOptFeatureVal: save the optimized split value
    //      vCurrentIndex: the indexes of samples in current Node
    //      vCurrentNodeTrainingX: the traning samples' feature vector
    //      vCurrentNodeTrainingY: the label of the training sample
    // return:
    //      void
    virtual void optSplitPos(int &nOptFeatureIndex,
                    float &nOptFeatureVal,
                    std::vector<int32_t> &vCurrentIndex,
					std::vector<int32_t> &vFeatureIndex) {}	
	virtual void optSplitPosMultiThread(int &nOptFeatureIndex,
			float &nOptFeatureVal,
			std::vector<int32_t> &vCurrentIndex,
			std::vector<int32_t> &vFeatureIndex) {}
	// build tree with depth-first algorithm with the limitation of max Node count
    // and max Node depth, this function needs to be implemented by the child class
    // argv:
    //      oTreeNode: current Node needs to be built
    //      nCurNodeIndex: the index of the current Node
    //      nCurNodeLevel: the level of the current Node
    //      vCurrentIndex: the indexes of the samples split on the current Node
    // return:
    //      void
    virtual void buildTree(struct Node<T>* &oTreeNode,
                int32_t nCurNodeIndex,
                int32_t nCurNodeLevel,
                std::vector<int32_t> &vCurrentIndex,
                std::vector<int32_t> &vFeatureIndex) {}
	
	// predict the value given the test feautre, the function needs to be implemented
    // by the  child class
    // arg:
    //      testFeatureX: the test feature
    // return:
    //      T
    virtual T predict(const std::vector<float> &testFeatureX) {}
	
	void setData(std::vector<std::vector<float> > &vTrainingX, 
			std::vector<T> &vTrainingY);

	
	void sort_index_vec(std::vector<int32_t> &sort_index,
		const std::map<int32_t, float> & sort_value);

	// train the model
	void train();
	void display();

};

template <class T>
class RegressionTree : public Tree<T> {
public:

    RegressionTree() {} 
    RegressionTree(int32_t maxNodeCnt, int32_t maxDepth, bool isMultiThreadOn) : Tree<T>(maxNodeCnt, maxDepth, isMultiThreadOn) {}
	
	void optSplitPos(int &nOptFeatureIndex,
                float &nOptFeatureVal,
                std::vector<int32_t> &vCurrentIndex,
                std::vector<int32_t> &vFeatureIndex); 
	
	void optSplitPosMultiThread(int &nOptFeatureIndex,
			float &nOptFeatureVal,
			std::vector<int32_t> &vCurrentIndex,
			std::vector<int32_t> &vFeatureIndex);

	friend void* selectFeatureFunc(void* param);

	void buildTree(struct Node<T>* &oTreeNode,
                int32_t nCurNodeIndex,
                int32_t nCurNodeLevel,
                std::vector<int32_t> &vCurrentIndex, 
                std::vector<int32_t> &vFeatureIndex);


    T predict(const std::vector<float> &testFeatureX);
};

template <class T>
class ClassificationTree : public Tree<T> {
public:
    ClassificationTree(int32_t maxNodeCnt, int32_t maxDepth) : Tree<T>(maxNodeCnt, maxDepth){}
    
	void optSplitPos(int &nOptFeatureIndex,
                    float &nOptFeatureVal,
                    std::vector<int32_t> &vCurrentIndex,
                    std::vector<int32_t> &vFeatureIndex);

    
	void buildTree(Node<T>* &oTreeNode);

    T predict(const std::vector<float> &testFeatureX);

};

template <class T>

class ThreadParam {
public:
	Tree<T>* m_oTree;
	std::vector<int32_t> m_vCurrentIndex;
	int32_t m_nFeatureIndex;

	ThreadParam() {}
	ThreadParam(Tree<T>* tree,
			std::vector<int32_t> vCurrentIndex, 
			int nFeatureIndex) {
		m_oTree = tree;
		m_vCurrentIndex = vCurrentIndex;
		m_nFeatureIndex = nFeatureIndex;
	}
};

}
#endif
