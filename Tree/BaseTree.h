
// define for tree

#ifndef __BASETREE_H__
#define __BASETREE_H__

#include "Util.h"
#include "pthread.h"


namespace suml {
namespace basic {

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
};

template <class T>
class Tree { 
private:
    struct Node<T> *m_oTreeRootNode;
    int32_t m_nMaxNodeCnt;
    int32_t m_nMaxDepth;
    int32_t m_nMinSampleCnt;
    int32_t m_nLabelCnt;
    bool    m_bMultiThreadOn;
    bool    m_bEnsemble;
    int32_t m_nRandFeatureCnt;
    std::vector<std::vector<float> > m_vTrainingX;
    std::vector<T> m_vTrainingY;

public:
    std::vector<T>& getTrainingY();
    std::vector<std::vector<float> >& getTrainingX();
    int32_t &getMaxNodeCnt();
    int32_t &getMaxDepth();
    int32_t &getMinSampleCnt();
    int32_t &getRandFeatureCnt();
    int32_t &getLabelCnt();
    bool &getMultiThreadOn();
    bool &getEnsemble();
    
    struct Node<T>* &getTreeRootNode();
    
    Tree(){}
    
    Tree(int32_t maxNodeCnt, 
            int32_t maxDepth, 
            bool multiThreadOn,
            int32_t labelCnt,    
            bool ensemble) : 
        m_nMaxNodeCnt(maxNodeCnt), 
        m_nMaxDepth(maxDepth), 
        m_bMultiThreadOn(multiThreadOn),
        m_nLabelCnt(labelCnt),
        m_bEnsemble(ensemble) {
    }
    
    ~Tree() {}
    
    virtual void optSplitPos(int &nOptFeatureIndex,
                    float &nOptFeatureVal,
                    std::vector<int32_t> &vCurrentIndex,
                    std::vector<int32_t> &vFeatureIndex) {
    }    
    
    virtual void optSplitPosMultiThread(int &nOptFeatureIndex,
            float &nOptFeatureVal,
            std::vector<int32_t> &vCurrentIndex,
            std::vector<int32_t> &vFeatureIndex) {}

    virtual void splitData(Node<T>* &node,
        const int &nOptFeatureIndex,
        const T &fOptFeatureVal,
        const std::vector<int32_t> &vTempCurrentIndex,
           std::vector<int32_t> &vLeftIndex,
        std::vector<int32_t> &vRightIndex) {
    }
    virtual T predict(const std::vector<float> &testFeatureX) {}
    
    void sortIndexVec(std::vector<int32_t> &vCurrentIndex, const std::map<int32_t, float>& vFeatureValue);
    
    void buildTree(struct Node<T>* &oTreeNode,
                int32_t nCurNodeIndex,
                int32_t nCurNodeLevel,
                std::vector<int32_t> &vCurrentIndex,
                std::vector<int32_t> &vFeatureIndex) ;
    
    void setData(std::vector<std::vector<float> > &vTrainingX, 
            std::vector<T> &vTrainingY);

    void train();

    void display();

};

template <class T>
void Tree<T>::setData(std::vector<std::vector<float> > &vTrainingX,
        std::vector<T> &vTrainingY) {
    m_vTrainingY = vTrainingY;
    m_vTrainingX = vTrainingX;
    if (m_bEnsemble) {
        m_nRandFeatureCnt = 0.8 * m_vTrainingX[0].size();
    } else {
        m_nRandFeatureCnt = m_vTrainingX[0].size();
    }    
}

template <class T>
std::vector<T>& Tree<T>::getTrainingY(){
    return m_vTrainingY;
}

template <class T>
std::vector<std::vector<float> > & Tree<T>::getTrainingX() {
    return m_vTrainingX;
}

template <class T>
int32_t& Tree<T>::getMaxNodeCnt() {
    return m_nMaxNodeCnt;
}

template <class T>
int32_t& Tree<T>::getMaxDepth() {
    return m_nMaxDepth;
}

template <class T>
int32_t& Tree<T>::getMinSampleCnt() {
    return m_nMinSampleCnt;
}

template <class T>
int32_t& Tree<T>::getLabelCnt() {
    return m_nLabelCnt;
}

template <class T>
bool& Tree<T>::getMultiThreadOn() {
    return m_bMultiThreadOn;
}

template <class T>
bool& Tree<T>::getEnsemble() {
    return m_bEnsemble;
}

template <class T>
int32_t& Tree<T>::getRandFeatureCnt() {
    return m_nRandFeatureCnt;
}

template <class T>
struct Node<T>*& Tree<T>::getTreeRootNode() {
    return m_oTreeRootNode;
}

template <class T>
void Tree<T>::sortIndexVec(std::vector<int32_t> &vCurrentIndex, 
        const std::map<int32_t, float>& vFeatureValue) {
    int32_t start = 0, end = vCurrentIndex.size() - 1;
    suml::util::quick_sort(vCurrentIndex, vFeatureValue, start, end);
}

template <class T>
void Tree<T>::train() {
    std::vector<int32_t> vFeatureIndex, vCurrentIndex;
    for (int32_t i = 0; i < (int32_t)m_vTrainingX.size(); i ++) {
        vCurrentIndex.push_back(i);
    }
    for (int32_t i = 0; i < (int32_t)m_vTrainingX[0].size(); i ++) {
        vFeatureIndex.push_back(i);
    }
    
    buildTree(
                m_oTreeRootNode, 
                1, 
                1, 
                vCurrentIndex, 
                vFeatureIndex
    );
    display();
}

template <class T>
void Tree<T>::display() {
    
    if (NULL == m_oTreeRootNode) {
        std::cout << "Empty Tree! Are U Kidding Me?" << std::endl;
        return ;
    }
    std::queue<Node<T> *> treeNodeQueue;
    treeNodeQueue.push(m_oTreeRootNode);
    std::map<Node<T> *, int32_t> levelMap;
    levelMap[m_oTreeRootNode] = -1;
    int cnt = 0;
    while (! treeNodeQueue.empty()) {
        
        Node<T> *top = treeNodeQueue.front();
        treeNodeQueue.pop();

        std::cout << "Node: " << top->index << ",Father " << levelMap[top] << std::endl;
        std::cout << "Lev: " << top->level << std::endl;
        std::cout << "OptFeatureIndex: " << top->m_nCurrentOptSplitIndex << std::endl;
        std::cout << "CurrentOptSplitValue: " << top->m_fCurrentOptSplitValue << std::endl;
        std::cout << "Current Sample On this Node:" << std::endl;
        for (int32_t i = 0; i < top->m_vCurrentNodeTampleIndexVec.size(); i ++) {
            std::cout << top->m_vCurrentNodeTampleIndexVec[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Label: " << top->label << std::endl;

        if (top->m_oLeft) {
            levelMap[top->m_oLeft] = top->index;
            treeNodeQueue.push(top->m_oLeft);
        } 
        if (top->m_oRight) {
            levelMap[top->m_oRight] = top->index;
            treeNodeQueue.push(top->m_oRight);    
        }

    }

}

template <class T>
void Tree<T>::buildTree(struct Node<T>* &oTreeNode,
            int32_t curNodeIndex,
            int32_t curNodeLevel,
            std::vector<int32_t> &vCurrentIndex,
            std::vector<int32_t> &vFeatureIndex) {

    std::queue<Node<T> *> treeNodeQueues;
    
    oTreeNode = new Node<T>(vCurrentIndex, vFeatureIndex, 1, 1);
    treeNodeQueues.push(oTreeNode);
    
    int32_t index = 1;
    
    while (! treeNodeQueues.empty()) {
        
        Node<T> *top = treeNodeQueues.front();
        
        treeNodeQueues.pop();
        
        if (top->level > m_nMaxDepth) {
            continue;
        }
        if (top->index > m_nMaxNodeCnt) {
            continue;
        }
        
        std::vector<int32_t> vTmpFeatureIndex(top->m_vFeatureIndexVec.begin(), top->m_vFeatureIndexVec.end());
        std::vector<int32_t> vTmpCurrentIndex(top->m_vCurrentNodeTampleIndexVec.begin(), top->m_vCurrentNodeTampleIndexVec.end());
        
        if (vTmpCurrentIndex.size() == 0 || vTmpFeatureIndex.size() == 0) {
            continue;
        }
        
        int32_t nOptFeatureIndex;
        float   fOptFeatureVal;

        if (this->m_bMultiThreadOn) {
            
            optSplitPosMultiThread(nOptFeatureIndex, 
                        fOptFeatureVal, 
                        vTmpCurrentIndex,
                        vTmpFeatureIndex);
        
        } else {

            optSplitPos(nOptFeatureIndex, 
                    fOptFeatureVal, 
                    vTmpCurrentIndex,
                    vTmpFeatureIndex);    
        }
        std::cout << nOptFeatureIndex << " : " << fOptFeatureVal << std::endl;    
        std::vector<int32_t> vLeftIndex, vRightIndex;
        
        splitData(top, 
                nOptFeatureIndex, 
                fOptFeatureVal,
                vTmpCurrentIndex,
                vLeftIndex,
                vRightIndex);    

        // if (vLeftIndex.size() == 0 || vRightIndex.size() == 0) {
        //     continue ;
        // }
        
        // wipe out the feature index
        // if (! getEnsemble()) {
        //        vTmpFeatureIndex.erase(remove(vTmpFeatureIndex.begin(), vTmpFeatureIndex.end(), nOptFeatureIndex), 
        //            vTmpFeatureIndex.end());
        // }
        //
        // build left Node

#ifdef DEBUG
        std::cout << vLeftIndex.size() << " " << vRightIndex.size() << " ";
        std::cout << m_nMaxNodeCnt << " " << m_nMinSampleCnt << " " << " " << top->m_vCurrentNodeTampleIndexVec.size() << std::endl;
#endif
        
        if (vLeftIndex.size() > 0 
                && vTmpFeatureIndex.size() > 0
                && top->level + 1 < m_nMaxDepth
                && index < m_nMaxNodeCnt
                && top->m_vCurrentNodeTampleIndexVec.size() >= m_nMinSampleCnt) {
            top->m_oLeft = new Node<T>(vLeftIndex, vTmpFeatureIndex, top->level + 1, index);
            index += 1;
            treeNodeQueues.push(top->m_oLeft);
        
        }
        
        // build right Node
        if (vRightIndex.size() > 0 
                && vTmpFeatureIndex.size() > 0
                && top->level + 1 < m_nMaxDepth
                && index < m_nMaxNodeCnt
                && top->m_vCurrentNodeTampleIndexVec.size() >= m_nMinSampleCnt) {
            top->m_oRight = new Node<T>(vRightIndex, vTmpFeatureIndex, top->level + 1, index);
            index += 1;
            treeNodeQueues.push(top->m_oRight);
        }

        
    }
}

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
}
#endif
