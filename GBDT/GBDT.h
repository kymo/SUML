// gbdt head file
// definition for gradient boosting decision  tree
// apply regression and classification

#ifndef __GBDT_H__
#define __GBDT_H__


#include "Model.h"
#include "RegTree.h"

namespace suml {
namespace gbdt{

class GradientBoostingRegressionTree : public suml::model::Model<float> {

public:
    std::vector<suml::tree::RegressionTree *> trees;
    
    // define for the model
    int32_t m_nTreeNum;                // number of trees
    int32_t m_nTreeDepth;                // max tree node
    int32_t m_nNodeCnt;                //     max node count
    
    int32_t m_nMinSampleCnt;        // minimum sample count in single node

    float m_fLearningRate;             // learning rate

    std::vector<float> m_vGradient;
    std::vector<float> m_vTempGradient;


    bool m_bMultiThreadOn;

    GradientBoostingRegressionTree() {}
    GradientBoostingRegressionTree(int32_t treeNum, 
            int32_t treeDepth, 
            int32_t nodeCnt, 
            int32_t minSampleCnt, 
            float learningRate, 
            bool isMultiThreadOn) :
        m_nTreeNum(treeNum),
        m_nTreeDepth(treeDepth),
        m_nNodeCnt(nodeCnt),
        m_nMinSampleCnt(minSampleCnt),
        m_fLearningRate(learningRate),
           m_bMultiThreadOn(isMultiThreadOn){}


    void train(int32_t opt_type);

    float predict(const std::vector<float> &feature);

    void load_model(const char* model_file_name);

    void dump_model(const char* model_file_name);

};
}

};

#endif
