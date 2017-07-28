
/* test tree */

#include <time.h>
#include "ClaRF.h"

int main(int argv, char* argc[]) {

    std::cout << "Testing Classification Tree" << std::endl;
    std::cout << "Loading the tree" << std::endl;
    
    if (argv < 5) {
        std::cout << "Usage: ./cla_rf [training_file] [tree_cnt] [max_depth] [max_node_path] [min_sample_cnt] [multi thread or not] [label_cnt]" << std::endl;
        return 0;
    }

    // load feature
    int sampleCnt = 0, featureCnt = 0;
    int maxDepth, maxNodePath;
    std::ifstream fis(argc[1]);
    std::string line;
    std::vector<int32_t> vctLabel;
    std::vector<std::vector<float> > vctFeature;
    
    while (getline(fis, line)) {
        sampleCnt += 1;
        std::vector<float> tempFeature;
        std::vector<std::string> vctSplitRes;
        suml::util::split(line, '\t', vctSplitRes);
        for (size_t i = 0; i < vctSplitRes.size(); i ++) {
            if (i + 1 < vctSplitRes.size()) {
                tempFeature.push_back(atof(vctSplitRes[i].c_str()));
            } else {
                vctLabel.push_back(atof(vctSplitRes[i].c_str()));
            }
        }
        vctFeature.push_back(tempFeature);
    }
    int splitPos = (int)(0.8 * sampleCnt);
    std::vector<int32_t> vCurrentIndex;
    for (int32_t i = 0; i < splitPos; i ++) {
        vCurrentIndex.push_back(i);
    }
    std::vector<std::vector<float> > vctTrainFeature(vctFeature.begin(), vctFeature.begin() + splitPos);
    std::vector<std::vector<float> > vctTestFeature(vctFeature.begin() + splitPos, vctFeature.end());
    std::vector<float> vctTrainLabel(vctLabel.begin(), vctLabel.begin() + splitPos);
    std::vector<float> vctTestLabel(vctLabel.begin() + splitPos, vctLabel.end());
    
    if (vctFeature.size() <= 0) {
        std::cout << "Loading Feature Error!" << std::endl;
        return 0;
    }

    featureCnt = vctFeature[0].size();
    int32_t tree_cnt = atoi(argc[2]);
    maxNodePath = atoi(argc[3]);
    maxDepth = atoi(argc[4]);
    int32_t min_sample_cnt = atoi(argc[5]);
    bool isMultiThreadOn = atoi(argc[6]);
    int32_t label_cnt = atoi(argc[7]);

    // train the rf 

    suml::rf::RandomForest<float> *rf = new suml::rf::RandomForestClassifier(tree_cnt, maxNodePath, maxDepth, min_sample_cnt, label_cnt, isMultiThreadOn);


    suml::feature::feature_discretization(argc[8], vctTrainFeature);
    

    rf->set_data(vctTrainFeature, vctTrainLabel);
    
    clock_t start = clock();

    rf->train(DEFAULT);

    clock_t end = clock();

    std::cout << "Cost:" << end - start << std::endl;

    int right = 0, tot = 0;
    for (size_t i = 0; i < vctTrainFeature.size(); ++i) {
    
        if (vctTrainLabel[i] == rf->predict(vctTrainFeature[i])) {
            right += 1;
        }
        tot += 1;
    }
    std::cout << "Training Set Precision:" << right * 1.0 / tot << std::endl;
    right = 0, tot = 0;
    
    suml::feature::feature_discretization(argc[8], vctTestFeature);
    
    for (size_t i = 0; i < vctTestFeature.size(); ++i) {
        
        if (vctTestLabel[i] == rf->predict(vctTestFeature[i])) {
            right += 1;
        }
        tot += 1;
    }
    std::cout << "Testing Set Precision:" << right * 1.0 / tot << std::endl;

    return 0;
}
