
/* test tree */

#include "gbdt.h"
#include "gbdt.cpp"

int main(int argv, char* argc[]) {

    std::cout << "Testing Regression Tree" << std::endl;
    std::cout << "Loading the tree" << std::endl;
    if (argv < 6) {
        std::cout << "Usage: ./test_gbdt [training_file] [max_depth] [max_node_path] [tree_num] [min_sample_cnt] [learning rate] [multithread on]" << std::endl;
        return 0;
    }

    // load feature
    int sampleCnt = 0, featureCnt = 0;
    int maxDepth, maxNodePath;
	std::ifstream fis(argc[1]);
	std::string line;
    std::vector<float> vctLabel;
	std::vector<std::vector<float> > vctFeature;
	std::cout << "test error" << std::endl;
	while (getline(fis, line)) {
        sampleCnt += 1;
        std::vector<float> tempFeature;
		std::vector<std::string> vctSplitRes;
        util::split(line, '\t', vctSplitRes);
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
    maxNodePath = atoi(argc[2]);
    maxDepth = atoi(argc[3]);
	int32_t treeNum = atoi(argc[4]);
	int32_t minSampleCnt = atoi(argc[5]);
	float learningRate = atof(argc[6]);
	bool isMultiThreadOn = atoi(argc[7]);

    // train the regreTree 
	gbdt::GradientBoostingTree<float> *g = new gbdt::GradientBoostingRegressionTree<float>(treeNum, maxDepth, maxNodePath, minSampleCnt, learningRate, isMultiThreadOn);
	
	clock_t start = clock();
	g->train(vctTrainFeature, vctTrainLabel);
	clock_t end = clock();

	std::cout << "Cost:" << end - start << std::endl;
	// g->test();
	// test the tree
    for (size_t i = 0; i < vctTrainFeature.size(); ++i) {
        std::cout << i << " " << vctTrainLabel[i] << " vs " << g->predict(vctTrainFeature[i]) << std::endl;
    }
	/*
    for (size_t i = 0; i < vctTestFeature.size(); ++i) {
        std::cout << vctTestLabel[i] << " vs " << regreTree->predict(vctTestFeature[i]) << std::endl;
    }
	*/
    return 0;
}
