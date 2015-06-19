#include "ANN.h" 
#include "Feature.h"

int main(int argv, char* argc[]) {

	suml::nn::ANN *ann = new suml::nn::ANN();

	if (argv < 3) {
		std::cout << "Usage: ./ANN [learning data] [learning_rate] [max iter cnt]  [hidden-level-cnt] [out-level-cnt] " << std::endl;
		exit(0);
	}


	std::ifstream fis(argc[1]);
	std::string line;
    std::vector<int32_t> vctLabel;
	std::vector<std::vector<float> > vctFeature;
	std::cout << "test error" << std::endl;
	while (getline(fis, line)) {
        std::vector<float> tempFeature;
		std::vector<std::string> vctSplitRes;
		suml::util::split(line, '\t', vctSplitRes);
		tempFeature.push_back(1.0);
		for (size_t i = 0; i < vctSplitRes.size(); i ++) {
            if (i + 1 < vctSplitRes.size()) {
                tempFeature.push_back(atof(vctSplitRes[i].c_str()));
            } else {
                vctLabel.push_back(atoi(vctSplitRes[i].c_str()));
            }
        }
        vctFeature.push_back(tempFeature);
    }
   	
	int sampleCnt = vctFeature.size(); 
	int splitPos = (int)(0.8 * sampleCnt);
    
	std::vector<std::vector<float> > vctTrainFeature(vctFeature.begin(), vctFeature.begin() + splitPos);
    
	std::vector<std::vector<float> > vctTestFeature(vctFeature.begin() + splitPos, vctFeature.end());
    
	std::vector<int32_t> vctTrainLabel(vctLabel.begin(), vctLabel.begin() + splitPos);
    
	std::vector<int32_t> vctTestLabel(vctLabel.begin() + splitPos, vctLabel.end());
	
	float learning_rate = atof(argc[2]);
	int32_t max_iter_cnt = atoi(argc[3]);
	int32_t hid_lev_cnt = atoi(argc[4]);
	int32_t out_lev_cnt = atoi(argc[5]);
	int32_t nor_type = atoi(argc[6]);

	suml::feature::feature_normalize(nor_type, vctTrainFeature);	
  	//ann->feature_normalize(MIN_MAX_NOR_TYPE, vctTestFeature);
	ann->set_data(vctTrainFeature, vctTrainLabel);
	ann->set_hid_lev_cnt(hid_lev_cnt);
	ann->set_learning_rate(learning_rate);
	ann->set_out_lev_cnt(out_lev_cnt);
	ann->set_max_iter_cnt(max_iter_cnt);

	ann->train(DEFAULT);

	int32_t tot = 0, right = 0;
	for (int32_t i = 0; i < vctTrainFeature.size(); i ++) {
		std::vector<float> ret;
		ann->predict(vctTrainFeature[i], ret);
		int tag = 0;
		for (int i = 0; i < out_lev_cnt; i ++) {
			std::cout << ret[i] << " ";
			if (ret[i] > 0.5) {
				tag = i;
			}
		}
		std::cout << std::endl;
		if (tag == vctTrainLabel[i]) {
			right += 1;
		}
		tot += 1;
	}
	
	std::cout << right * 1.0 / tot << std::endl;
	
	tot = 0, right = 0;

	for (int32_t i = 0; i < vctTestFeature.size(); i ++) {
		std::vector<float> ret;
		ann->predict(vctTestFeature[i], ret);
		int tag = 0;
		for (int i = 0; i < out_lev_cnt; i ++) {
			std::cout << ret[i] << " ";
			if (ret[i] > 0.5) {
				tag = i;
			}
		}
		std::cout << std::endl;
		if (tag == vctTestLabel[i]) {
			right += 1;
		}
		tot += 1;
	}
	std::cout << right * 1.0 / tot << std::endl;
	
	return 0;

}
