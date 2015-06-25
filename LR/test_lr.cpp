#include "LR.h"
#include "Feature.h"

int main(int argv, char* argc[]) {

	suml::lr::LR *lr = new suml::lr::LR();

	if (argv < 3) {
		std::cout << "Usage: ./LR [training data] [learning_rate] [max iter cnt] [nor type] [opt type] " << std::endl;
		std::cout << "	nor type: 0-min_max  1-square normalization" << std::endl;
		std::cout << "	optimize type: 0-gradient decent 1-stochastic gradient descent" << std::endl;
		std::cout << "	regularization type: 0-l1 1-l2 " << std::endl;
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
	int32_t nor_type = atoi(argc[4]);
	int32_t opt_type = atoi(argc[5]);
	int32_t reg_type = -1;
	float lambda = 0.0;

	reg_type = atoi(argc[6]);
	lambda = atof(argc[7]);

	lr->set_parameter(max_iter_cnt, learning_rate, reg_type, lambda, 0);

	
  	

	
	lr->set_data(vctTrainFeature, vctTrainLabel);
	
	suml::feature::feature_normalize(nor_type, vctTrainFeature);	
	suml::feature::feature_normalize(nor_type, vctTestFeature);
	
	lr->train(opt_type);
	

	int32_t tot = 0, right = 0;
	for (int32_t i = 0; i < vctTrainFeature.size(); i ++) {
		std::cout << vctTrainLabel[i] << " " << lr->predict(vctTrainFeature[i]) << std::endl;
		if (lr->predict(vctTrainFeature[i]) < 0.5 && vctTrainLabel[i] == 0 ||
				lr->predict(vctTrainFeature[i]) > 0.5 && vctTrainLabel[i] == 1) {
			right += 1;
		}
		tot += 1;
	}
	std::cout << right * 1.0 / tot << std::endl;
	tot = 0, right = 0;
	
	for (int32_t i = 0; i < vctTestFeature.size(); i ++) {
		
		std::cout << vctTestLabel[i] << " " << lr->predict(vctTestFeature[i]) << std::endl;
		if (lr->predict(vctTestFeature[i]) < 0.5 && vctTestLabel[i] == 0 ||
				lr->predict(vctTestFeature[i]) > 0.5 && vctTestLabel[i] == 1) {
			right += 1;
		}
		tot += 1;
	
	}
	std::cout << right * 1.0 / tot << std::endl;

	
	if (argv == 7) {
		lr->dump_model(argc[6]);
	}
	return 0;
}
