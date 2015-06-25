
#include "SVM.h"
#include "Feature.h"

int main(int argv, char* argc[]) {

	if (argv < 3) {
		std::cout << "Usage: ./svm [training_file] [c] [max_iter_cnt]" << std::endl;
		return 0;
	}

	std::ifstream fis(argc[1]);
	std::string line;
    std::vector<int32_t> vctLabel;
	std::vector<std::vector<float> > vctFeature;
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
    
	std::vector<float> vctTrainLabel(vctLabel.begin(), vctLabel.begin() + splitPos);
    std::vector<float> vctTestLabel(vctLabel.begin() + splitPos, vctLabel.end());
	
	
	float c = atof(argc[2]);
	int32_t max_iter_cnt = atoi(argc[3]);
	int32_t nor_type = atoi(argc[3]);

	suml::svm::SVM *svm = new suml::svm::SVM(c, max_iter_cnt);


	suml::feature::feature_normalize(nor_type, vctTrainFeature);

	svm->set_data(vctTrainFeature, vctTrainLabel);
	svm->init_alpha();
	svm->train(DEFAULT);
	
	suml::feature::feature_normalize(nor_type, vctTestFeature);

	
	int32_t tot = 0, right = 0;
	for (int32_t i = 0; i < vctTrainFeature.size(); i ++) {
		if (svm->predict(vctTrainFeature[i]) < 0 && vctTrainLabel[i] == 0 ||
				svm->predict(vctTrainFeature[i]) > 0 && vctTrainLabel[i] == 1) {
			right += 1;
		}
		tot += 1;
	}
	std::cout << right * 1.0 / tot << std::endl;
	tot = 0, right = 0;
	
	for (int32_t i = 0; i < vctTestFeature.size(); i ++) {
		
		std::cout << vctTestLabel[i] << " " << svm->predict(vctTestFeature[i]) << std::endl;
		if (svm->predict(vctTestFeature[i]) < 0 && vctTestLabel[i] == 0 ||
				svm->predict(vctTestFeature[i]) > 0 && vctTestLabel[i] == 1) {
			right += 1;
		}
		tot += 1;
	
	}
	std::cout << right * 1.0 / tot << std::endl;

	
	return 0;
}
