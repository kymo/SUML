
#include "Model.h"



namespace suml {
namespace model {


void Model::set_data(std::vector<std::vector<float> > &feature,
		std::vector<int32_t> &label) {
	_feature = feature;
	_label = label;
	_sample_size = (int32_t)feature.size();
	if (0 == _sample_size) {
		std::cerr << "Error When Setting the Training Sample!" << std::endl;
		exit(0);
	}
	_feature_dim = (int32_t)feature[0].size();

}

void Model::feature_normalize(int32_t normalize_type, std::vector<std::vector<float> > &feature) {
	int sample_size = feature.size();
	int feature_dim = feature[0].size();
	if (normalize_type == MIN_MAX_NOR_TYPE) {
		for (int32_t i = 0; i < feature_dim; ++i) {
			float minfeature_value = INT_MAX;
			float maxfeature_value = - INT_MAX;
			for (int32_t j = 0; j < sample_size; ++ j) {
				if (minfeature_value > feature[j][i]) {
					minfeature_value = feature[j][i];
				} 
				if (maxfeature_value < feature[j][i]) {
					maxfeature_value = feature[j][i];
				}
			}

			for (int32_t j = 0; j < sample_size; j ++) {
				if (maxfeature_value == minfeature_value) {
					feature[j][i] = 1.0;
				} else {
					feature[j][i] = (feature[j][i] - minfeature_value) / (maxfeature_value - minfeature_value);
				}
				
			}
		}	
	} else if (normalize_type == SQUARE_NOR_TYPE) {
		for (int32_t i = 0; i < sample_size; i ++) {
			float tot_val = 0.0;
			for (int32_t j = 0; j < feature_dim; j ++) {
				tot_val += feature[i][j] * feature[i][j];
			}
			for (int32_t j = 0; j < feature_dim; j ++) {
				feature[i][j] /= sqrt(tot_val);
			}
		}
	}
}

void Model::feature_select() {
}

void Model::k_fold_validation(int k) {
}

}
}
