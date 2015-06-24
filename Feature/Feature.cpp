
#include "Feature.h"

namespace suml {
namespace feature {

void feature_normalize(int32_t normalize_type, 
		std::vector<std::vector<float> > &feature) {
	
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

void feature_discretization(const std::string &dis_primitive_str,
	std::vector<std::vector<float> > &feature) {
	// 1-7:10 means split the 1-7th feature into 10 segments, 1-7:10;2:5 means
   	// split the 1-7the feature into 10segments and split the 2th feature into 5th
	int sample_size = feature.size();	
	std::vector<std::string> dis_seg;
	suml::util::split(dis_primitive_str, ',', dis_seg);

	if (0 == (int32_t)dis_seg.size()) {
		std::cerr << "Discret the feature error: bad discret primitive string!" << std::endl;
		exit(0);
	}

	for (size_t i = 0; i < dis_seg.size(); ++i) {
		
		std::vector<std::string> dim_seg_vec;
		suml::util::split(dis_seg[i], ':', dim_seg_vec);
		
		if (2 != (int32_t)dim_seg_vec.size()) {
			std::cerr << "Discret the feature error: bad discret primitive sub string!" \
			   	<< std::endl;
			exit(0);
		}

		for (int32_t j = 0; j < dim_seg_vec[0].length(); ++j) {
			if (dim_seg_vec[0][j] >= '0' && dim_seg_vec[0][j] <= '9') {
				continue;
			}
			if (dim_seg_vec[0][j] != '-') {
				std::cerr << "Discret the feature error: bad discret primitive sub string!" \
					<< std::endl;
				exit(0);
			}
		}

		for (int32_t j = 0; j < dim_seg_vec[1].length(); ++j) {
			if (! (dim_seg_vec[1][j] >= '0' && dim_seg_vec[1][j])) {
				std::cerr << "Discret the feature error: bad discret primitive sub string!" \
					<< std::endl;
				exit(0);	
			}
		}
			
		bool is_continue_seg = (-1 != dim_seg_vec[0].find("-"));

		int32_t start, end;
		int32_t seg_cnt = atoi(dim_seg_vec[1].c_str());

		if (is_continue_seg) {
			std::vector<std::string> nums;
			suml::util::split(dim_seg_vec[0], '-', nums);
			if (2 != nums.size()) {
				std::cerr << "Discret the feature error: bad discret primitive sub string!" \
					<< std::endl;
				exit(0);	
			}
			start = atoi(nums[0].c_str());
			end = atoi(nums[1].c_str());
		} else {
			start = atoi(dim_seg_vec[0].c_str());
			end = start;
		}
		for (int32_t j = start; j <= end; ++j) {
			
			std::vector<float> feature_value_vec;
			for (int32_t ins = 0; ins < sample_size; ++ins) {
				feature_value_vec.push_back(feature[ins][j]);
			}
			
			int32_t avgsample_cnt = sample_size / seg_cnt;
			sort(feature_value_vec.begin(), feature_value_vec.end());
			for (int32_t ins = 0; ins < sample_size; ++ins) {
			
				for (int32_t seg_index = 0; seg_index < seg_cnt; ++seg_index) {
					float seg_start_val = feature_value_vec[seg_index * avgsample_cnt];
					float seg_end_val = feature_value_vec[(1 + seg_index) * avgsample_cnt - 1];					
					if (seg_start_val <= feature[ins][j] && seg_end_val >= feature[ins][j]) {
						feature[ins][j] = seg_index;
						break;	
					}
				}

			}
		}
	}
}
}
}
