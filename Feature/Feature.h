
#ifndef __FEATURE_H_
#define __FEATURE_H_

#include "Util.h"

namespace suml {
namespace feature {

void feature_normalize(int32_t normalize_type,
		std::vector<std::vector<float> > &feature);

void feature_select();

void k_fold_validation(int k);	

void feature_discretization(const std::string& dis_primitive_str, 
		std::vector<std::vector<float> > &feature);

}

}

#endif
