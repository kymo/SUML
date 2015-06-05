
#ifndef __LR_H_
#define __LR_H_

#include "Model.h"

namespace suml {
namespace lr {

class LR : public suml::model::Model {

public:
	int32_t _max_iter_cnt;
	float	_learning_rate;
	int32_t _thread_cnt;

	std::vector<float> _w;

	LR() {}
	virtual ~LR() {}

	float calc_likely_hood();

	bool stochatic_gradient_descent(std::vector<float> &w,
			float learning_rate,
			const std::vector<std::vector<float> > &feature,
			const std::vector<int32_t> &label) ;
	void gradient_descent(std::vector<float> &w,
			float learning_rate,
			const std::vector<std::vector<float> > &feature,
			const std::vector<int32_t> &label) ;

	void set_parameter(int32_t max_iter_cnt,
			float learning_rate,
			int _thread_cnt = 0);

	float sigmoid(float x);
	void train(int32_t opt_type);
	float predict(const std::vector<float> &feature);	
	void dump_model(const char* model_file_name);
	void load_model(const char* model_file_name);
};

}
}

#endif
