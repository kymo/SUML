#include "LR.h"


namespace suml {
namespace lr {

void LR::set_parameter(int32_t max_iter_cnt,
		float learning_rate,
		int thread_cnt) {
	_max_iter_cnt = max_iter_cnt;
	_learning_rate = learning_rate;
	_thread_cnt = thread_cnt;
}

float LR::sigmoid(float x) {
	return 1.0 / (1 + exp(-x));
}

void LR::gradient_descent(std::vector<float> &w,
		float learning_rate,
		const std::vector<std::vector<float> > &feature,
		const std::vector<int32_t> &label) {
	
	std::vector<float>  temp_x(_sample_size, 0.0);
	
	for (int32_t i = 0; i < _sample_size; i ++) {
		for (int32_t j = 0; j < _feature_dim; j ++) {
			temp_x[i] += w[j] * feature[i][j];
		}
		temp_x[i] = sigmoid(temp_x[i]);
	}
	for (int32_t i = 0; i < _feature_dim; i ++) {
		float gradient = 0.0;
		for (int32_t j = 0; j < _sample_size; j ++) {
			gradient += (label[j] - temp_x[j]) * feature[j][i];
		}
		w[i] = w[i] + learning_rate * gradient;
	}
}

bool LR::stochatic_gradient_descent(std::vector<float> &w,
		float learning_rate,
		const std::vector<std::vector<float> > &feature,
		const std::vector<int32_t> &label) {
	std::vector<float>  temp_x(_sample_size, 0.0);
	
	for (int32_t i = 0; i < _sample_size; i ++) {
		for (int32_t j = 0; j < _feature_dim; j ++) {
			temp_x[i] += w[j] * feature[i][j];
		}
		temp_x[i] = sigmoid(temp_x[i]);
	}

	for (int32_t i = 0; i < _sample_size; i ++) {
		float stochastic_error = 0.0;
		for (int32_t j = 0; j < _feature_dim; j ++) {
			float gradient = (label[i] - temp_x[i]) * feature[i][j];
			w[j] = w[j] + learning_rate * gradient;
			stochastic_error += gradient * gradient;
		}
		if (stochastic_error < SGD_EPS) {
			return true;
		}
	}
	return false;
}

float LR::calc_likely_hood() {
	float ret_val = 0.0;
	for (int32_t i = 0; i < _sample_size; ++i) {
		float temp_val = 0.0;
		for (int32_t j = 0; j < _feature_dim; ++j) {
			temp_val += _w[j] * _feature[i][j];
		}
		ret_val += _label[i] * log(sigmoid(temp_val)) + (1 - _label[i]) * log(1 - sigmoid(temp_val));
	}	
	return ret_val;
}

void LR::train(int32_t opt_type) {
	float last_value = 0.0;

	srand( (unsigned)time(NULL));
	for (int32_t i = 0; i < _feature_dim; i ++) {
		_w.push_back((rand() % 100) / 10000.0);
	}
	for (int32_t iter = 0; iter < _max_iter_cnt; iter ++) {
		float temp_value = calc_likely_hood();
		std::cout << "iter " << iter << " : " <<  temp_value << std::endl;
		if (opt_type == GD) {
			gradient_descent(_w, _learning_rate, _feature, _label);
			if (fabs(temp_value - last_value) < LR_EPS) {
				break;
			}
			last_value = temp_value;
		} else if (opt_type == SGD) {
			if(stochatic_gradient_descent(_w, _learning_rate, _feature, _label)) {
				break;
			}
		}
	}
}

float LR::predict(const std::vector<float> &feature) {
	float value = 0.0;
	for (int i = 0; i < _feature_dim; i ++) {
		value += _w[i] * feature[i];
	}
	value = sigmoid(value);
	return value;
}

void LR::dump_model(const char* model_file_name) {
	std::ofstream model_file(model_file_name);
	model_file << _feature_dim << std::endl;
	model_file << _sample_size << std::endl;
	for (int32_t i = 0; i < _feature_dim; i ++) {
		model_file << _w[i] << std::endl;
	}
}

void LR::load_model(const char* model_file_name) {

}

}
}
