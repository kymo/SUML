#include "LR.h"


namespace suml {
namespace lr {

void LR::set_parameter(int32_t max_iter_cnt,
		float learning_rate,
		int32_t reg_type,
		float lambda,
		int32_t thread_cnt) {
	_max_iter_cnt = max_iter_cnt;
	_learning_rate = learning_rate;
	_reg_type = reg_type;
	_lambda = lambda;
	_thread_cnt = thread_cnt;

}

float LR::sigmoid(float x) {
	return 1.0 / (1 + exp(-x));
}

void LR::bfgs(std::vector<float> &w,
		float learning_rate,
		const std::vector<std::vector<float> > &feature,
		const std::vector<int32_t> &label,
		float reg_type = -1,
		float lambda = 0.0) {


}

void LR::gradient_descent(std::vector<float> &w,
		float learning_rate,
		const std::vector<std::vector<float> > &feature,
		const std::vector<int32_t> &label,
		float reg_type = -1,
		float lambda = 0.0) {
	
	std::vector<float>  output(_sample_size, 0.0);
	float w_sum = 0.0;
	
	if (reg_type != -1) {
		if (lambda == 0.0) {
			lambda = learning_rate;
		}
	}

	if (reg_type == REG_L1) {
		for (int32_t i = 0; i < _feature_dim; i ++) {
			w_sum += w[i] * w[i];
		}
	} 

	for (int32_t i = 0; i < _sample_size; i ++) {
		for (int32_t j = 0; j < _feature_dim; j ++) {
			output[i] += w[j] * feature[i][j];
		}
		output[i] = sigmoid(output[i]);
	}

	for (int32_t i = 0; i < _feature_dim; i ++) {
		float gradient = 0.0;
		
		for (int32_t j = 0; j < _sample_size; j ++) {
			gradient += (label[j] - output[j]) * feature[j][i];
		}
		// regularization
		if (reg_type == REG_L1) {
	
			gradient -= (w[i] > 0 ? lambda  : - lambda) ;
		
		} else if (reg_type == REG_L2) {
			
			gradient -= 2 * lambda * w[i];
		
		}
		gradient /= 2 * _sample_size;
		
		w[i] = w[i] + learning_rate * gradient;
	}
}




bool LR::stochastic_gradient_descent(std::vector<float> &w,
		float learning_rate,
		const std::vector<std::vector<float> > &feature,
		const std::vector<int32_t> &label,
		float reg_type = -1,
		float lambda = 0.0) {
	std::vector<float>  output(_sample_size, 0.0);
	
	for (int32_t i = 0; i < _sample_size; i ++) {
		for (int32_t j = 0; j < _feature_dim; j ++) {
			output[i] += w[j] * feature[i][j];
		}
		output[i] = sigmoid(output[i]);
	}

	for (int32_t i = 0; i < _sample_size; i ++) {
		
		float stochastic_error = 0.0;
		
		for (int32_t j = 0; j < _feature_dim; j ++) {
			float gradient = (label[i] - output[i]) * feature[i][j];
			w[j] = w[j] + learning_rate * gradient;
			stochastic_error += gradient * gradient;
		}
		
		if (learning_rate * stochastic_error < SGD_EPS) {
			return true;
		}

	}
	return false;
}

float LR::calc_loss_value() {
	
	float ret_val = 0.0;
	
	for (int32_t i = 0; i < _sample_size; ++i) {
		float temp_val = 0.0;
		for (int32_t j = 0; j < _feature_dim; ++j) {
			temp_val += _w[j] * _feature[i][j];
		}
		
		ret_val += _label[i] * log(sigmoid(temp_val)) + (1 - _label[i]) * log(1 - sigmoid(temp_val));
		
		//ret_val += (_label[i] == 1) ? log(sigmoid(temp_val)) : log(1 - sigmoid(temp_val));
	}	
	
	return ret_val;
}

void LR::train(int32_t opt_type) {
	float last_value = 0.0;

	srand( (unsigned)time(NULL));
	
	for (int32_t i = 0; i < _feature_dim; i ++) {
		
		_w.push_back((rand() % 100) / 1000.0);
	
	}
	
	for (int32_t iter = 0; iter < _max_iter_cnt; iter ++) {
	
		float temp_value = calc_loss_value();
#ifdef DEBUG
		std::cout << "iter " << iter << " : " <<  temp_value << std::endl;
#endif	
		if (opt_type == GD) {
			gradient_descent(_w, _learning_rate, _feature, _label, _reg_type, _lambda);
			if ( fabs(temp_value - last_value) < LR_EPS ) {
				break;
			}
			last_value = temp_value;
		
		} else if (opt_type == SGD) {
			
			if ( stochastic_gradient_descent(_w, _learning_rate, _feature, _label, _reg_type, _lambda) ) {
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
