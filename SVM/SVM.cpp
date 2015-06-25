

#include "SVM.h"


namespace suml {
namespace svm {

float SVM::predict(const std::vector<float> &feature) {
	float val = 0.0;
	for (std::vector<int>::iterator it = _support_vec_index.begin(); 
			it != _support_vec_index.end() ; it++) {
		val += _alpha[*it] * _label[*it] * kernel_cal(_feature[*it], feature);
	}
	return val + _b;
}

float SVM::kernel_cal(const std::vector<float> &x,
		const std::vector<float> &feature) {
	float ret = 0.0;
	for (int i = 0; i < _feature_dim; i ++) {
		ret += x[i] * feature[i];
	}
	return ret;
}

void SVM::init_alpha() {
	_b = 0.0;
	for (int i = 0; i < _sample_size; i ++) {
		_alpha.push_back(0);
		_non_bound_vec_index.push_back(i);
		_error.push_back(- _label[i]);
	}
	

	for (int i = 0; i < _sample_size; i ++) {
		_kernel.push_back(std::vector<float>(_sample_size, 0.0));
	}

	for (int i = 0; i < _sample_size; i ++) {
		for (int j = i; j < _sample_size; j ++) {
			float kernel_value = kernel_cal(_feature[i], _feature[j]);
			_kernel[i][j] = kernel_value;
			_kernel[j][i] = kernel_value;
		}
	}
}

bool SVM::fit_kkt(int i) {
	if (_alpha[i] > 0 && _alpha[i] < _c) {
		return (_label[i] * _error[i] - 0.0) < EPS;
	} else if (fabs(_alpha[i] - 0) < EPS) {
		return _label[i] * _error[i] > 0.0;
	} else if (fabs(_alpha[i] - _c) < EPS) {
		return _label[i] * _error[i] < 0.0;
	}
	return true;
}

int SVM::select_j(int i) {
	int ret_index;
	float max_error_margin = 0.0;
	bool tag = false;
	
	for (int j = 0; j < _sample_size; j ++) {
		if (i != j && fabs(_error[i] - _error[j]) >= max_error_margin) {
			max_error_margin = fabs(_error[i] - _error[j]);
			ret_index = j;
			tag = true;
			break;
		}
	}
	if (! tag) {
		ret_index = i;
		while (ret_index == i) {
			srand((unsigned)time(NULL));
			ret_index = rand() % _sample_size;
		}
	}

	return ret_index;
}


bool SVM::inner_loop(int i) {
	// choose the second parameter j
	int j;
	float L, H, b1, b2;
	float eta, alpha_j_new, alpha_j_new_unc, alpha_i_new;
	
	j = select_j(i);
	
	eta = _kernel[i][i] + _kernel[j][j] - 2*_kernel[i][j];

	if (fabs(eta - 0.0) < EPS) {
		return false;
	}

	alpha_j_new_unc = _alpha[j] + (_label[j] * (_error[i] - _error[j]) / eta);
	// define the boundary
	// the alpha has the constraints 0 <= alpha <= c, and
	// alpha_old_i * _label[i] + alpha_old_j * _label[j] = 
	// alpha_new_i * _label[i] + alpha_new_j * _label[j]
	// following, the _alpha[i], _alpha[j] stand for alpha_old_i and alpha_old_j
	if (_label[i] * _label[j] != 1) {
		L = std::max((float)0, _alpha[j] - _alpha[i]);
		H = std::min(_c, _c + _alpha[j] - _alpha[i]);
	} else {
		L = std::max((float)0, _alpha[j] + _alpha[i] - _c);
		H = std::min(_c, _alpha[j] + _alpha[i]);
	}
	
	// cliped the alpha_j
	if (alpha_j_new_unc > H) alpha_j_new = H;
	else if (alpha_j_new_unc >= L && alpha_j_new_unc <= H) alpha_j_new = alpha_j_new_unc;
	else alpha_j_new = L;
	
	// update the alpah_i
	alpha_i_new = _alpha[i] + _label[i]*_label[j] * (_alpha[j] - alpha_j_new);
	
	//  update the bias
	b1 = _b - _error[i] - _label[i] * (alpha_i_new - _alpha[i]) * _kernel[i][i] - 
		_label[j] * _kernel[i][j] * (alpha_j_new - _alpha[j]);
	b2 = _b - _error[j] - _label[i] * _kernel[i][j] * (alpha_i_new - _alpha[i]) -
		_label[j] * _kernel[j][j] * (alpha_j_new - _alpha[j]);
	if (alpha_i_new < _c && alpha_i_new > 0) {
		_b = b1;
	} else if (alpha_j_new < _c && alpha_j_new > 0) {
		_b = b2;
	} else {
		_b = (b1 + b2) / 2;
	}
	_alpha[i] = alpha_i_new;
	_alpha[j] = alpha_j_new;
	// update the error
	float base_error_i = 0.0, base_error_j = 0.0;
	
	for (int i = 0; i < _sample_size; i ++) {
		base_error_i += _label[i] * _alpha[i] * _kernel[i][i];
		base_error_j += _label[i] * _alpha[i] * _kernel[j][i];
	}
	_error[i] = base_error_i + _b - _label[i];
	_error[j] = base_error_j + _b - _label[j];
		
	return true;
}

void SVM::train(int32_t opt_type) {

	int cur_iter = 0;
	while (cur_iter ++ < _max_iter_cnt) {	
		int tag = 0;
		// find alpha_i in support std::vector
		std::cout << "Iter " << cur_iter << std::endl;
		for (std::vector<int>::iterator it = _bound_vec_index.begin(); 
				it != _bound_vec_index.end(); it ++) {
			if (! fit_kkt(*it)) {
				if (inner_loop(*it)) tag += 1;
			}
		}
		if (0 == tag) {
			// find in non-support std::vector
			for (std::vector<int>::iterator it = _non_bound_vec_index.begin();
					it != _non_bound_vec_index.end(); it ++) {
				if (! fit_kkt(*it)) {
					if(inner_loop(*it)) tag += 1;
				}
			}
		}
		// update non-support std::vector & support std::vector
		update_bound_vector_index();
		update_support_vector_index();
		std::cerr << "support std::vector cnt:" <<  _support_vec_index.size() << std::endl;
		if (0 == tag) break;
	}
}

void SVM::update_support_vector_index() {
	_support_vec_index.clear();
	for (std::vector<int>::iterator it = _bound_vec_index.begin();
			it != _bound_vec_index.end(); it ++) {
		if (fit_kkt(*it))  {
			_support_vec_index.push_back(*it);
		}
	}
}

void SVM::update_bound_vector_index() {
	_bound_vec_index.clear();
	_non_bound_vec_index.clear();
	for (int i = 0; i < _sample_size; i ++) {
		if (fabs(_alpha[i] - 0.0) <= EPS || fabs(_alpha[i] - _c) <= EPS) {
			_non_bound_vec_index.push_back(i);
		} else {
			_bound_vec_index.push_back(i);
		}
	}
}

void SVM::load_model(const char* file_name) {}

void SVM::dump_model(const char* file_name) {}

}
}
