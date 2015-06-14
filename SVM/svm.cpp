// support machine learning source code written by aron@whu
// this source file is designed to implement the definiton in 
// svm.h 

// auth : aron
// date : 2014-11-28

#include "svm.h"

float SVM::predict(const vector<float> &feature) {
	float val = 0.0;
	for (vector<int>::iterator it = _support_vec_index.begin(); 
			it != _support_vec_index.end() ; it++) {
		val += _alpha[*it] * _train_y[*it] * kernel_cal(_train_x[*it], feature);
	}
	return val + _b;
}

float SVM::kernel_cal(const vector<float> &x,
		const vector<float> &feature) {
	float ret = 0.0;
	for (int i = 0; i < _p; i ++) {
		ret += x[i] * feature[i];
	}
	return ret;
}

void SVM::split(string &s, vector<string> &split_ret, string &tag) {
	if (s.find(tag) == string::npos) {
		return ;
	}
	int cur_pos = 0;
	int find_pos = 0;
	s += tag;
	while ((find_pos = s.find(tag, cur_pos)) != string::npos) {
		split_ret.push_back(s.substr(cur_pos, find_pos - cur_pos));
		cur_pos = find_pos + 1;
	}
}

bool SVM::load_feature(const char *file_name) {
	ifstream feature_file(file_name);
	string line;
	int sample_cnt = 0;
	string tag;
	tag = "\t";
	vector<vector<float> > input_x;
	vector<float> input_y;
	while (getline(feature_file, line)) {
		vector<string> split_ret;
		vector<float> x;
		split(line, split_ret, tag);
		if (_p == 0) {
			_p = split_ret.size() - 1;
		}
		if ((int) split_ret.size() != _p + 1) {
			cerr << "Error when load feature in " << sample_cnt << "line" << endl; 
			return false;
		}
		// fit output
		input_y.push_back(atoi(split_ret[_p].c_str()));
		// fit input
		for (int i = 0; i < _p; i ++) {
			x.push_back(atof(split_ret[i].c_str()));
		}
		for (int i = 0;  i < (int)x.size(); i ++) {
			cout << x[i] << " ";
		}
		cout << endl;
		input_x.push_back(x);
		sample_cnt += 1;
	}

	for (int i = 0; i < sample_cnt; i ++) {
		if (i < sample_cnt * 0.8) {
			_train_x.push_back(input_x[i]);
			_train_y.push_back(input_y[i]);
		} else {
			_test_x.push_back(input_x[i]);
			_test_y.push_back(input_y[i]);
		}
	}

	_n = sample_cnt * 0.8;
	for (int i = 0; i < _n; i ++) {
		_kernel.push_back(vector<float>(_n, 0.0));
	}

	for (int i = 0; i < _n; i ++) {
		for (int j = i; j < _n; j ++) {
			float kernel_value = kernel_cal(_train_x[i], _train_x[j]);
			_kernel[i][j] = kernel_value;
			_kernel[j][i] = kernel_value;
		}
	}
	return true;
}

void SVM::init_alpha() {
	_b = 0.0;
	for (int i = 0; i < _n; i ++) {
		_alpha.push_back(0);
		_non_bound_vec_index.push_back(i);
		_error.push_back(- _train_y[i]);
	}
}

bool SVM::fit_kkt(int i) {
	if (_alpha[i] > 0 && _alpha[i] < _c) {
		return (_train_y[i] * _error[i] - 0.0) < eps;
	} else if (fabs(_alpha[i] - 0) < eps) {
		return _train_y[i] * _error[i] > 0.0;
	} else if (fabs(_alpha[i] - _c) < eps) {
		return _train_y[i] * _error[i] < 0.0;
	}
	return true;
}

int SVM::select_j(int i) {
	int ret_index;
	float max_error_margin = 0.0;
	bool tag = false;
	
	for (int j = 0; j < _n; j ++) {
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
			ret_index = rand() % _n;
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

	if (fabs(eta - 0.0) < eps) {
		return false;
	}

	alpha_j_new_unc = _alpha[j] + (_train_y[j] * (_error[i] - _error[j]) / eta);
	// define the boundary
	// the alpha has the constraints 0 <= alpha <= c, and
	// alpha_old_i * _train_y[i] + alpha_old_j * _train_y[j] = 
	// alpha_new_i * _train_y[i] + alpha_new_j * _train_y[j]
	// following, the _alpha[i], _alpha[j] stand for alpha_old_i and alpha_old_j
	if (_train_y[i] * _train_y[j] != 1) {
		L = max((float)0, _alpha[j] - _alpha[i]);
		H = min(_c, _c + _alpha[j] - _alpha[i]);
	} else {
		L = max((float)0, _alpha[j] + _alpha[i] - _c);
		H = min(_c, _alpha[j] + _alpha[i]);
	}
	
	// cliped the alpha_j
	if (alpha_j_new_unc > H) alpha_j_new = H;
	else if (alpha_j_new_unc >= L && alpha_j_new_unc <= H) alpha_j_new = alpha_j_new_unc;
	else alpha_j_new = L;
	
	// update the alpah_i
	alpha_i_new = _alpha[i] + _train_y[i]*_train_y[j] * (_alpha[j] - alpha_j_new);
	
	//  update the bias
	b1 = _b - _error[i] - _train_y[i] * (alpha_i_new - _alpha[i]) * _kernel[i][i] - 
		_train_y[j] * _kernel[i][j] * (alpha_j_new - _alpha[j]);
	b2 = _b - _error[j] - _train_y[i] * _kernel[i][j] * (alpha_i_new - _alpha[i]) -
		_train_y[j] * _kernel[j][j] * (alpha_j_new - _alpha[j]);
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
	
	for (int i = 0; i < _n; i ++) {
		base_error_i += _train_y[i] * _alpha[i] * _kernel[i][i];
		base_error_j += _train_y[i] * _alpha[i] * _kernel[j][i];
	}
	_error[i] = base_error_i + _b - _train_y[i];
	_error[j] = base_error_j + _b - _train_y[j];
		
	return true;
}

void SVM::train() {
	int cur_iter = 0;
	
	while (cur_iter ++ < _max_iter_cnt) {	
		int tag = 0;
		// find alpha_i in support vector
		cout << "Iter " << cur_iter << endl;
		for (vector<int>::iterator it = _bound_vec_index.begin(); 
				it != _bound_vec_index.end(); it ++) {
			if (! fit_kkt(*it)) {
				if (inner_loop(*it)) tag += 1;
			}
		}
		if (0 == tag) {
			// find in non-support vector
			for (vector<int>::iterator it = _non_bound_vec_index.begin();
					it != _non_bound_vec_index.end(); it ++) {
				if (! fit_kkt(*it)) {
					if(inner_loop(*it)) tag += 1;
				}
			}
		}
		// update non-support vector & support vector
		update_bound_vector_index();
		update_support_vector_index();
		cerr << "support vector cnt:" <<  _support_vec_index.size() << endl;
		if (0 == tag) break;
	}
}

void SVM::test() {
	int right_cnt = 0;
	for (int i = 0; i < _test_x.size(); i ++) {
		cout << _test_y[i] << " " << predict(_test_x[i]) << endl;
		if (predict(_test_x[i]) <= -1 && _test_y[i] == -1 ||
				predict(_test_x[i]) >= 1 && _test_y[i] == 1) {
			right_cnt += 1;
		}
	}
	cerr << "Test precision: " << right_cnt * 1.0 * 100 / _test_x.size() << "%" << endl;
}

void SVM::update_support_vector_index() {
	_support_vec_index.clear();
	for (vector<int>::iterator it = _bound_vec_index.begin();
			it != _bound_vec_index.end(); it ++) {
		if (fit_kkt(*it))  {
			_support_vec_index.push_back(*it);
		}
	}
}

void SVM::update_bound_vector_index() {
	_bound_vec_index.clear();
	_non_bound_vec_index.clear();
	for (int i = 0; i < _n; i ++) {
		if (fabs(_alpha[i] - 0.0) <= eps || fabs(_alpha[i] - _c) <= eps) {
			_non_bound_vec_index.push_back(i);
		} else {
			_bound_vec_index.push_back(i);
		}
	}
}

