// implemetation of ANN.h

#include "ANN.h"

namespace suml {
namespace nn {



void ANN::set_max_iter_cnt(int32_t max_iter_cnt) {
    _max_iter_cnt = max_iter_cnt;
}

void ANN::set_learning_rate(float learning_rate) {
    _learning_rate = learning_rate;
}

void ANN::set_reg_type(int32_t reg_type) {
    _reg_type = reg_type;
}

void ANN::set_lambda(float lambda) {
    _lambda = lambda;
}

void ANN::set_hid_lev_cnt(int32_t hid_lev_cnt) {
    _hid_lev_cnt = hid_lev_cnt;
}

void ANN::set_out_lev_cnt(int32_t out_lev_cnt) {
    _out_lev_cnt = out_lev_cnt;
}

float ANN::calc_loss_val() const {
    float ret_val = 0.0;
    for (int32_t ins = 0; ins < _sample_size; ins ++) {
        for (int32_t k = 0; k < _out_lev_cnt; k ++) {
            ret_val += (_dis_label[ins][k] - _out_lev_output[k]) * \
                       (_dis_label[ins][k] - _out_lev_output[k]);
        }
    }

    return ret_val / (2 * _sample_size);
}

void ANN::shuffle() {    
    
    std::vector<int32_t> new_indexes(_sample_size, -1);
    
    srand( (unsigned)time(NULL));
    
    for (int32_t i = 0; i < _sample_size; ++i) {
        int32_t rand_idx = rand() % _sample_size;    
        std::swap(_feature[i], _feature[rand_idx]);
        std::swap(_label[i], _label[rand_idx]);
    }



}

void ANN::stochastic_gradient_descent() {
    
    for (int32_t ins = 0; ins < _sample_size; ++ins) {
        for (int32_t j = 0; j < _hid_lev_cnt; ++j) {
            _hid_lev_output[j] = 0.0;

            for (int32_t i = 0; i < _feature_dim; ++i) {
                _hid_lev_output[j] += _feature[ins][i] * _in_hid_w[i][j];
            }
            _hid_lev_output[j] = suml::util::sigmoid(_hid_lev_output[j]);    
        
        }

        for (int32_t k = 0; k < _out_lev_cnt; ++k) {
            _out_lev_output[k] = 0.0;
            for (int32_t j = 0; j < _hid_lev_cnt; ++j ) {
                _out_lev_output[k] += _hid_lev_output[j] * _hid_out_w[j][k];
            }
            _out_lev_output[k] = suml::util::sigmoid(_out_lev_output[k]);
        }

        

        for (int32_t k = 0; k < _out_lev_cnt; ++k) {

            _out_loss_val[k] = _out_lev_output[k] 
                                * (1.0 - _out_lev_output[k]) 
                                * (_dis_label[ins][k] - _out_lev_output[k]);
        }    
        

        for (int32_t j = 0; j < _hid_lev_cnt; ++j) {
            float val = 0.0;
            for (int32_t k = 0; k < _out_lev_cnt; ++k) {
                val += _hid_out_w[j][k] * _out_loss_val[k];    
            }
            _hid_loss_val[j] = _hid_lev_output[j] * (1 - _hid_lev_output[j]) * val;
        }

        for (int32_t j = 0; j < _hid_lev_cnt; ++j) {
            for (int32_t k = 0; k < _out_lev_cnt; ++k) {
                _hid_out_w[j][k] = _hid_out_w[j][k] 
                                + _learning_rate * _out_loss_val[k] * _hid_lev_output[j];
            }
        }

        for (int32_t i = 0; i < _feature_dim; ++i) {
            for (int32_t j = 0; j < _hid_lev_cnt; ++j) {
                _in_hid_w[i][j] = _in_hid_w[i][j] + _learning_rate * _hid_loss_val[j] * _feature[ins][i];
            }
        }
    }
}

void ANN::train(int32_t opt_type) {
    
    label_decode();
    srand( (unsigned)time(NULL));
    
    _in_hid_w.resize(_feature_dim, std::vector<float>(_hid_lev_cnt, 0.0));
    _hid_out_w.resize(_hid_lev_cnt, std::vector<float>(_out_lev_cnt, 0.0));
    
    _hid_loss_val.resize(_hid_lev_cnt, 0.0);
    _out_loss_val.resize(_out_lev_cnt, 0.0);
    
    for (int32_t j = 0; j < _hid_lev_cnt; ++j ){
        
        for (int32_t i = 0; i < _feature_dim; ++i) {
            _in_hid_w[i][j] = (rand() % 100) / 10000.0;    
        }

        for (int32_t k = 0; k < _out_lev_cnt; ++k) {
            _hid_out_w[j][k] = (rand() % 100) / 10000.0;
        }

    }
    
    _hid_lev_output.resize(_hid_lev_cnt, 0.0);
    _out_lev_output.resize(_out_lev_cnt, 0.0);

    float last_loss_val = 0.0;
    
    for (int32_t iter = 0; iter < _max_iter_cnt; ++iter) {
        // TODO shuffle the samples
        

        stochastic_gradient_descent();
        float loss = calc_loss_val();
        std::cout << "Iter " << iter + 1 << " : " << loss << std::endl;
        
        for (int32_t i = 0; i < _feature_dim; ++i) {
            for (int32_t j = 0; j < _hid_lev_cnt; ++j ){
                std::cout << _in_hid_w[i][j] << " ";        
            }
            std::cout << std::endl;
        }
        

        if (fabs(last_loss_val - loss) < EPS) {
            break;
        }
        last_loss_val = loss;
    }
}


float ANN::predict(const std::vector<float> &feature) {
}

void ANN::predict(const std::vector<float> &feature, std::vector<float> &ret) {
    for (int32_t j = 0; j < _hid_lev_cnt; ++j) {
        _hid_lev_output[j] = 0.0;
        for (int32_t i = 0; i < _feature_dim; ++i) {
            _hid_lev_output[j] += feature[i] * _in_hid_w[i][j];
        }
        _hid_lev_output[j] = suml::util::sigmoid(_hid_lev_output[j]);    
    }

    for (int32_t k = 0; k < _out_lev_cnt; ++k) {
        _out_lev_output[k] = 0.0;
        for (int32_t j = 0; j < _hid_lev_cnt; ++j ) {
            _out_lev_output[k] += _hid_lev_output[j] * _hid_out_w[j][k];
        }
        _out_lev_output[k] = suml::util::sigmoid(_out_lev_output[k]);    
    }
    ret.assign(_out_lev_output.begin(), _out_lev_output.end());
}

void ANN::label_decode() {
    _dis_label.resize(_sample_size, std::vector<float>(_out_lev_cnt, 0.0));
    
    for (int32_t ins = 0; ins < _sample_size; ++ins) {
        for (int32_t k = 0; k < _out_lev_cnt; ++k) {
            if (_label[ins] == k) {
                _dis_label[ins][k] = 1;
            } else {
                _dis_label[ins][k] = 0;
            }
        }
    }

}


void ANN::load_model(const char* model_file_name) {
}

void ANN::dump_model(const char* model_file_name) {
}

}

}
