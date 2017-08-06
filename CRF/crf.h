#ifndef CRF_H_
#define CRF_H_

#include <iostream>
#include <string.h>
#include <vector>
#include <map>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <algorithm>

template <typename T>
T max(T a, T b) {
    return a > b ? a : b;
}

class CRF {

public:
    CRF(int hidden_state_cnt, int ob_state_cnt) {
        _hidden_state_cnt = hidden_state_cnt;
        _ob_state_cnt = ob_state_cnt;
        _f = new int**[ob_state_cnt + 1];
        _lambda = new double**[ob_state_cnt + 1];
        _delta_lambda = new double**[ob_state_cnt + 1];
        for (int i = 0; i < ob_state_cnt + 1; i++) {
            _f[i] = new int*[_hidden_state_cnt + 1];
            _lambda[i] = new double*[_hidden_state_cnt + 1];
            _delta_lambda[i] = new double*[_hidden_state_cnt + 1];
            for (int j = 0; j < _hidden_state_cnt + 1; j++) {
                _f[i][j] = new int[_hidden_state_cnt + 1];
                _lambda[i][j] = new double[_hidden_state_cnt + 1];    
                _delta_lambda[i][j] = new double[_hidden_state_cnt + 1];    
                for (int k = 0; k < _hidden_state_cnt + 1;k ++) {
                    _f[i][j][k] = 0.0;
                    _lambda[i][j][k] = 0.1;
                    _delta_lambda[i][j][k] = 0.1;
                }
            }
        }
        _f_cnt = (ob_state_cnt + 1) * (_hidden_state_cnt + 1) * (_hidden_state_cnt + 1);
    }
    
    int _hidden_state_cnt;        // 状态数量
    int _ob_state_cnt;

    int*** _f;    // feature function

    double*** _lambda; // lambda
    
    double*** _delta_lambda;     // lambda
    
    int _f_cnt;    // feature function cnt

    std::vector<std::vector<int> > features;
    std::vector<std::vector<int> > labels;

    void read_data(const char*file_name) {

        std::ifstream fis(file_name);
        std::string ob_state_str;
        int tag = 0;
        int last_tag = 0;
        std::vector<int> feature;
        std::vector<int> label;
        feature.push_back(0);
        label.push_back(0);
        while (fis >> ob_state_str) {
            if (ob_state_str == ";") {
                last_tag = 0;

                features.push_back(feature);
                labels.push_back(label);
                std::vector<int>().swap(feature);
                std::vector<int>().swap(label);
                feature.push_back(0);
                label.push_back(0);
                continue;
            }
            int ob_state = atoi(ob_state_str.c_str());
            fis >> tag;
            if (tag >= _ob_state_cnt) {
                std::cerr << "Trainning data Error!" << std::endl;
                exit(1);
            }
            feature.push_back(ob_state);
            label.push_back(tag);
            std::cout << ob_state << " " << last_tag << " " << tag << std::endl;
            _f[ob_state][last_tag][tag] = 1; // feature function
            std::cout << "end" << std::endl;
            last_tag = tag;
        }

        if (ob_state_str != ";") {
            features.push_back(feature);
            labels.push_back(label);
        }
    }

    void crf_train(const std::vector<std::vector<int> >& train_x_features,
        const std::vector<std::vector<int> >& train_y_labels) {
        
        // memset(_delta_lambda, 0, sizeof(_delta_lambda));
        
        double cost = 0.0;
        for (int f_idx = 0; f_idx <= _ob_state_cnt; f_idx ++) {
            for (int s_idx = 0; s_idx <= _hidden_state_cnt; s_idx++) {
                for (int t_idx = 0; t_idx <= _hidden_state_cnt; t_idx ++) {
                    _delta_lambda[f_idx][s_idx][t_idx] = 0.0;
                }
            }
        }
        for (int i = 0; i < train_x_features.size(); i++) {
            const std::vector<int>& feature = train_x_features[i];
            const std::vector<int>& label = train_y_labels[i];

            std::cout << std::endl;
            std::cout << "feature end" << std::endl;
            if (feature.size() != label.size()) {
                std::cerr << "Error training data: feature size and label size not match!" << std::endl;
                exit(1);
            }
            
            int T = label.size() - 1;
            std::vector<std::vector<double> > alpha(T + 1, std::vector<double>(_hidden_state_cnt + 1, 0));
            std::vector<std::vector<double> > beta(T + 1, std::vector<double>(_hidden_state_cnt + 1, 0));
            // calculate alpha
            double alpha_tot_val = 0.0;
            std::cout << "--alpha value--" << std::endl;
            for (int t = 1; t <= T; t++) {
                for (int si = 1; si <= _hidden_state_cnt; si++) {
                    if (t == 1) {
                        // int lambda_idx = (feature[t]) * (_hidden_state_cnt + 1) * (_hidden_state_cnt + 1) + si;
                        // alpha[t][si] = _lambda[lambda_idx] * _f[feature[t]][0][si];
                        alpha[t][si] = _lambda[feature[t]][0][si] * _f[feature[t]][0][si];
                    } else {
                        alpha[t][si] = -1.0;
                        for (int sj = 1; sj <= _hidden_state_cnt; sj++) {
                            int lambda_idx = (feature[t]) * _hidden_state_cnt * _hidden_state_cnt +
                                (sj) * _hidden_state_cnt + si;

                            std::cout << lambda_idx << std::endl;    
                            alpha[t][si] = max<double>(alpha[t][si], 
                                alpha[t - 1][sj] * _lambda[feature[t]][sj][si] * _f[feature[t]][sj][si]); 
                        
                        }
                    }
                    if (t == T) {
                        alpha_tot_val += alpha[t][si];
                    }
                    std::cout << alpha[t][si] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "alpha end!" << std::endl;
            std::cout << "tot " << alpha_tot_val << std::endl;
            // calculate beta
            for (int t = T; t >= 1; t--) {
                for (int si = 1; si <= _hidden_state_cnt; si++) {
                    if (t == T) {
                        beta[t][si] = 1;
                    } else {
                        beta[t][si] = -1.0;
                        for (int sj = 1; sj <= _hidden_state_cnt; sj++) {
                            int lambda_idx = (feature[t + 1]) * _hidden_state_cnt * _hidden_state_cnt +
                                (sj) * _hidden_state_cnt + si;
                            beta[t][si] = max<double>(beta[t][si], beta[t + 1][sj] * _lambda[feature[t + 1]][sj][si] * 
                                _f[feature[t + 1]][sj][si]);
                        }
                    }
                    std::cout << beta[t][si] << " ";
                }
                std::cout << std::endl;
            }

            // calculate gradient
            
            for (int f_idx = 0; f_idx <= _ob_state_cnt; f_idx ++) {
                for (int s_idx = 0; s_idx <= _hidden_state_cnt; s_idx++) {
                    for (int t_idx = 0; t_idx <= _hidden_state_cnt; t_idx ++) {
                        double v = 0.0;
                            
                        for (int t = 1; t <= T; t++) {
                            if (feature[t] >= _ob_state_cnt) {
                                std::cout << "Error training data: feature data is not legal!" << std::endl;
                                exit(1);
                            }
                            // int k_idx = (feature[t]) * (_hidden_state_cnt + 1) * (_hidden_state_cnt + 1) + 
                            //    (label[t - 1]) * (_hidden_state_cnt + 1) + label[t] ;
                            std::cout << "he1" << std::endl;
                            if (f_idx == feature[t] && s_idx == label[t - 1] && t_idx == label[t]) {
                                v += _f[f_idx][s_idx][t_idx];
                            }
                            std::cout << "he" << std::endl;
                            
                            for (int si = 1; si <= _hidden_state_cnt; si++) {
                                for (int sj = 1; sj <= _hidden_state_cnt; sj++) {
                                    std::cout << "0" << " " << t << " " << feature[t] << std::endl;
                                    if (f_idx != feature[t] || s_idx != sj || t_idx != si) {
                                        continue;
                                    }
                                    std::cout << "1" << std::endl;
                                    double p = alpha[t - 1][sj] * _lambda[feature[t]][sj][si] * 
                                        _f[feature[t]][sj][si] * beta[t][si];
                                    v += p * _f[feature[t]][sj][si] / alpha_tot_val;
                                    std::cout << "2" << std::endl;
                                }
                            }
                        }
                        std::cout << "3" << std::endl;
                        std::cout << f_idx << " " << s_idx << " " << t_idx << " " << v << std::endl;
                        std::cout << _delta_lambda[0][0][0] << std::endl;
                        _delta_lambda[f_idx][s_idx][t_idx] += v;
                        std::cout << "4t" << std::endl;
                    }
                }
            }
            // calculate cost
            for (int t = 1; t <= T; t++) {
                cost += _lambda[feature[t]][label[t - 1]][label[t]] * 
                    _f[feature[t]][label[t - 1]][label[t]] - log(alpha_tot_val);
            }
        }
        std::cout << "cost " << cost << std::endl;
        
        // update gradient
        double eta = -0.005;
        for (int i = 0; i <= _ob_state_cnt; i++) {
            for (int j = 0; j <= _hidden_state_cnt; j++) {
                for (int k = 0; k <= _hidden_state_cnt; k++) {
                    _lambda[i][j][k] = _lambda[i][j][k] + _delta_lambda[i][j][k] * eta;
                }
            }
        }
    }

    void train() {

        int epoch_cnt = 100;
        int tot = features.size();
        std::cout <<"tot" <<  tot << std::endl;
        int batch_size = 1;
        for (int epoch = 0; epoch < epoch_cnt; epoch++) {
            
            for (int i = 0; i < tot / batch_size; i++) {
                std::vector<std::vector<int> > train_x_features;
                std::vector<std::vector<int> > train_y_labels;
                for (int j = i * batch_size; j < i * batch_size + batch_size; j++) {
                    train_x_features.push_back(features[j]);
                    train_y_labels.push_back(labels[j]);
                }
                crf_train(train_x_features, train_y_labels);

            }

        }
    }
};

#endif
