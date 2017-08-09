// Copyright (c) 2017 kymowind@gmail.com. All Rights Reserve.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//    http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. 
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
/**
* crf class
*/

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
                    _delta_lambda[i][j][k] = 0.0;
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
            _f[ob_state][last_tag][tag] = 1; // feature function
            last_tag = tag;
        }

        if (ob_state_str != ";") {
            features.push_back(feature);
            labels.push_back(label);
        }
    }

    void gradient_check(const std::vector<std::vector<int> >& train_x_features,
        const std::vector<std::vector<int> >& train_y_labels) {
        for (int i = 0; i <= _ob_state_cnt; i++) {
            for (int j = 0; j <= _hidden_state_cnt; j++) {
                for (int k = 0; k <= _hidden_state_cnt; k++) {
                    /// _lambda[i][j][k] = _lambda[i][j][k] + _delta_lambda[i][j][k] * eta;
                    double tv = _lambda[i][j][k];
                    
                    _lambda[i][j][k] = tv + 1.0e-4;
                    // calculate alpha
                    double f1 = 0.0;
                    for (int n = 0; n < train_x_features.size(); n++) {
                        const std::vector<int>& feature1 = train_x_features[n];
                        const std::vector<int>& label1 = train_y_labels[n];
                        int T = label1.size() - 1;
                        double alpha_tot_val1 = 0.0;
                        std::vector<std::vector<double> > alpha1(T + 1, std::vector<double>(_hidden_state_cnt + 1, 0));
                        for (int t = 1; t <= T; t++) {
                            for (int si = 1; si <= _hidden_state_cnt; si++) {
                                if (t == 1) {
                                    alpha1[t][si] = exp(_lambda[feature1[t]][0][si] * _f[feature1[t]][0][si]);
                                } else {
                                    alpha1[t][si] = 0.0;
                                    for (int sj = 1; sj <= _hidden_state_cnt; sj++) {
                                        alpha1[t][si] +=
                                            alpha1[t - 1][sj] * exp(_lambda[feature1[t]][sj][si] * _f[feature1[t]][sj][si]); 
                                    }
                                }
                                if (t == T) {
                                    alpha_tot_val1 += alpha1[t][si];
                                }
                            }
                        }

                        for (int t = 1; t <= T; t++) {
                            f1 += _lambda[feature1[t]][label1[t - 1]][label1[t]] * 
                                _f[feature1[t]][label1[t - 1]][label1[t]];
                        }
                        f1 -= log(alpha_tot_val1);
                    }

                    double f2 = 0.0;
                    _lambda[i][j][k] = tv - 1.0e-4;
                    // calculate alpha

                    for (int n = 0; n < train_x_features.size(); n++) {
                        const std::vector<int>& feature2 = train_x_features[n];
                        const std::vector<int>& label2 = train_y_labels[n];
                        int T = label2.size() - 1;
                        
                        double alpha_tot_val2 = 0.0;
                        std::vector<std::vector<double> > alpha2(T + 1, std::vector<double>(_hidden_state_cnt + 1, 0));
                        for (int t = 1; t <= T; t++) {
                            for (int si = 1; si <= _hidden_state_cnt; si++) {
                                if (t == 1) {
                                    alpha2[t][si] = exp(_lambda[feature2[t]][0][si] * _f[feature2[t]][0][si]);
                                } else {
                                    alpha2[t][si] = 0.0;
                                    for (int sj = 1; sj <= _hidden_state_cnt; sj++) {
                                        alpha2[t][si] +=
                                            alpha2[t - 1][sj] * exp(_lambda[feature2[t]][sj][si] * _f[feature2[t]][sj][si]); 
                                    }
                                }
                                if (t == T) {
                                    alpha_tot_val2 += alpha2[t][si];
                                }
                            }
                        }

                        for (int t = 1; t <= T; t++) {
                            f2 += _lambda[feature2[t]][label2[t - 1]][label2[t]] * 
                                _f[feature2[t]][label2[t - 1]][label2[t]];
                        }
                        f2 -= log(alpha_tot_val2);
                    }
                    // std::cout << f1 << " " << f2 << " "
                    // std::cout << "[" << _delta_lambda[i][j][k] << "," <<  (f1 - f2) / (2.0e-4) << "] ";

                    if (fabs(_delta_lambda[i][j][k] -  (f1 - f2) / (2.0e-4) ) > 1.0e-6) {
                        std::cout << "error when check gradient " << _delta_lambda[i][j][k] << "-" << (f1 - f2) / (2.0e-4) << std::endl;
                    }
                    _lambda[i][j][k] = tv;
                }
            }
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

            if (feature.size() != label.size()) {
                std::cerr << "Error training data: feature size and label size not match!" << std::endl;
                exit(1);
            }
            
            int T = label.size() - 1;
            std::vector<std::vector<double> > alpha(T + 1, std::vector<double>(_hidden_state_cnt + 1, 0));
            std::vector<std::vector<double> > beta(T + 1, std::vector<double>(_hidden_state_cnt + 1, 0));
            // forward: calculate alpha 
            double alpha_tot_val = 0.0;
            for (int t = 1; t <= T; t++) {
                for (int si = 1; si <= _hidden_state_cnt; si++) {
                    if (t == 1) {
                        alpha[t][si] = exp(_lambda[feature[t]][0][si] * _f[feature[t]][0][si]);
                    } else {
                        alpha[t][si] = 0.0;
                        for (int sj = 1; sj <= _hidden_state_cnt; sj++) {
                            alpha[t][si] += alpha[t - 1][sj] * exp(_lambda[feature[t]][sj][si] * _f[feature[t]][sj][si]);
                        }
                    }
                    if (t == T) {
                        alpha_tot_val += alpha[t][si];
                    }
                }
            }
            // backward: calculate beta
            for (int t = T; t >= 1; t--) {
                for (int si = 1; si <= _hidden_state_cnt; si++) {
                    if (t == T) {
                        beta[t][si] = 1;
                    } else {
                        beta[t][si] = 0.0;
                        for (int sj = 1; sj <= _hidden_state_cnt; sj++) {
                            beta[t][si] += beta[t + 1][sj] * exp(_lambda[feature[t + 1]][si][sj] * 
                                _f[feature[t + 1]][si][sj]);
                        }
                    }
                }
            }

            // calculate gradient
            for (int f_idx = 1; f_idx <= _ob_state_cnt; f_idx ++) {
                for (int s_idx = 0; s_idx <= _hidden_state_cnt; s_idx++) {
                    for (int t_idx = 1; t_idx <= _hidden_state_cnt; t_idx ++) {
                        double v = 0.0;
                            
                        for (int t = 1; t <= T; t++) {
                            if (feature[t] >= _ob_state_cnt) {
                                std::cout << "Error training data: feature data is not legal!" << std::endl;
                                exit(1);
                            }
                            if (f_idx == feature[t] && s_idx == label[t - 1] && t_idx == label[t]) {
                                v += _f[f_idx][s_idx][t_idx];
                            }
                       }

                       for (int t = 1; t <= T; t++) {
                           if (t == 1) {
                               for (int si = 1; si <= _hidden_state_cnt; si++) {
                                    if (f_idx != feature[t] || s_idx != 0 || t_idx != si) {
                                        continue;
                                    }
                                    double p = exp(_lambda[feature[t]][0][si] * _f[feature[t]][0][si]) * 
                                        beta[t][si];
                                    v -= p * _f[feature[t]][0][si] / alpha_tot_val;
                               }
                               continue;
                           } 
                               
                           for (int sj = 1; sj <= _hidden_state_cnt; sj++) {
                                for (int si = 1; si <= _hidden_state_cnt; si++) {
                                    if (f_idx != feature[t] || s_idx != sj || t_idx != si) {
                                        continue;
                                    }
                                    double p = alpha[t - 1][sj] * exp(_lambda[feature[t]][sj][si] * 
                                        _f[feature[t]][sj][si]) * beta[t][si];
                                    v -= p * _f[feature[t]][sj][si] / alpha_tot_val;
                                }
                            }
                        }
                        _delta_lambda[f_idx][s_idx][t_idx] += v;
                    }
                }
            }
            // calculate cost
            for (int t = 1; t <= T; t++) {
                cost += _lambda[feature[t]][label[t - 1]][label[t]] * 
                    _f[feature[t]][label[t - 1]][label[t]];
            }
            cost -= log(alpha_tot_val);
        }

        std::cout << "cost " << cost << std::endl;
        std::vector<int> res;
        vertebi(train_x_features[0], res);
        for (int i = 1; i <= train_y_labels[0].size() - 1; i++) {
            std::cout << train_y_labels[0][i] << " ";
        }
        std::cout << std::endl;
        for (int i = 1; i <= res.size() - 1; i++) {
            std::cout << res[i] << " ";
        }
        std::cout << std::endl;
        // gradient check
        gradient_check(train_x_features, train_y_labels);
        // update gradient
        double eta = 0.005;
        for (int i = 0; i <= _ob_state_cnt; i++) {
            for (int j = 0; j <= _hidden_state_cnt; j++) {
                for (int k = 0; k <= _hidden_state_cnt; k++) {
                    _lambda[i][j][k] = _lambda[i][j][k] + _delta_lambda[i][j][k] * eta;
                }
            }
        }
    }

    void vertebi(const std::vector<int>& feature, std::vector<int>& label) {
        
        int T = feature.size() - 1;
        std::vector<std::vector<double> > sigma(T + 1, std::vector<double>(_hidden_state_cnt + 1, 0.0));
        std::vector<std::vector<int> > path(T + 1, std::vector<int>(_hidden_state_cnt + 1, 0));
        int last_max_pos = 0;
        int last_max_val = -1.0;
        for (int t = 1; t <= T; t++) {
            for (int si = 1; si <= _hidden_state_cnt; si++) {
                int last_pos = 0;
                if (t == 1) {
                    sigma[t][si] = exp(_lambda[feature[t]][0][si] * _f[feature[t]][0][si]);
                } else {
                    sigma[t][si] = -1.0;
                    for (int sj = 1; sj <= _hidden_state_cnt; sj++) {
                        double val = sigma[t - 1][sj] * exp(_lambda[feature[t]][sj][si] * _f[feature[t]][sj][si]);
                        if (val > sigma[t][si]) {
                            sigma[t][si] = val;
                            last_pos = sj;
                        }
                    }
                }
                path[t][si] = last_pos;
                if (t == T) {
                    if (sigma[t][si] > last_max_pos) {
                        last_max_pos = sigma[t][si];
                        last_max_pos = si;
                    }
                }
            }
        }
        label.push_back(last_max_pos);
        for (int t = T; t >= 2; t--) {
            last_max_pos = path[t][last_max_pos];
            label.push_back(last_max_pos);
        }
        label.push_back(0);
        std::reverse(label.begin(), label.end());
    }

    void train() {

        int epoch_cnt = 100;
        int tot = features.size();
        std::cout <<"tot" <<  tot << std::endl;
        int batch_size = 10;
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
