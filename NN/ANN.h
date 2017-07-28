#ifndef __ANN_H_
#define __ANN_H_

#include <iostream>
#include <vector>
#include <map>
#include <time.h>

#include "Model.h"
#include "Util.h"

namespace suml {
namespace nn {

class ANN : public suml::model::Model<int32_t> {
    
private:
    std::vector<std::vector<float> > _dis_label;
    
    std::vector<float> _hid_lev_output;
    std::vector<float> _out_lev_output;
    
    std::vector<std::vector<float> > _in_hid_w;
    std::vector<std::vector<float> > _hid_out_w;
    
    std::vector<float> _hid_loss_val;
    std::vector<float> _out_loss_val;
    
    int32_t     _max_iter_cnt;
    float       _learning_rate;
    float         _reg_type;
    float        _lambda;
    int32_t     _hid_lev_cnt;
    int32_t     _out_lev_cnt;

public:
    ANN () : _reg_type(-1), _lambda(0.0) {}
    ~ANN() {}    

    void set_max_iter_cnt(int32_t max_iter_cnt);
    void set_learning_rate(float learning_rate);
    void set_reg_type(int32_t reg_type);
    void set_lambda(float lambda);
    void set_hid_lev_cnt(int32_t hid_lev_cnt);
    void set_out_lev_cnt(int32_t out_lev_cnt);
    
    float calc_loss_val() const;

    void shuffle();

    void label_decode();

    void stochastic_gradient_descent();
    
    void predict(const std::vector<float> &feature,
            std::vector<float> &ret);

    void train(int32_t opt_type) ;
    
    float predict(const std::vector<float> &feature);
    void load_model(const char* model_file_name);
    void dump_model(const char* model_file_name);
};

}
}


#endif
