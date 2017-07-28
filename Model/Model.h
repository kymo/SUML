#ifndef __MODEL_H_
#define __MODEL_H_


#include "Util.h"

namespace suml {
namespace model {

template <class T>    

class Model {    

public:

    Model() {}
    virtual ~Model() {}

    std::string _name;

    int32_t _sample_size;
    int32_t _feature_dim;

    std::vector<std::vector<float> > _feature;
    std::vector<T> _label;    

    void set_data(std::vector<std::vector<float> > &feature,
            std::vector<T> &label);
    
    virtual void train(int32_t opt_type){}
    
    virtual float predict(const std::vector<float> &feautre){}
    // load model from file model_file_name
    virtual void load_model(const char* model_file_name) {}
    // dump model into file model_file_name
    virtual void dump_model(const char* model_file_name){}

};


template <class T>
void Model<T>::set_data(std::vector<std::vector<float> > &feature,
        std::vector<T> &label) {
    _feature = feature;
    _label = label;
    _sample_size = (int32_t)feature.size();
    if (0 == _sample_size) {
        std::cerr << "Error When Setting the Training Sample!" << std::endl;
        exit(0);
    }
    _feature_dim = (int32_t)feature[0].size();

}

}
}


#endif
