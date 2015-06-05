#ifndef __MODEL_H_
#define __MODEL_H_


#include "Util.h"

namespace suml {
namespace model {
	
class Model {	
public:
	Model() {}
	virtual ~Model() {}
	
	std::string _name;
	int32_t _sample_size;
	int32_t _feature_dim;

	std::vector<std::vector<float> > _feature;
	std::vector<int32_t> _label;	

	void set_data(std::vector<std::vector<float> > &feature,
			std::vector<int32_t> &label);
	
	void feature_normalize(int32_t normalize_type,
			std::vector<std::vector<float> > &feature);
	
	void feature_select();
	void k_fold_validation(int k);

	virtual void train(int32_t opt_type){}
	virtual float predict(const std::vector<float> &feautre){}
	// load model from file model_file_name
	virtual void load_model(const char* model_file_name) {}
	// dump model into file model_file_name
	virtual void dump_model(const char* model_file_name){}
};


}
}


#endif
