
// support machine learning source code written by aron @ whu
// This header file is designed to define the main class and some 
// attributes of the support machine learning

// auth : aron @ whu
// date : 2014-11-28

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
using namespace std;
#define eps 1.0e-8
// support std::vector machine main class definition
// 
class SVM {
private:
	
	std::vector<int> _support_vec_index;			// support vector index
	std::vector<int> _bound_vec_index;				// samples' indexes on bound
	std::vector<int> _non_bound_vec_index;			// samples' indexes not on bound
	std::vector<float> _alpha;						// langange multiplier
	std::vector<std::vector<float> > _kernel;				// save the kernel output
	std::vector<float> _error;							// error for ith sample
	float _b;									// bias 
	float _c;									// meta-parameter for C
	std::vector<std::vector<float> > _train_x;					// train sample feature
	std::vector<std::vector<float> > _test_x;					//test sample feature
	
	std::vector<int> _train_y;							// output
	std::vector<int> _test_y;							// output
	
	int _max_iter_cnt;							// max iter times
	int _n;										// sample size
	int _p;										// feature dimension

public:
	SVM(float C, int max_iter_cnt):_c(C), _max_iter_cnt(max_iter_cnt),_p(0) {};
	~SVM();

	// given the vector values of feature in test sample, output the outcome
	// >= 1: positive <= -1:negative otherwise: unknown
	float predict(const std::vector<float> &feature);	
	
	// train the model
	void train();	

	// test
	void test();
	
	// load feature from feature file named file_name
	bool load_feature(const char *file_name);		
	
	// split the string s by tag, and store the result in split_ret
	void split(std::string &s, std::vector<std::string> &split_ret,
			std::string &tag);								
	
	// init parameters: alpha->0.0 error->0.0 bias->0.0
	void init_alpha();
	
	// whether the ith sample fits the kkt condition:	
	// 0 < alpha_i < c <=> _y[i] * _error[i] == 0
	// 0 == alpha_i    <=> _y[i] * _error[i] > 0
	// c == alpha_i    <=> _y[i] * _error[i] < 0
	bool fit_kkt(int i);						

	// inner loop in standard svm source code
	// so as to find the alpah_j, and update alpha_i, alpha_j, bias, error
	// firstly ,we search the support vectors, if someone in them does'not match
	// the kkt condition, then we choose it as the alpha_i that stays to be optimized
	bool inner_loop(int i);					

	// slect alpha_j given alpha_i
	// normally, we search the alpha which can maximum |error_i - error_j|
	int  select_j(int i);		

	// update support vector indexes
	void update_support_vector_index(); 

	// update vector index as a result of updating alpha_i & alpha_j
	void  update_bound_vector_index();		
	
	// calculate the kernel output gona to be used in prediction stage
	float kernel_cal(const std::vector<float> &x,
			const std::vector<float> &feature);  
};
