#include "svm.h"
#include <ctime>


int main(int argv, char* argc[]) {
	if (argv < 3) {
		cout << "Usage: ./svm [C] [max_iter_cnt] [feature_file]" << endl;
		return 0;
	}
	SVM *svm = new SVM(atoi(argc[1]), atoi(argc[2]));
	cerr << "Loading feature." << endl;
	svm->load_feature(argc[3]);
	cerr << "Loading feature done!" << endl;
	cerr << "Initializing parameters, e.g alpha, bias, error, kernel." << endl;
	svm->init_alpha();
	cerr << "Initializing parameters done!" << endl;
	cerr << "Training." << endl;
	svm->train();
	cerr << "Training Done!" << endl;
	svm->test();
	return 0;
}
