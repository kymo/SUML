#include "crf.h"


void test_crf(char *file) {

	CRF*crf = new CRF(4, 5000);
	crf->read_data(file);
	crf->train();
}

int main(int argc, char*argv[]) {
	test_crf(argv[1]);
	return 0;
}
