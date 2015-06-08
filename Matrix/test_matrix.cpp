#include "Matrix.h"

using namespace std;
using namespace suml::matrix;
int main() {	
	int a, b;
	vector<float> A, B;
	for (int j = 0; j < 2; j++) {
		float tmp;
		cin >> tmp;
		A.push_back(tmp);
		B.push_back(-tmp);
	}

	Matrix t(A);
	Matrix ret = t + Matrix(B);
	Matrix trans = t.T();
	Matrix resuls = trans * Matrix(B);
	
	Matrix rets = resuls * 0.2;	
	cout << (rets(1, 0)) << endl;
	return 0;

}
