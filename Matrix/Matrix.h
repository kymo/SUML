#ifndef __MATRIX_H_
#define __MATRIX_H_

#include <iostream>
#include <string>
#include <vector>
#include <cmath>

namespace suml {
namespace matrix {

class Matrix {
public:
	std::vector<std::vector<float> > _data;
	int32_t _col;
	int32_t _row;

	Matrix() {}
	~Matrix() {}

	Matrix(const std::vector<float> &data);
	Matrix(const std::vector<std::vector<float> > &data);
	
	std::vector<std::vector<float> > &get_data();

	Matrix operator * (const Matrix &a) const;
	Matrix operator + (const Matrix &a) const;
	Matrix operator * (float ratio) const;
	Matrix operator - (const Matrix &a) const;

	std::vector<float>& operator [] (int index) const;
	std::vector<float> row(int i) const;
	std::vector<float> col(int i) const;

	float& operator () (int i, int j) const;

	Matrix T() const;
	Matrix I() const;

};

}
}
#endif
