
#include "Matrix.h"


namespace suml {
namespace matrix {

Matrix::Matrix(const std::vector<float> &data) {
	std::vector<float> tmp_value;
	for (int32_t i = 0; i < data.size(); i ++) {
		tmp_value.push_back(data[i]);
	}
	_data.push_back(tmp_value);
	_row = 1;
	_col = data.size();
	if (0 == _col) {
		std::clog << "Use a empty vector!" << std::endl;
	}
}

Matrix::Matrix(const std::vector<std::vector<float> > &data) {	
	_data = data;
	_row = data.size();
	if (data.size() == 0) {
		std::cerr << "Use a empty 2-dim vector!" << std::endl;
		exit(0);
	}
	_col = data[0].size();
}

Matrix Matrix::operator * (const Matrix &mat) const {
	int a_row = _row;
	int b_row = mat._row;
	if (0 == a_row || 0 == b_row) {
		std::cerr << "Multiply a valid Matrix!" << std::endl;
		exit(0);
	}
	int a_col = _col;
	int b_col = mat._col;

	if (a_col == 0 || b_col == 0) {
		std::cerr << "Multiply a vaid Matrix!" << std::endl;
	}
	if (a_col != b_row) {
		std::cerr << "Multiply two dis-match Matrix!" << std::endl;
		exit(0);
	}
	
	std::vector<std::vector<float> > results(a_row, std::vector<float>(b_col, 0.0));
	
	for (int32_t i = 0; i < a_row; i ++) {
		for (int32_t k = 0; k < a_col; k ++) {
			float tmp = _data[i][k];
			for (int32_t j = 0; j < b_col; j ++) {
				results[i][j] += tmp * mat(k, j);
			}
		}
	}

	return Matrix(results);

}


Matrix Matrix::operator + (const Matrix &mat) const {
	if (mat._row != _row || mat._col != _col) {
		std::cerr << "Add two mis-matched Matrix!" << std::endl;
		exit(0);
	}
	std::vector<std::vector<float> > results(_row, std::vector<float>(_col, 0.0));
	for (int32_t i = 0; i < _row; ++i) {
		for (int32_t j = 0; j < _col; ++j) {
			results[i][j] = _data[i][j] + mat(i, j);
		}
	}
	return Matrix(results);
}

Matrix Matrix::T() const{
	std::vector<std::vector<float> > result(_col, std::vector<float>(_row));
	for (int32_t i = 0; i < _row; ++ i) {
		for (int32_t j = 0; j < _col; ++j) {
			result[j][i] = _data[i][j];
		}
	}	
	return Matrix(result);
}

Matrix Matrix::I() const {
	if (_row != _col) {
		std::cerr << "Error When Calc The inverse Matrix: not a square matrix !" << std::endl;
		exit(0);
	}
}

Matrix Matrix::operator * (float ratio) const {
	std::vector<std::vector<float> > result(_row, std::vector<float>(_col));
	for (int32_t i = 0; i < _row; i ++) {
		for (int32_t j = 0; j < _col; j ++) {
			result[i][j] = _data[i][j] * ratio;
		}
	}
	return Matrix(result);
}

Matrix Matrix::operator - (const Matrix &mat) const {
	if (mat._row != _row || mat._col != _col) {
		std::cerr << "Error when minus the matrix!" << std::endl;
		exit(0);
	}
	std::vector<std::vector<float> >result(_row, std::vector<float>(_col));
	for (int32_t i = 0; i < _row; i ++) {
		for (int32_t j = 0; j < _col; j ++) {
			result[i][j] = _data[i][j] - mat(i, j);
		}
	}
	return Matrix(result);
}

float& Matrix::operator () (int i, int j) const {
	if (i >= _row || j >= _col) {
		std::cerr << "The matrix is smaller than you think, check you i,j!" << std::endl;
		exit(0);
	}
	float val = _data[i][j];
	return val;
}

std::vector<float> Matrix::row(int i) const {
	if (i >= _row) {
		std::cerr << "The Matrix Has More rows than i!" << std::endl;
		exit(0);
	}
	std::vector<float> result = _data[i];
	return result;
}

std::vector<float> Matrix::col(int i) const {
	if (i >= _col) {
		std::cerr << "The Matrix Has More Columns Than You Think!" << std::endl;
		exit(0);
	}
	std::vector<float> result;
	for (int32_t j = 0; j < _row; j ++) {
		result[j] = _data[j][i];
	}
	return result;
}


}
}
