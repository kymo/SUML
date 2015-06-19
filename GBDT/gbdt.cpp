
// gbdt implementation source file

#include "gbdt.h"

namespace gbdt {

// implementation of the functions of regression

template <class T>
void GradientBoostingRegressionTree<T>::train(std::vector<std::vector<float> > &trainingX,
		std::vector<T> &trainingY) {
	
	if (0 == trainingX.size()) {
		std::cerr << "Error Input Training Feature!" << std::endl;
		return ;
	}
	int32_t sampleCnt = trainingY.size();
	int32_t featureDim = trainingX[0].size();
	for (int32_t i = 0; i < sampleCnt; i ++) {
		this->m_vTempGradient.push_back(0.0);
		this->m_vGradient.push_back(trainingY[i]);
	}

	for (int32_t i = 0; i < this->m_nTreeNum; i ++) {	
		Tree<T>* subTree = new RegressionTree<T>(this->m_nTreeDepth, this->m_nNodeCnt, this->m_bMultiThreadOn);
		subTree->setData(trainingX, this->m_vGradient);

		subTree->train();
		
		for (int32_t j = 0; j < sampleCnt; j ++) {
			T value = subTree->predict(trainingX[j]);
			this->m_vTempGradient[j] += value;
			this->m_vGradient[j] = trainingY[j] - this->m_fLearningRate * this->m_vTempGradient[j];	
		}
		this->trees.push_back(subTree);
	}	
};

template <class T>
T GradientBoostingRegressionTree<T>::predict(const std::vector<T> &feature) {
	T value = 0;
	for (int j = 0; j < this->m_nTreeNum; j ++) {
		value += this->trees[j]->predict(feature);
	}
	return value * this->m_fLearningRate;
}


// implementation of the functions of classification

template <class T>
void GradientBoostingClassificationTree<T>::train() {
}

template <class T>
T GradientBoostingClassificationTree<T>::predict(const std::vector<float> &feature) {
}

};
