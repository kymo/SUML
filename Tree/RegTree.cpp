// source code for tree


#include "RegTree.h"

namespace suml {

namespace tree {

void* selectFeatureFunc(void* param) {	
	
	suml::basic::ThreadParam<float> *p = (suml::basic::ThreadParam<float> *)param;
	std::map<int32_t, float> tempFeatureValue;
	std::vector<int32_t> vTempCurrentIndex(p->m_vCurrentIndex.begin(), p->m_vCurrentIndex.end());
	
	for (int32_t i = 0; i < p->m_vCurrentIndex.size(); i ++) {
		tempFeatureValue[p->m_vCurrentIndex[i]] = p->m_oTree->getTrainingX()[p->m_vCurrentIndex[i]][p->m_nFeatureIndex];
	}

	p->m_oTree->sortIndexVec(vTempCurrentIndex, tempFeatureValue);
	
	float totValue = 0.0;
	float totSqaValue = 0.0;
	
	for (int32_t j = 0; j < (int32_t)vTempCurrentIndex.size(); j ++) {
		float tmpVal = p->m_oTree->getTrainingY()[vTempCurrentIndex[j]];
		totValue += tmpVal;
		totSqaValue += tmpVal * tmpVal;
	}

	float fOptFeatureVal = 0.0;	
	float curTotVal = 0.0;
	float curTotSqaVal = 0.0;
	float minDevia = INT_MAX;
	for (int32_t j = 0; j < (int32_t)vTempCurrentIndex.size(); ++ j) {
		float tmpVal = p->m_oTree->getTrainingY()[vTempCurrentIndex[j]];
		curTotVal += tmpVal;
		curTotSqaVal += tmpVal * tmpVal;
		float curDevia = totSqaValue - curTotVal * curTotVal / (j + 1);
		if (j + 1 != (int32_t)vTempCurrentIndex.size()) {
			curDevia -= (totValue - curTotVal) * (totValue - curTotVal) / (vTempCurrentIndex.size() - j - 1);
		}
		if (curDevia < minDevia) {
			minDevia = curDevia;
			fOptFeatureVal = tempFeatureValue[vTempCurrentIndex[j]];
		}
	}
	char ret[64];
	sprintf(ret, "%f+%f", fOptFeatureVal, minDevia);
	return (void*)ret;

}	

void RegressionTree::optSplitPosMultiThread(int &nOptFeatureIndex,
            float &fOptFeatureVal,
            std::vector<int32_t> &vCurrentIndex,
            std::vector<int32_t> &vFeatureIndex) {

	float minDevia = INT_MAX;
	int featureCnt = vFeatureIndex.size();
	std::vector<pthread_t> vThreadIds(featureCnt);
	for (int32_t i = 0; i < featureCnt; ++ i) {
		suml::basic::ThreadParam<float> *param = new suml::basic::ThreadParam<float>(this, vCurrentIndex, vFeatureIndex[i]);
		pthread_create(&vThreadIds[i], NULL, selectFeatureFunc, (void*)param);
	}

	std::vector<char*> vFeatureValue(featureCnt);
	for (int32_t i = 0; i < featureCnt; ++ i) {

		pthread_join(vThreadIds[i], (void**)&vFeatureValue[i]);

		char* optValue =  strtok(vFeatureValue[i], "+");
		char* devia    = strtok(NULL, "+");
		
		float fOptValue = atof(optValue);
		float fDevia = atof(devia);
		
		if (fDevia < minDevia) {
			minDevia = fDevia;
			nOptFeatureIndex = vFeatureIndex[i];
			fOptFeatureVal = fOptValue;
		}
	}
}

void RegressionTree::optSplitPos(int &nOptFeatureIndex,
            float &fOptFeatureVal,
            std::vector<int32_t> &vCurrentIndex,
            std::vector<int32_t> &vFeatureIndex) {	
    
	float minDevia = INT_MAX;
	
	// sample the feature
	std::vector<int32_t> vTempFeatureIndex;
	
	if (getEnsemble()) {
		std::cout << "fuck" << std::endl;
		srand( (unsigned)time(NULL));
		for (int32_t i = 0; i < vFeatureIndex.size(); ++i) {
			int32_t t = rand() % vFeatureIndex.size();
			std::swap(vFeatureIndex[i], vFeatureIndex[t]);
		}
	}
		
	vTempFeatureIndex.assign(vFeatureIndex.begin(),	vFeatureIndex.begin() + getRandFeatureCnt());
	std::cout << getRandFeatureCnt() << std::endl;
	for (int i = 0; i < vTempFeatureIndex.size();  ++i) {
		std::cout << vTempFeatureIndex[i] << " ";
	}
	std::cout << std::endl;

	for (int32_t i = 0; i < (int32_t)vTempFeatureIndex.size(); i ++) {
		
		std::map<int32_t, float> tmpFeatureValue;
		for (int32_t j = 0; j < vCurrentIndex.size(); j ++) {
			float tmpVal = getTrainingX()[vCurrentIndex[j]][vTempFeatureIndex[i]];
			tmpFeatureValue[vCurrentIndex[j]] = tmpVal;
		}
		
		std::vector<int32_t> vTempCurrentIndex(vCurrentIndex.begin(), vCurrentIndex.end());
		sortIndexVec(vTempCurrentIndex, tmpFeatureValue);
		float totValue = 0.0;
		float totSqaValue = 0.0;
		for (int32_t j = 0; j < (int32_t)vTempCurrentIndex.size(); j ++) {
			float tmpVal = getTrainingY()[vTempCurrentIndex[j]];
			totValue += tmpVal;
			totSqaValue += tmpVal * tmpVal;
		}
		float curTotVal = 0.0;
		float curTotSqaVal = 0.0;
		float minDeviaTemp = INT_MAX, featureVal = 0.0;
		for (int32_t j = 0; j < (int32_t)vTempCurrentIndex.size(); ++ j) {
			
			float tmpVal = getTrainingY()[vTempCurrentIndex[j]];
			curTotVal += tmpVal;
			curTotSqaVal += tmpVal * tmpVal;
			
			float curDevia = totSqaValue - curTotVal * curTotVal / (j + 1);
			
			if (j + 1 != (int32_t)vTempCurrentIndex.size()) {
				curDevia -= (totValue - curTotVal) *
				   		(totValue - curTotVal) / 
						(vTempCurrentIndex.size() - j - 1);
			}
#ifdef DEBG
			std::cout << vTempCurrentIndex.size() << " " << tmpFeatureValue.size() << std::endl;
			std::cout << curDevia << " " << vTempFeatureIndex[i] << " " << tmpFeatureValue[vTempCurrentIndex[j]] << std::endl;
#endif
			if (curDevia < minDevia) {
				minDevia = curDevia;
				nOptFeatureIndex = vTempFeatureIndex[i];
				fOptFeatureVal = tmpFeatureValue[vTempCurrentIndex[j]];
			}
#ifdef DEBUG
			if (curDevia < minDeviaTemp) {
				minDeviaTemp = curDevia;
				featureVal = tmpFeatureValue[vTempCurrentIndex[j]];
			}
#endif
		}

	}
}

void RegressionTree::splitData(suml::basic::Node<float>* &top,
		const int &nOptFeatureIndex,
		const float &fOptFeatureVal,
		const std::vector<int32_t> &vTmpCurrentIndex,
   		std::vector<int32_t> &vLeftIndex,
		std::vector<int32_t> &vRightIndex) {

	float label = 0.0;
	for (int32_t i = 0; i < vTmpCurrentIndex.size(); i ++) {
		label += getTrainingY()[vTmpCurrentIndex[i]];
	}
	label /= (int32_t)vTmpCurrentIndex.size();
	
	top->m_nCurrentOptSplitIndex = nOptFeatureIndex;
	top->m_fCurrentOptSplitValue = fOptFeatureVal;
	top->label = label;

	for (int32_t j = 0; j < vTmpCurrentIndex.size(); j ++) {
		if (getTrainingX()[vTmpCurrentIndex[j]][nOptFeatureIndex] <= fOptFeatureVal) {
			vLeftIndex.push_back(vTmpCurrentIndex[j]);
		} else {
			vRightIndex.push_back(vTmpCurrentIndex[j]);
		}
	}
}

float RegressionTree::predict( const std::vector<float> &testFeatureX) {
    
	suml::basic::Node<float>* oTreeNode = getTreeRootNode();
	while (true) {
        
		if (NULL == oTreeNode->m_oLeft && NULL == oTreeNode->m_oRight) {
            return oTreeNode->label;
        }
        
		if (testFeatureX[oTreeNode->m_nCurrentOptSplitIndex] <= oTreeNode->m_fCurrentOptSplitValue) {
            if (NULL == oTreeNode->m_oLeft) {
				return oTreeNode->label;
            } else {
                oTreeNode = oTreeNode->m_oLeft;
            }
        } else {
            if (NULL == oTreeNode->m_oRight) {
                return oTreeNode->label;
            } else {
                oTreeNode = oTreeNode->m_oRight;
            }
        }

    }

}

}
}
