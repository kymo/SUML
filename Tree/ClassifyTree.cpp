// source code for tree


#include "ClassifyTree.h"

namespace suml {
namespace tree {

void* selectFeatureFuncC(void* param) {	
	
	suml::basic::ThreadParam<int32_t> *p = (suml::basic::ThreadParam<int32_t> *)param;
	std::map<int32_t, float> tempFeatureValue;
	std::vector<int32_t> vTempCurrentIndex(p->m_vCurrentIndex.begin(), p->m_vCurrentIndex.end());
	
	for (int32_t i = 0; i < p->m_vCurrentIndex.size(); i ++) {
		tempFeatureValue[p->m_vCurrentIndex[i]] = p->m_oTree->getTrainingX()[p->m_vCurrentIndex[i]][p->m_nFeatureIndex];
	}

	std::set<int32_t> featureValueSet;
	std::map<int32_t, int32_t> featureValCnt;
	
	std::vector<int32_t> labelCnt(p->m_oTree->getLabelCnt(), 0);
	std::map<int32_t, std::vector<int32_t> > featureLabelCnt;

	for (int j = 0; j < p->m_vCurrentIndex.size(); ++j) {
		int32_t val = (int32_t)p->m_oTree->getTrainingX()[p->m_vCurrentIndex[j]][p->m_nFeatureIndex];
		int32_t label = (int32_t)p->m_oTree->getTrainingY()[p->m_vCurrentIndex[j]];

		featureValueSet.insert(val);

		if (featureValCnt.find(val) == featureValCnt.end()) {
		
			featureValCnt[val] = 1;
			std::vector<int32_t> t(p->m_oTree->getLabelCnt(), 0);
			featureLabelCnt[val] = t;
			featureLabelCnt[val][label] += 1;
		
		} else {
		
			featureValCnt[val] += 1;
			featureLabelCnt[val][label] += 1;
		
		}

		labelCnt[label] += 1;
	}	
	
	float minDevia = INT_MAX;
	float fOptFeatureVal = 0.0;
	int32_t totLabelCnt = p->m_vCurrentIndex.size();
	for (std::set<int32_t>::iterator it = featureValueSet.begin(); it != featureValueSet.end(); ++it) {
		int32_t val = *it;
		int cnt = featureValCnt[val];
		
		std::vector<int32_t> labelNum = featureLabelCnt[val];
		float gini = 0.0;
		float rTot = 0.0;
		float lTot = 0.0;
		for (int32_t k = 0; k < p->m_oTree->getLabelCnt(); ++k) {

			lTot += labelNum[k] * labelNum[k] * 1.0 / (cnt * cnt);
			rTot += (labelCnt[k] - labelNum[k]) * (labelCnt[k] - labelNum[k]) * 1.0 /
					((totLabelCnt - cnt) * (totLabelCnt - cnt));
		
		}
		gini = (1 - lTot) * cnt * 1.0 + (1 - rTot) * (totLabelCnt - cnt) * 1.0;
		gini = gini / totLabelCnt;
		if (gini < minDevia) {
			gini = minDevia;
			fOptFeatureVal = val;
		}
	}

	char ret[64];
	sprintf(ret, "%f+%f", fOptFeatureVal, minDevia);
	return (void*)ret;
}	

void ClassificationTree::optSplitPosMultiThread(int &nOptFeatureIndex,
            float &fOptFeatureVal,
            std::vector<int32_t> &vCurrentIndex,
            std::vector<int32_t> &vFeatureIndex) {

	float minDevia = INT_MAX;
	int featureCnt = vFeatureIndex.size();
	std::vector<pthread_t> vThreadIds(featureCnt);
	for (int32_t i = 0; i < featureCnt; ++ i) {
		suml::basic::ThreadParam<float> *param = new suml::basic::ThreadParam<float>(this, vCurrentIndex, vFeatureIndex[i]);
		pthread_create(&vThreadIds[i], NULL, selectFeatureFuncC, (void*)param);
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

void ClassificationTree::optSplitPos(int &nOptFeatureIndex,
            float &fOptFeatureVal,
            std::vector<int32_t> &vCurrentIndex,
            std::vector<int32_t> &vFeatureIndex) {	
    
	float minDevia = INT_MAX;

	// sample the feature
	std::vector<int32_t> vTempFeatureIndex;

	if (getEnsemble()) {
		srand( (unsigned)time(NULL));
		for (int32_t i = 0; i < vFeatureIndex.size(); ++i) {
			int32_t t = rand() % vFeatureIndex.size();
			std::swap(vFeatureIndex[i], vFeatureIndex[t]);
		}
		vTempFeatureIndex.assign(vFeatureIndex.begin(),	vFeatureIndex.begin() + getRandFeatureCnt());
	} else {
		vTempFeatureIndex.assign(vFeatureIndex.begin(), vFeatureIndex.end());
	}

	// gini data
	int32_t totLabelCnt = vCurrentIndex.size();

	for (int i = 0; i < vTempFeatureIndex.size(); ++i) {

		std::set<int32_t> featureValueSet;
		std::map<int32_t, int32_t> featureValCnt;
		
		std::vector<int32_t> labelCnt(getLabelCnt(), 0);
		std::map<int32_t, std::vector<int32_t> > featureLabelCnt;

		for (int j = 0; j < vCurrentIndex.size(); ++j) {
			
			int32_t val = (int32_t)getTrainingX()[vCurrentIndex[j]][vTempFeatureIndex[i]];
			int32_t label = (int32_t)getTrainingY()[vCurrentIndex[j]];

			featureValueSet.insert(val);

			if (featureValCnt.find(val) == featureValCnt.end()) {
			
				featureValCnt[val] = 1;
				std::vector<int32_t> t(getLabelCnt(), 0);
				featureLabelCnt[val] = t;
				featureLabelCnt[val][label] += 1;
			
			} else {
			
				featureValCnt[val] += 1;
				featureLabelCnt[val][label] += 1;
			
			}

			labelCnt[label] += 1;
		}	
		
		for (std::set<int32_t>::iterator it = featureValueSet.begin(); it != featureValueSet.end(); ++it) {
			int32_t val = *it;
			int cnt = featureValCnt[val];
			
			std::vector<int32_t> labelNum = featureLabelCnt[val];
			float gini = 0.0;
			float rTot = 0.0;
			float lTot = 0.0;
			for (int32_t k = 0; k < getLabelCnt(); ++k) {

				lTot += labelNum[k] * labelNum[k] * 1.0 / (cnt * cnt);
				rTot += (labelCnt[k] - labelNum[k]) * (labelCnt[k] - labelNum[k]) * 1.0 /
						((totLabelCnt - cnt) * (totLabelCnt - cnt));
			
			}
			gini = (1 - lTot) * cnt * 1.0 + (1 - rTot) * (totLabelCnt - cnt) * 1.0;
			gini = gini / totLabelCnt;
			if (gini < minDevia) {
				gini = minDevia;
				fOptFeatureVal = val;
				nOptFeatureIndex = vTempFeatureIndex[i];
			}
		}
	}	
}

void ClassificationTree::splitData(suml::basic::Node<float>* &top,
		const int &nOptFeatureIndex,
		const float &fOptFeatureVal,
		const std::vector<int32_t> &vTmpCurrentIndex,
   		std::vector<int32_t> &vLeftIndex,
		std::vector<int32_t> &vRightIndex) {
	
	std::map<int32_t, int32_t> labelCnt;
	int32_t cnt = 0, label;
	
	for (int32_t i = 0; i < vTmpCurrentIndex.size(); i ++) {
		int32_t tmpLabel = getTrainingY()[vTmpCurrentIndex[i]];
		
		if (labelCnt.find(tmpLabel) == labelCnt.end()) {
			labelCnt[tmpLabel] = 1;
		} else {
			labelCnt[tmpLabel] += 1;
		}
		
		if (cnt < labelCnt[tmpLabel]) {
			label = tmpLabel;
			cnt = labelCnt[tmpLabel];
		}
	}

	
	top->m_nCurrentOptSplitIndex = nOptFeatureIndex;
	top->m_fCurrentOptSplitValue = fOptFeatureVal;
	top->label = label;

	for (int32_t j = 0; j < vTmpCurrentIndex.size(); j ++) {
		if (getTrainingX()[vTmpCurrentIndex[j]][nOptFeatureIndex] != fOptFeatureVal) {
			vLeftIndex.push_back(vTmpCurrentIndex[j]);
		} else {
			vRightIndex.push_back(vTmpCurrentIndex[j]);
		}
	}
}

float ClassificationTree::predict( const std::vector<float> &testFeatureX) {
    
	suml::basic::Node<float>* oTreeNode = getTreeRootNode();
	while (true) {
        
		if (NULL == oTreeNode->m_oLeft && NULL == oTreeNode->m_oRight) {
            return oTreeNode->label;
        }
        
		if (testFeatureX[oTreeNode->m_nCurrentOptSplitIndex] != oTreeNode->m_fCurrentOptSplitValue) {
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
