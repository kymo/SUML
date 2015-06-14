// source code for tree
#include "tree.h"


namespace gbdt {


template <class T>
std::vector<T>& Tree<T>::getTrainingY(){
	return m_vTrainingY;
}

template <class T>
std::vector<std::vector<float> > & Tree<T>::getTrainingX() {
	return m_vTrainingX;
}

template <class T>
int32_t& Tree<T>::getMaxNodeCnt() {
	return m_nMaxNodeCnt;
}

template <class T>
int32_t& Tree<T>::getMaxDepth() {
	return m_nMaxDepth;
}

template <class T>
struct Node<T>*& Tree<T>::getTreeRootNode() {
	return m_oTreeRootNode;
}

template <class T>
void Tree<T>::train() {
	std::vector<int32_t> vFeatureIndex, vCurrentIndex;
	for (int32_t i = 0; i < (int32_t)m_vTrainingX.size(); i ++) {
		vCurrentIndex.push_back(i);
	}
	for (int32_t i = 0; i < (int32_t)m_vTrainingX[0].size(); i ++) {
		vFeatureIndex.push_back(i);
	}
    
	buildTree(m_oTreeRootNode, 1, 1, 
				vCurrentIndex, 
				vFeatureIndex);
	//display();
}

template <class T>
void Tree<T>::display() {
	if (NULL == m_oTreeRootNode) {
		std::cout << "Empty Tree! Are U Kidding Me?" << std::endl;
		return ;
	}
	std::queue<Node<T> *> treeNodeQueue;
	treeNodeQueue.push(m_oTreeRootNode);
	std::map<Node<T> *, int32_t> levelMap;
	levelMap[m_oTreeRootNode] = -1;
	int cnt = 0;
	while (! treeNodeQueue.empty()) {
		Node<T> *top = treeNodeQueue.front();
		treeNodeQueue.pop();

		std::cout << "Node: " << top->index << ",Father " << levelMap[top] << std::endl;
		std::cout << "OptFeatureIndex: " << top->m_nCurrentOptSplitIndex << std::endl;
		std::cout << "CurrentOptSplitValue: " << top->m_fCurrentOptSplitValue << std::endl;
		std::cout << "Current Sample On this Node:" << std::endl;
		for (int32_t i = 0; i < top->m_vCurrentNodeTampleIndexVec.size(); i ++) {
			std::cout << top->m_vCurrentNodeTampleIndexVec[i] << " ";
		}
		std::cout << std::endl;
		std::cout << "Label: " << top->label << std::endl;

		if (top->m_oLeft) {
			levelMap[top->m_oLeft] = top->index;
			treeNodeQueue.push(top->m_oLeft);
		} 
		if (top->m_oRight) {
			levelMap[top->m_oRight] = top->index;
			treeNodeQueue.push(top->m_oRight);	
		}
	}	
}

template <class T>
void Tree<T>::sort_index_vec(std::vector<int32_t> &sort_index,
    const std::map<int32_t, float>& sort_value) {

	int32_t start = 0, end = sort_index.size() - 1;
    quickSort(sort_index, sort_value, start, end);
}

template <class T>
void Tree<T>::setData(std::vector<std::vector<float> > &vTrainingX,
		std::vector<T> &vTrainingY) {
	this->getTrainingX() = vTrainingX;
	this->getTrainingY() = vTrainingY;
}

void* selectFeatureFunc(void* param) {	
	ThreadParam<float> *p = (ThreadParam<float> *)param;
	std::map<int32_t, float> tempFeatureValue;
	std::vector<int32_t> vTempCurrentIndex(p->m_vCurrentIndex.begin(), p->m_vCurrentIndex.end());
	
	for (int32_t i = 0; i < p->m_vCurrentIndex.size(); i ++) {
		tempFeatureValue[p->m_vCurrentIndex[i]] = p->m_oTree->getTrainingX()[p->m_vCurrentIndex[i]][p->m_nFeatureIndex];
	}

	sortIndexVec(vTempCurrentIndex, tempFeatureValue);
	
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

template<class T>
void RegressionTree<T>::optSplitPosMultiThread(int &nOptFeatureIndex,
            float &fOptFeatureVal,
            std::vector<int32_t> &vCurrentIndex,
            std::vector<int32_t> &vFeatureIndex) {

	float minDevia = INT_MAX;
	int featureCnt = vFeatureIndex.size();
	std::vector<pthread_t> vThreadIds(featureCnt);
	for (int32_t i = 0; i < featureCnt; ++ i) {
		ThreadParam<T> *param = new ThreadParam<T>(this, vCurrentIndex, vFeatureIndex[i]);
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

template <class T>
void RegressionTree<T>::optSplitPos(int &nOptFeatureIndex,
            float &fOptFeatureVal,
            std::vector<int32_t> &vCurrentIndex,
            std::vector<int32_t> &vFeatureIndex) {	
    float minDevia = INT_MAX;
	for (int32_t i = 0; i < (int32_t)vFeatureIndex.size(); i ++) {
		std::map<int32_t, float> tmpFeatureValue;
		for (int32_t j = 0; j < vCurrentIndex.size(); j ++) {
			float tmpVal = this->getTrainingX()[vCurrentIndex[j]][vFeatureIndex[i]];
			tmpFeatureValue[vCurrentIndex[j]] = tmpVal;
		}
		
		std::vector<int32_t> vTempCurrentIndex(vCurrentIndex.begin(), vCurrentIndex.end());
		util::sortIndexVec(vTempCurrentIndex, tmpFeatureValue);
		
		float totValue = 0.0;
		float totSqaValue = 0.0;
		for (int32_t j = 0; j < (int32_t)vTempCurrentIndex.size(); j ++) {
			float tmpVal = this->getTrainingY()[vTempCurrentIndex[j]];
			totValue += tmpVal;
			totSqaValue += tmpVal * tmpVal;
		}
		float curTotVal = 0.0;
		float curTotSqaVal = 0.0;
		float minDeviaTemp = INT_MAX, featureVal = 0.0;
		for (int32_t j = 0; j < (int32_t)vTempCurrentIndex.size(); ++ j) {
			float tmpVal = this->getTrainingY()[vTempCurrentIndex[j]];
			curTotVal += tmpVal;
			curTotSqaVal += tmpVal * tmpVal;
			float curDevia = totSqaValue - curTotVal * curTotVal / (j + 1);
			if (j + 1 != (int32_t)vTempCurrentIndex.size()) {
				curDevia -= (totValue - curTotVal) * (totValue - curTotVal) / (vTempCurrentIndex.size() - j - 1);
			
			}
			//std::cout << vTempCurrentIndex.size() << " " << tmpFeatureValue.size() << std::endl;
			//std::cout << curDevia << " " << vFeatureIndex[i] << " " << tmpFeatureValue[vTempCurrentIndex[j]] << std::endl;
			if (curDevia < minDevia) {
				minDevia = curDevia;
				nOptFeatureIndex = vFeatureIndex[i];
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


template <class T>
void RegressionTree<T>::buildTree(struct Node<T>* &oTreeNode,
            int32_t curNodeIndex,
            int32_t curNodeLevel,
            std::vector<int32_t> &vCurrentIndex,
            std::vector<int32_t> &vFeatureIndex) {
	std::queue<Node<T>*> treeNodeQueues;
	oTreeNode = new Node<T>(vCurrentIndex, vFeatureIndex, 1, 1);
	treeNodeQueues.push(oTreeNode);
	int index = 1;
	
	while (! treeNodeQueues.empty()) {
		Node<T> *top = treeNodeQueues.front();
		treeNodeQueues.pop();
		if (top->level > this->getMaxDepth()) {
			continue;
		}
		if (top->index > this->getMaxNodeCnt()) {
			continue;
		}
		std::vector<int32_t> vTmpFeatureIndex(top->m_vFeatureIndexVec.begin(), top->m_vFeatureIndexVec.end());
		std::vector<int32_t> vTmpCurrentIndex(top->m_vCurrentNodeTampleIndexVec.begin(), top->m_vCurrentNodeTampleIndexVec.end());
		if (vTmpCurrentIndex.size() == 0) {
			continue;
		}
		
		int32_t nOptFeatureIndex;
		float   fOptFeatureVal;

		if (this->m_bMultiThreadOn) {
			optSplitPosMultiThread(nOptFeatureIndex, 
						fOptFeatureVal, 
						vTmpCurrentIndex,
						vTmpFeatureIndex);
		} else {

			optSplitPos(nOptFeatureIndex, 
					fOptFeatureVal, 
					vTmpCurrentIndex,
					vTmpFeatureIndex);	
		}
#ifdef DEBUG
		std::cout << "thread begin" << std::endl;
#endif

		float label = 0.0;
		for (int32_t i = 0; i < vTmpCurrentIndex.size(); i ++) {
			label += this->getTrainingY()[vTmpCurrentIndex[i]];
		}
		label /= (int32_t)vTmpCurrentIndex.size();
		top->m_nCurrentOptSplitIndex = nOptFeatureIndex;
		top->m_fCurrentOptSplitValue = fOptFeatureVal;
		top->label = label;

		std::vector<int32_t> vLeftIndex, vRightIndex;
		for (int32_t j = 0; j < vTmpCurrentIndex.size(); j ++) {
			if (this->getTrainingX()[vTmpCurrentIndex[j]][nOptFeatureIndex] <= fOptFeatureVal) {
				vLeftIndex.push_back(vTmpCurrentIndex[j]);
			} else {
				vRightIndex.push_back(vTmpCurrentIndex[j]);
			}
		}


		// wipe out the current feature
		vTmpFeatureIndex.erase(remove(vTmpFeatureIndex.begin(), vTmpFeatureIndex.end(), nOptFeatureIndex), 
					vTmpFeatureIndex.end());
		
		if (vLeftIndex.size() == 0 || vRightIndex.size() == 0) {
			continue ;
		}
		
		

		// build left Node
		if (top->level + 1 < this->getMaxDepth() 
				&& index < this->getMaxNodeCnt()
				&& top->m_vCurrentNodeTampleIndexVec.size() >= 4) {
			top->m_oLeft = new Node<T>(vLeftIndex, vTmpFeatureIndex, top->level + 1, index);
			index += 1;
			treeNodeQueues.push(top->m_oLeft);
		}
		if (top->level + 1 < this->getMaxDepth()
				&& index < this->getMaxNodeCnt()
				&& top->m_vCurrentNodeTampleIndexVec.size() >= 4) {
			top->m_oRight = new Node<T>(vRightIndex, vTmpFeatureIndex, top->level + 1, index);
			index += 1;
			treeNodeQueues.push(top->m_oRight);
		}
		// build right Node
	}
}


//template <class T>
//void RegressionTree<T>::buildTree(struct Node<T>* &oTreeNode,
//            int32_t curNodeIndex,
//            int32_t curNodeLevel,
//            std::vector<int32_t> &vCurrentIndex,
//            std::vector<int32_t> &vFeatureIndex,
//            const std::vector<std::vector<float> >&vCurrentNodeTrainingX,
//                const std::vector<T>&vCurrentNodeTrainingY) {
//    
//    if (curNodeIndex > m_nMaxNodeCnt) return;
//    if (curNodeLevel > m_nMaxDepth) return;
//	if (vCurrentIndex.size() == 0 || vFeatureIndex.size() == 0) return ;
//
//    int32_t nOptFeatureIndex;
//    float   fOptFeatureVal;
//
//    optSplitPos(nOptFeatureIndex, fOptFeatureVal, vCurrentIndex, vFeatureIndex, vCurrentNodeTrainingX, vCurrentNodeTrainingY);
//	
//	// now the vCurrentIndex vector has been sorted by function optSplitPos
//    
//	std::cout << nOptFeatureIndex << " " << fOptFeatureVal << std::endl;
//	
//	std::vector<int32_t> vLeftIndex, vRightIndex;
//    for (int32_t j = 0; j < vCurrentIndex.size(); j ++) {
//		if (vCurrentNodeTrainingX[vCurrentIndex[j]][nOptFeatureIndex] <= fOptFeatureVal) {
//			vLeftIndex.push_back(vCurrentIndex[j]);
//		} else {
//			vRightIndex.push_back(vCurrentIndex[j]);
//		}
//	}
//
//
//	// wipe out the current feature
//	vFeatureIndex.erase(remove(vFeatureIndex.begin(), vFeatureIndex.end(), nOptFeatureIndex), 
//				vFeatureIndex.end());
//	float label = 0.0;
//	for (int32_t i = 0; i < vCurrentIndex.size(); i ++) {
//		label += vCurrentNodeTrainingY[vCurrentIndex[i]];
//	}
//	label /= vCurrentIndex.size();
//	oTreeNode = new Node<T>(nOptFeatureIndex, fOptFeatureVal, vCurrentIndex, label);
//
//	if (vLeftIndex.size() == 0 || vRightIndex.size() == 0) return ;
//	
//	// build left Node
//	
//    buildTree(oTreeNode->m_oLeft, curNodeIndex + 1, curNodeLevel + 1, vLeftIndex, vFeatureIndex,
//                vCurrentNodeTrainingX, vCurrentNodeTrainingY);
//    // build right Node
//    buildTree(oTreeNode->m_oRight, curNodeIndex + 2, curNodeLevel + 1, vRightIndex, vFeatureIndex,
//                vCurrentNodeTrainingX, vCurrentNodeTrainingY);
//}
//*/

template <class T>
T RegressionTree<T>::predict( const std::vector<float> &testFeatureX) {
    Node<T>* oTreeNode = this->getTreeRootNode();
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


template <class T>
void ClassificationTree<T>::optSplitPos(int &nOptFeatureIndex,
            float &nOptFeatureVal,
            std::vector<int32_t> &vCurrentIndex,
            std::vector<int32_t> &vFeatureIndex) {
}

template <class T>
void ClassificationTree<T>::buildTree(Node<T>* &oTreeNode) {

}

template <class T>
T ClassificationTree<T>::predict(const std::vector<float> &testFeatureX) {

}




}
