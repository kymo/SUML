#ifndef RF_H_
#define RF_H_
/*
 * Name : rf.h
 * Desc : random forest source code 
 * Date : 2014-10-13
 * Auth : aron
 */
#include <iostream>
#include <algorithm>
#include <map>
#include <vector>
#include <queue>
#include <time.h>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
using namespace std;

#define M 1024
#define N 1024
#define MTN 10
#define MLN 4

typedef struct node {
    float feat_value;
    int feat_index;
    bool is_leaf_node;
    float reg_value;
    int node_layer_index;
    vector<int> indexes;
    struct node *l_child;
    struct node *r_child;
    node () {}
    node (vector<int> _indexes, int _node_layer_index): l_child(NULL), r_child(NULL) {
        indexes = _indexes;
        node_layer_index = _node_layer_index;
    }
}TreeNode;

int random_int(int n, int seed) {
	long long multiplier = 0x5DEECE66DL, mask = (1L << 48) - 1, addend = 0xBL;
	if (n <= 1) return 0;
	if ((n & -n) == n) {
		return (int) ((n * (long) ((int) ((seed = (seed * multiplier + addend) & mask) >> 17))) >> 31);
	}
	int bits, val;
	do {
		bits = (int) ((seed = (seed * multiplier + addend) & mask) >> 17);
		val = bits % n;
	} while (bits - val + (n - 1) < 0);
	return val;
}

class CART {

private:
    TreeNode *root;
    float x[M][N];
    float y[M];
    int K;
	int i_seed;
    float calc_gini_v(int fea_index, float feat_value, const vector<int> &indexes);    // for regression
    float calc_max_variance(int feat_index, float feat_value, const vector<int> &indexes);    // for classification
    void find_best_feat(int &fea_index, float &fea_value, const vector<int> &indexes);
    void split_data(vector<int> &l_indexes, vector<int> &r_indexes, int feat_index, float feat_value, const vector<int> &indexes);
public:
    CART(){}
	~CART(){}
	CART(float *X[M], float *Y, int _K, int i); 
	void build_tree(TreeNode *root);
    float predict(float *f);
	float predict(TreeNode *root, float *f);
};


class RandomForest {

private:
    vector<CART*> trees;
	int tree_cnt_, node_cnt_;
public:
	RandomForest(int tree_cnt, int node_cnt) : tree_cnt_(tree_cnt), node_cnt_(node_cnt){}
    void add(CART *tree);
    int size();
    void clear();
    void train(float *x[M], float *y);
    float predict(float *f);
};

#endif
