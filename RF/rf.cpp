/* Name : rf.cpp
 * Desc : random forest souce code
 * Date : 2014-10-13
 * Auth : aron
 */

#include "rf.h"


CART::CART(float *X[M], float *Y, int _K, int i) {
	memcpy(x, X, sizeof(float) * M * N);
	memcpy(y, Y, sizeof(float) * M);
	K = _K;
	i_seed = i;
	build_tree(root);
}

void CART::build_tree(TreeNode *tnode) {
    queue<TreeNode*> node_queue;   
    int cur_node_index = 0;
    int cur_node_layer_index = 0;
    // random n times to get the sample
    vector<int> indexes;
    for (int i = 0; i < N; i ++) {
        indexes.push_back(random_int(N, i_seed + 20141015));
    }
    tnode = new TreeNode(indexes, 1);
    while (!node_queue.empty()) {
        if (cur_node_index > MTN) {
            break;
        }
        node *tp = node_queue.front();
        node_queue.pop();
        int feat_index;
        float feat_value;
        vector<int> l_indexes, r_indexes;
        find_best_feat(feat_index, feat_value, tp->indexes);
        split_data(l_indexes, r_indexes, feat_index, feat_value, tp->indexes);
        
        float tot = 0.0;
        for (int j = 0; j < tp->indexes.size(); j ++) {
            tot += y[tp->indexes[j]];
        }
        if (tp->node_layer_index == MLN) {
            tp->is_leaf_node = true;
        }
        else {
            tp->is_leaf_node = false;
        }
        
        tp->l_child = new TreeNode(l_indexes, tp->node_layer_index + 1);
        tp->r_child = new TreeNode(r_indexes, tp->node_layer_index + 1);
        node_queue.push(tp->l_child);
        node_queue.push(tp->r_child);
    }
}


void CART::find_best_feat(int &feat_index, float &feat_value,const vector<int> &indexes) {
    float gini_value = 0.0;
    
    // random k features without reture back
    set<int> f_indexes;
    srand(time(NULL));
    while (f_indexes.size() < K) {
        f_indexes.insert(random_int(N, i_seed + 20141013));
    }

    for (set<int>::iterator ite = f_indexes.begin(); ite != f_indexes.end(); ite ++) {
        set<int> feat_values;
        for (int i = 0; i < indexes.size(); i ++) {
            feat_values.insert(x[indexes[i]][*ite]);
        }
        for (set<int>::iterator v = feat_values.begin(); v!= feat_values.end(); v ++) {
            float tmp_variance_v = calc_max_variance(*ite, *v, indexes);
            if (gini_value < tmp_variance_v) {
                gini_value = tmp_variance_v;
                feat_index = *ite;
                feat_value = *v;
            }
        }
    }
}


float CART::calc_gini_v(int feat_index, float feat_value, const vector<int> &indexes) {

}


float CART::calc_max_variance(int feat_index, float feat_value,const vector<int> &indexes) {
    float avg = 0.0, n = indexes.size();
    float avg_left = 0.0, avg_right = 0.0;
    int left_cnt = 0, right_cnt = 0;
    float varia_total = 0.0, varia_left = 0.0, varia_right = 0.0;
    for (int i = 0; i < n; i ++) {
        avg += x[indexes[i]][feat_index];
        if (x[indexes[i]][feat_index] < feat_value) {
            avg_left += x[indexes[i]][feat_index];
            left_cnt += 1;
        }
        else {
            avg_right += x[indexes[i]][feat_index];
            right_cnt += 1;
        }
    }
    avg /= n;
    avg_left /= left_cnt;
    avg_right /= right_cnt;
    for (int i = 0; i < n; i ++) {
        float tmp_feat_value = x[indexes[i]][feat_index];
        varia_total += (avg - tmp_feat_value) * (avg - tmp_feat_value);
        if (tmp_feat_value < feat_value) {
            varia_left += (avg_left - tmp_feat_value) * (avg_left - tmp_feat_value);
        }
        else {
            varia_right += (avg_right - tmp_feat_value) * (avg_right - tmp_feat_value);
        }
    }
    return varia_total - varia_left - varia_right;
}


void CART::split_data(vector<int> &l_indexes, vector<int> &r_indexes, int feat_index, float feat_value,const vector<int> &indexes) {
    for (int i = 0; i < indexes.size(); i ++) {
        if (x[indexes[i]][feat_index] < feat_value) {
            l_indexes.push_back(indexes[i]);
        }
        else {
            r_indexes.push_back(indexes[i]);
        }
    }
}


float CART::predict(float *f) {
    return predict(root, f);
}

float CART::predict(TreeNode *root, float *f) {
    if (root->is_leaf_node) {
        return root->reg_value;
    }
    if (f[root->feat_index] < root->feat_value) {
        return predict(root->l_child, f);
    }
    else {
        return predict(root->r_child, f);
    }
}






void RandomForest::add(CART *tree) {
    trees.push_back(tree);
}

int RandomForest::size() {
    return trees.size();
}

void RandomForest::train(float *X[M], float *Y) {
    for (int j = 0; j < tree_cnt_; j ++) {
        add(new CART(X, Y, node_cnt_, j));
    }
}

float RandomForest::predict(float *f) {
    float tot = 0;
    for (int j = 0; j < tree_cnt_; j ++) {
        tot += trees[j]->predict(f);
    }
    return tot / tree_cnt_;
}

void RandomForest::clear() {
    trees.clear();
}
