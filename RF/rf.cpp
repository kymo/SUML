

#include "rf.h"



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
