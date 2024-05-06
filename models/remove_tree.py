import json

def rmv_tree( file: str, index: int):

    f = open(file, 'r')

    data = json.load(f)

    f.close()

    learner = data['learner']
    gradient_booster = learner['gradient_booster']
    model = gradient_booster['model']
    gbtree_model_param = model['gbtree_model_param']
    tree_info = model['tree_info'] 
    trees = model['trees'] 

    gbtree_model_param['num_trees'] = str(int(gbtree_model_param['num_trees']) - 1)

    removed_tree = trees[index]
    del tree_info[index]
    del trees[index]

    for idx, tree in enumerate(trees):
        tree['id'] = idx

    model['trees'] = trees
    model['tree_info'] = tree_info
    gradient_booster['model'] = model
    learner['gradient_booster'] = gradient_booster
    data['learner'] = learner

    f = open(file, 'w')
    json.dump(data, f)

    f.close()

    return removed_tree