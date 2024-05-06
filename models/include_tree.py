import json

def incl_tree( file: str, new_trees: object):

    f = open(file, 'r')

    data = json.load(f)

    f.close()

    learner = data['learner']
    gradient_booster = learner['gradient_booster']
    model = gradient_booster['model']
    gbtree_model_param = model['gbtree_model_param']
    tree_info = model['tree_info']
    trees = model['trees']

    gbtree_model_param['num_trees'] = str(int(gbtree_model_param['num_trees']) + len(new_trees))

    for tree in new_trees:
        trees.append(tree)

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