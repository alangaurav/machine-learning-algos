import math

def calculate_entropy(target):
    count = len(target)
    probabilities = dict()
    for ele in target:
        probabilities[ele] = probabilities.get(ele, 0) + 1
    probabilities = {k: v/count for k,v in probabilities.items()}
    entropy = 0
    for v in probabilities.values():
        entropy -= v * math.log2(v)

    return entropy

def calculate_information_gain(target, feature, root_entropy):
    feature_groups = set(feature)
    count = len(feature)
    probabilities = dict()
    
    for ele in feature:
        probabilities[ele] = probabilities.get(ele, 0) + 1
    probabilities = {k: v/count for k,v in probabilities.items()}

    information_gain = root_entropy
    for group in feature_groups:
        target_subset = list()
        for i in range(len(target)): 
            if feature[i] == group:
                target_subset.append(target[i])
        entropy = calculate_entropy(target_subset)
        information_gain -= probabilities.get(group, 0) * entropy

    return information_gain

def calculate_gini(target):
    count = len(target)
    probabilities = dict()
    for ele in target:
        probabilities[ele] = probabilities.get(ele, 0) + 1
    probabilities = {k: v/count for k, v in probabilities.items()}
    gini = 1
    for v in probabilities.values():
        gini -= v*v

    return gini

def calculate_gini_gain(target, feature, root_gini):
    feature_groups = set(feature)
    count = len(feature)
    probabilities = dict()
    
    for ele in feature:
        probabilities[ele] = probabilities.get(ele, 0) + 1
    probabilities = {k: v/count for k,v in probabilities.items()}

    gini = 0
    for group in feature_groups:
        feature_subset = list()
        for i in range(len(target)): 
            if feature[i] == group:
                feature_subset.append(feature[i])
        gini += probabilities.get(group, 0) * calculate_gini(feature_subset)

    gini_gain = root_gini - gini

    return gini_gain