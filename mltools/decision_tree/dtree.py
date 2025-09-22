from mltools.decision_tree.tree import NAryTree
from mltools.decision_tree.impurity import calculate_entropy, calculate_information_gain, calculate_gini, calculate_gini_gain

def split_data(target, feature_data, feature, value):
        feature_data_copy = feature_data.copy()
        feature_values = feature_data_copy.pop(feature)
        target_subset = list()
        feature_subset = {k: list() for k in feature_data}
        feature_subset.pop(feature)
        for i in range(len(feature_values)):
            if feature_values[i] == value:
                target_subset.append(target[i])
                for k in feature_subset:
                    feature_subset[k].append(feature_data[k][i])
                
        return feature_subset, target_subset

class ID3:
    def __init__(self):
        self.tree = None
        self.target = None
        self.feature_data = None

    def set_feature_data(self, features):
        self.feature_data = features

    def set_target_data(self, target):
        self.target = target    

    def build_tree(self):
        # Identify root node
        root_entropy = calculate_entropy(self.target)
        root_feature = None
        max_ig = 0
        for feature, data in self.feature_data.items(): # type: ignore
            ig = calculate_information_gain(self.target, data, root_entropy)
            if ig > max_ig:
                max_ig = ig
                root_feature = feature
        
        self.tree = NAryTree(root_feature, max_ig, None, "root", self.feature_data, self.target)
        feature_groups = set(self.feature_data.get(root_feature)) # type: ignore
        for group in feature_groups:
            child_feature_data, child_target = split_data(self.target, self.feature_data, root_feature, group)
            node = self.recursive_build(group, child_feature_data, child_target, group)
            if node is not None:
                self.tree.insert_child(node)

    def recursive_build(self, criterion, feature_data, target, group):
        # Check if the node should be a leaf-node
        if len(set(target)) == 1:
            root = NAryTree(None, None, criterion, 'leaf', None, None)
            root.root.decision = target[0]
            return root
        
        # Check if there are features remaining
        if len(feature_data) == 0:
            return None
        
        root_entropy = calculate_entropy(target)
        root_feature = None
        max_ig = 0
        for feature, data in feature_data.items():
            ig = calculate_information_gain(target, data, root_entropy)
            if ig > max_ig:
                max_ig = ig
                root_feature = feature
        
        root = NAryTree(root_feature, max_ig, group, "internal", feature_data, target)
        feature_groups = set(feature_data.get(root_feature))
        for group in feature_groups:
            child_feature_data, child_target = split_data(target, feature_data, root_feature, group)
            node = self.recursive_build(group, child_feature_data, child_target, group)
            if node is not None:
                self.tree.insert_child(node) # type: ignore

        return root

class CART:
    def __init__(self):
        self.tree = None
        self.target = None
        self.feature_data = None

    def set_feature_data(self, features):
        self.feature_data = features

    def set_target_data(self, target):
        self.target = target

    def build_tree(self):
        # Identify root node
        root_gini = calculate_gini(self.target)
        root_feature = None
        max_gg = 0
        for feature, data in self.feature_data.items(): # type: ignore
            gg = calculate_gini_gain(self.target, data, root_gini)
            if gg > max_gg:
                max_gg = gg
                root_feature = feature
        
        self.tree = NAryTree(root_feature, max_gg, None, "root", self.feature_data, self.target)
        feature_groups = set(self.feature_data.get(root_feature)) # type: ignore
        for group in feature_groups:
            child_feature_data, child_target = split_data(self.target, self.feature_data, root_feature, group)
            node = self.recursive_build(group, child_feature_data, child_target, group)
            if node is not None:
                self.tree.insert_child(node)

    def recursive_build(self, criterion, feature_data, target, group):
        # Check if the node should be a leaf-node
        if len(set(target)) == 1:
            root = NAryTree(None, None, criterion, 'leaf', None, None)
            root.root.decision = target[0]
            return root
        
        # Check if there are features remaining
        if len(feature_data) == 0:
            return None
        
        root_entropy = calculate_gini(target)
        root_feature = None
        max_ig = 0
        for feature, data in feature_data.items():
            ig = calculate_gini_gain(target, data, root_entropy)
            if ig > max_ig:
                max_ig = ig
                root_feature = feature
        
        root = NAryTree(root_feature, max_ig, group, "internal", feature_data, target)
        feature_groups = set(feature_data.get(root_feature))
        for group in feature_groups:
            child_feature_data, child_target = split_data(target, feature_data, root_feature, group)
            node = self.recursive_build(group, child_feature_data, child_target, group)
            if node is not None:
                self.tree.insert_child(node) # type: ignore

        return root
