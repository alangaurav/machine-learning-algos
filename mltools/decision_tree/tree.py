class Node:
    # TODO: Consider adding class methods for constructor overloading (instead of variable parameters with default args)
    def __init__(self, feature=None, impurity=None, criterion=None, type=None, feature_data=None, target=None):
        self.feature = feature
        self.impurity = impurity
        self.criterion = criterion
        self.type = type
        self.feature_data = feature_data
        self.target = target
        self.decision = None

class Tree():
    '''
        This class includes all the traversal methods for the trees.
    '''
    def __init__(self):
        pass

   
                

class BinaryTree(Tree):
    def __init__(self, feature, impurity, criterion, type, feature_data, target):
        super().__init__()
        self.root = Node(feature, impurity, criterion, type, feature_data, target)
        self.left = None
        self.right = None

    def insert_left(self, feature, impurity, criterion, type, feature_data, target):
        self.left = Node(feature, impurity, criterion, type, feature_data, target)

    def insert_right(self, feature, impurity, criterion, type, feature_data, target):      
        self.right = Node(feature, impurity, criterion, type, feature_data, target)

class NAryTree(Tree):
    def __init__(self, feature, impurity, criterion, type, feature_data, target):
        super().__init__()
        self.root = Node(feature, impurity, criterion, type, feature_data, target)
        self.children = list()

    def bfs(self, root):
        if root is not None:
            print(f"Node type:{root.root.type} criterion: {root.root.criterion} feature: {root.root.feature} decision: {root.root.decision}")
            for child in root.children:
                self.bfs(child)

    def insert_child(self, node):      
        self.children.append(node) 
