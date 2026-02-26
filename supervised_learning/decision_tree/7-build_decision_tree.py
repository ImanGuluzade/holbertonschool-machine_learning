#!/usr/bin/env python3
"""
Module to build and train a Decision Tree
"""
import numpy as np


class Node:
    """Represents an internal node in a decision tree"""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth
        self.lower = None
        self.upper = None
        self.indicator = None

    def max_depth_below(self):
        """Calculates the maximum depth below this node"""
        if self.is_leaf:
            return self.depth
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """Recursively counts nodes below this node"""
        if self.is_leaf:
            return 1
        left_count = self.left_child.count_nodes_below(only_leaves)
        right_count = self.right_child.count_nodes_below(only_leaves)
        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count

    def get_leaves_below(self):
        """Returns the list of all leaves below this node"""
        if self.is_leaf:
            return [self]
        return self.left_child.get_leaves_below() + \
            self.right_child.get_leaves_below()

    def update_bounds_below(self):
        """Recursively computes the feature bounds for each node"""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        for child in [self.left_child, self.right_child]:
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()

        self.left_child.lower[self.feature] = self.threshold
        self.right_child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Computes the indicator function for the node"""
        def is_large_enough(x):
            """Checks if features are greater than lower bounds"""
            return np.all([np.greater(x[:, key], self.lower[key])
                           for key in self.lower.keys()], axis=0)

        def is_small_enough(x):
            """Checks if features are less than or equal to upper bounds"""
            return np.all([np.less_equal(x[:, key], self.upper[key])
                           for key in self.upper.keys()], axis=0)

        self.indicator = lambda x: np.all(np.array([is_large_enough(x),
                                                    is_small_enough(x)]),
                                          axis=0)

    def pred(self, x):
        """Standard recursive prediction for a single individual"""
        if self.is_leaf:
            return self.value
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)

    def left_child_add_prefix(self, text):
        """Adds prefix to left child string representation"""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Adds prefix to right child string representation"""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """Returns string representation of the node"""
        if self.is_leaf:
            return f"leaf [value={self.value}]"
        if self.is_root:
            out = (f"root [feature={self.feature}, "
                   f"threshold={self.threshold}]\n")
        else:
            out = (f"node [feature={self.feature}, "
                   f"threshold={self.threshold}]\n")
        out += self.left_child_add_prefix(self.left_child.__str__())
        out += self.right_child_add_prefix(self.right_child.__str__())
        return out.rstrip()


class Leaf(Node):
    """Represents a leaf node in a decision tree"""
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def update_bounds_below(self):
        """Leaf base case: do nothing"""
        pass

    def update_indicator(self):
        """Leaf-specific indicator update"""
        super().update_indicator()


class Decision_Tree():
    """Represents a decision tree classifier"""
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def np_extrema(self, arr):
        """Returns min and max of an array"""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Randomly selects a feature and threshold for splitting"""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit(self, explanatory, target, verbose=0):
        """Trains the decision tree"""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            # Gini_split_criterion placeholder
            pass
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
- Depth                     : {self.depth()}
- Number of nodes           : {self.count_nodes()}
- Number of leaves          : {self.count_nodes(only_leaves=True)}
- Accuracy on training data : {self.accuracy(self.explanatory, self.target)}""")

    def fit_node(self, node):
        """Recursively fits nodes of the tree"""
        node.feature, node.threshold = self.split_criterion(node)

        mask_left = self.explanatory[:, node.feature] > node.threshold
        mask_right = self.explanatory[:, node.feature] <= node.threshold

        left_pop = np.logical_and(node.sub_population, mask_left)
        right_pop = np.logical_and(node.sub_population, mask_right)

        def is_leaf(pop, depth):
            if np.sum(pop) < self.min_pop or depth >= self.max_depth:
                return True
            return len(np.unique(self.target[pop])) == 1

        # Process left child
        if is_leaf(left_pop, node.depth + 1):
            node.left_child = self.get_leaf_child(node, left_pop)
        else:
            node.left_child = self.get_node_child(node, left_pop)
            self.fit_node(node.left_child)

        # Process right child
        if is_leaf(right_pop, node.depth + 1):
            node.right_child = self.get_leaf_child(node, right_pop)
        else:
            node.right_child = self.get_node_child(node, right_pop)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Creates a leaf child"""
        pop_targets = self.target[sub_population]
        value = np.bincount(pop_targets).argmax()
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Creates an internal node child"""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Calculates accuracy"""
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size

    def depth(self):
        """Returns max depth"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Returns node count"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_predict(self):
        """Updates the prediction function"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_func(A):
            results = np.array([leaf.indicator(A) * leaf.value
                                for leaf in leaves])
            return np.sum(results, axis=0)

        self.predict = predict_func

    def update_bounds(self):
        """Computes feature bounds"""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Returns list of leaves"""
        return self.root.get_leaves_below()

    def __str__(self):
        """Returns string representation"""
        return self.root.__str__() + "\n"
