#!/usr/bin/env python3
"""
Module to build a Decision Tree with efficient prediction
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
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """Recursively counts nodes below this node"""
        left_count = self.left_child.count_nodes_below(only_leaves)
        right_count = self.right_child.count_nodes_below(only_leaves)
        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count

    def get_leaves_below(self):
        """Returns the list of all leaves below this node"""
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

    def max_depth_below(self):
        """Returns the depth of the leaf"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Returns 1 for a leaf"""
        return 1

    def get_leaves_below(self):
        """Returns the leaf itself in a list"""
        return [self]

    def update_bounds_below(self):
        """Leaf base case: do nothing"""
        pass

    def update_indicator(self):
        """Leaf-specific indicator update"""
        super().update_indicator()

    def pred(self, x):
        """Base case prediction: returns the leaf value"""
        return self.value

    def __str__(self):
        """Returns string representation of the leaf"""
        return f"leaf [value={self.value}]"


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

    def depth(self):
        """Returns the maximum depth of the tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Returns number of nodes/leaves in the tree"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """Returns list of all leaves in the tree"""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Updates the bounds for all nodes in the tree"""
        self.root.update_bounds_below()

    def update_predict(self):
        """Computes the efficient prediction function"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_func(A):
            """The actual prediction calculation for matrix A"""
            # Collect indicators for all leaves
            results = np.array([leaf.indicator(A) * leaf.value
                                for leaf in leaves])
            # Sum across axis 0 to consolidate into a single prediction array
            return np.sum(results, axis=0)

        self.predict = predict_func

    def pred(self, x):
        """Public interface for recursive prediction"""
        return self.root.pred(x)

    def __str__(self):
        """Returns string representation of the decision tree"""
        return self.root.__str__() + "\n"
