#!/usr/bin/env python3
"""
Module to implement an Isolation Random Tree for outlier detection
"""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """
    Class that represents an isolation random tree
    """
    def __init__(self, max_depth=10, seed=0, root=None):
        """
        Initializes the isolation tree
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """String representation of the tree"""
        return self.root.__str__() + "\n"

    def depth(self):
        """Returns the maximum depth of the tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Counts the total number of nodes or leaves"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """Updates the feature bounds for each node"""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Returns a list of all leaves in the tree"""
        return self.root.get_leaves_below()

    def update_predict(self):
        """Updates the vectorized prediction function to return node depth"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_func(A):
            # For isolation trees, the leaf value is its depth
            results = np.array([leaf.indicator(A) * leaf.depth
                                for leaf in leaves])
            return np.sum(results, axis=0)

        self.predict = predict_func

    def np_extrema(self, arr):
        """Returns the min and max values of an array"""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Randomly selects a feature and a threshold for splitting"""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Creates a leaf child with the value set to its depth"""
        # Isolation trees predict the depth of the leaf
        leaf_child = Leaf(value=node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Creates an internal node child"""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Recursively determines if a node is a leaf or should split"""
        node.feature, node.threshold = self.random_split_criterion(node)

        left_population = np.logical_and(node.sub_population,
                                         self.explanatory[:, node.feature] >
                                         node.threshold)
        right_population = np.logical_and(node.sub_population,
                                          self.explanatory[:, node.feature] <=
                                          node.threshold)

        # In isolation trees, leaf criteria depend only on depth and pop size
        is_left_leaf = (np.sum(left_population) <= self.min_pop or
                        node.depth + 1 == self.max_depth)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (np.sum(right_population) <= self.min_pop or
                         node.depth + 1 == self.max_depth)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """Trains the isolation random tree"""
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        # Corrected mask initialization for the shape of explanatory data
        self.root.sub_population = np.ones(explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"  Training finished.")
            print(f"    - Depth                     : {self.depth()}")
            print(f"    - Number of nodes           : {self.count_nodes()}")
            print(f"    - Number of leaves          : "
                  f"{self.count_nodes(only_leaves=True)}")
