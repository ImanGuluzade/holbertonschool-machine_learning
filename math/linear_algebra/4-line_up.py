#!/usr/bin/env python3
"""
This module contains a function to add two arrays element-wise.

The add_arrays function returns a new list with element-wise sums.
If the arrays are not the same shape, it returns None.
"""


def add_arrays(arr1, arr2):
    """Adds two arrays element-wise"""
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
