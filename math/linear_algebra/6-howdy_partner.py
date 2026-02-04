#!/usr/bin/env python3
"""
This module contains a function to concatenate two arrays.

The cat_arrays function returns a new list containing all elements
from arr1 followed by all elements from arr2.
"""


def cat_arrays(arr1, arr2):
    """Concatenates two lists and returns a new list"""
    new_list = []
    for el in arr1:
        new_list.append(el)
    for el in arr2:
        new_list.append(el)
    return new_list
