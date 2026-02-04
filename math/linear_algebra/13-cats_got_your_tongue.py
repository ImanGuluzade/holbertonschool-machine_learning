#!/usr/bin/env python3
"""
This module containts a function to concatenate two numpy.ndarrays
along a specific axis.

The np_cat function returns a new numpy.ndarray with the concatenated 
result along the given axis.
"""

import numpy as np
ø


def  np_cat(mat1,mat2,axis=0):
     """Concatenates two numpy.ndarrays along a specified axis"""
     return np.concatenate((mat1,mat2),axis=axis)
