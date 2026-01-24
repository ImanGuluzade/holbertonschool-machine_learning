#!/usr/bin/env python3
def summation_i_squared(n):
    if type(n) is not int or n <= 0:
        return None
    if n == 1:
        return 1
    return n * n + summation_i_squared(n - 1)
