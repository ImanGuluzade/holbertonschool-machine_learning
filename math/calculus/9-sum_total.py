#!/usr/bin/env python3
def summation_i_squared(n):
    """
    Calculate the sum of squares from 1 to n:
    1^2 + 2^2 + ... + n^2

    Args:
        n (int): stopping number

    Returns:
        int: sum of squares, or None if n is invalid
    """

    # Validate input
    if type(n) is not int or n <= 0:
        return None

    # Base case for recursion
    if n == 1:
        return 1

    # Recursive case
    return n * n + summation_i_squared(n - 1)
