#!/usr/bin/env python3
def poly_derivative(poly):
    if type(poly) is not list or len(poly) == 0:
        return None

    for coef in poly:
        if type(coef) not in [int, float]:
            return None

    if len(poly) == 1:
        return [0]

    def deriv_helper(p, power=1):
        if len(p) == 1:
            return []
        return [p[1] * power] + deriv_helper(p[1:], power + 1)

    derivative = deriv_helper(poly)
    if len(derivative) == 0:
        return [0]
    return derivative
