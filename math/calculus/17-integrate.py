#!/usr/bin/env python3
def poly_integral(poly, C=0):
    if type(poly) is not list or type(C) not in [int, float]:
        return None
    if len(poly) == 0:
        return None

    for coef in poly:
        if type(coef) not in [int, float]:
            return None

    def integrate_helper(p, power=1):
        if len(p) == 0:
            return []
        coef = p[0] / power
        if coef.is_integer():
            coef = int(coef)
        return [coef] + integrate_helper(p[1:], power + 1)

    integral = [C] + integrate_helper(poly)
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()
    return integral
