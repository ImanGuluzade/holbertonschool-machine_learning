#!/usr/bin/env python3

import numpy as np
marginal = __import__('2-marginal').marginal

if __name__ == '__main__':
    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11
    print(marginal(26, 130, P, Pr))
