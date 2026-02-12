#!/usr/bin/env python3

import numpy as np
posterior = __import__('3-posterior').posterior

if __name__ == '__main__':
    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11
    print(posterior(26, 130, P, Pr))
