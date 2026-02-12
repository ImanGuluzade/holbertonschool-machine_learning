#!/usr/bin/env python3

import numpy as np
intersection = __import__('1-intersection').intersection

if __name__ == '__main__':
    # 11 hypothetical probabilities from 0 to 1
    P = np.linspace(0, 1, 11)

    # Prior assumes each of the 11 probabilities is equally likely (1/11)
    Pr = np.ones(11) / 11

    # Patients (n=130), Side Effects (x=26)
    try:
        print(intersection(26, 130, P, Pr))
    except Exception as e:
        print(e)
