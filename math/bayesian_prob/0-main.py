#!/usr/bin/env python3

import numpy as np
likelihood = __import__('0-likelihood').likelihood

if __name__ == '__main__':
    # Generates [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    P = np.linspace(0, 1, 11) 
    
    # x=26 (successes), n=130 (total trials)
    try:
        print(likelihood(26, 130, P))
    except Exception as e:
        print(e)
