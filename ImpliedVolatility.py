#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: songqsh, SID: 123456
"""

from scipy import optimize
from BSM_option_valuation import *

def IVolBsm(S0, K, T, r, P0):
    """
    Inputs:
        S0: spot price
        K: strike
        T: time to maturity in year
        r: rate
        P0: market price
    Outputs:
        Implied volatility
    """
    InitVol = .3
    error = lambda sigma: (BSM_call_value(S0, K, 0, T, r, sigma) - P0)**2
    opt = optimize.fmin(error, InitVol);
    return opt[0]

if __name__ == "__main__":
    print('Implied volatility is', IVolBsm(100, 100, 1, .02, 9))





