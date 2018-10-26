#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 09:23:30 2018

@author: songqsh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 22:28:36 2018

@author: songqsh
"""
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
from SDE_V01 import SDE


class GBM(SDE):
    def __init__(self, Drift, Vol, InitState):
        self.Drift = Drift #scalar
        self.Vol = Vol #scalar
        self.InitState = InitState
        self.Mu = lambda x, t: Drift * x
        self.Sigma = lambda x, t: Vol * x
        
    def _explicit_sol_(self, t, W_t):
        x0 = self.InitState
        b = self.Drift
        sigma = self.Vol
        return x0 * np.exp((b - sigma**2/2.) * t + sigma * W_t)
    
    def _Call_(self, K, T):
        x0 = self.InitState
        b = self.Drift
        sigma = self.Vol
        d1 = (np.log(x0 / K) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_value = x0 * ss.norm.cdf(d1) - np.exp(-b * T) * K * ss.norm.cdf(d2)
        return call_value
    
    def _Put_(self, K, T):
        x0 = self.InitState
        b = self.Drift
        put_value = self._Call_(K, T) - x0 + np.exp(-b * T) * K
        return put_value
        
if __name__ == "__main__":
    
    S0 = 100.0
    K = 110.0
    r=0.0475
    sigma = 0.20
    t = 0.
    T = 1.
    
    iGBM = GBM(r, sigma, S0)
    iGBM.PrtInitState()
    iGBM.PrtCoef(S0, T)
    
    #plot a figure of ten paths
    for i in range(10): 
        [t, Y, W] = iGBM.Euler(T, 40); 
        plt.plot(t,Y);
        
      
    callvalue = iGBM._Call_(K, T)
    print('call value is ' + str(callvalue))
    putvalue = iGBM._Put_(K, T)
    print('put value is ' + str(putvalue))
      
        
 