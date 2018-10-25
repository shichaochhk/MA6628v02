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
        d1 = (np.log(x0 / K) + (b + 0.5 * sigma ** 2)\
              ) * T / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_value = x0 * ss.norm.cdf(d1) - np.exp(-b * T\
                                     ) * K * ss.norm.cdf(d2)
        return call_value
    
    def _Put_(self, K, T):
        x0 = self.InitState
        b = self.Drift
        put_value = self._Call_(K, T) - x0 + np.exp(-b * T) * K
        return put_value
        
if __name__ == "__main__":
    iGBM = GBM(.05, .2, 10)
    iGBM.PrtInitState()
    iGBM.PrtCoef(20., 5.)
    
    #plot a figure of ten paths
    for i in range(10): 
        [t, Y, W] = iGBM.Euler(2., 4000); 
        plt.plot(t,Y);
        
      
    callvalue = iGBM._Call_(9, 1.)
    print('call value is ' + str(callvalue))
    putvalue = iGBM._Put_(9, 1)
    print('put value is ' + str(putvalue))
      
        
 