#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:04:34 2018

@author: songqsh
"""

"""
README
======
This file contains Python codes.
======
"""

""" Store common attributes of SDE """

import numpy as np


class SDE:

    def __init__(self, X0, DriftFun, VolFun):
        self.X0 = X0 #
        self.DriftFun = DriftFun #drift: function type of (x, t)
        self.VolFun = VolFun #volatility: Function type  of (x, t)
        
    def PrtCoef(self, x, t):
        print('At (' + str(x) + ',' + str(t) + ') ' + \
              'Drift is ' + str(self.DriftFun(x,t)) + \
              ' Volatility is ' + str(self.VolFun(x,t)))
        
    def PrtInit(self):
        print('Initial state is: ' + str(X0))
        
    def _EulerPath_(self, T, n): #para:  end time, and the mesh number 
        Mu = self.DriftFun
        Sigma = self.VolFun
        t = np.linspace(0, T, n+1) #init mesh     
        Xh = self.X0 + np.zeros(n+1) #init Xh
        for i in range(n): #run EM
            Xh[i+1] = Xh[i] + Mu(Xh[i], t[i]) * (t[i+1] - t[i]) + Sigma(Xh[i], t[i]) * np.sqrt(t[i+1] - t[i])*np.random.normal()
            
        return t, Xh
     
if __name__ == "__main__":
    
    #setup a standard BM
    X0 = 0
    b = lambda x, t: 0 #drift zero
    sigma = lambda x, t: 1 #vol is 1
    iSDE = SDE(X0, b, sigma)
    
    iSDE.PrtInit()
    iSDE.PrtCoef(1, 0)
