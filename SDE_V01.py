#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 19:41:01 2018

@author: songqsh
"""

import numpy as np

class SDE:
    def __init__(self, Drift, Vol, InitState):
        self.Drift = Drift
        self.Vol = Vol
        self.InitState = InitState
        
    def PrtCoef(self, x, t):
        print('At state x = ' + str(x) + ' time t = ' + str(t) + '\n')
        print('Drift is b = ' + str(self.Drift(x, t)) + '\n')
        print('Volatility is Sigma = ' + str(self.Vol(x, t)) + '\n')
        
    def PrtInitState(self):
        print('Initial state is ' + str(self.InitState) + '\n')
        
    def Euler(self, T, N):
        x0 = self.InitState
        Mu = self.Drift
        Sigma = self.Vol       
        t = np.linspace(0, T, N+1)
        
        Xh = x0 + np.zeros(N+1) #init Xh
        
        for i in range(N): #run EM
            Xh[i+1] = Xh[i] + Mu(Xh[i], t[i]) * (t[i+1] - t[i]) + \
            Sigma(Xh[i], t[i])* np.sqrt(t[i+1] - t[i])*np.random.normal()
            
        return t, Xh
        
if __name__ == '__main__':
    b = lambda x, t: 0.
    sigma = lambda x, t: 1.
    x0 = 0.
    iSDE = SDE(b, sigma, x0)
    iSDE.PrtInitState()
    iSDE.PrtCoef(1., 5.)
    
    