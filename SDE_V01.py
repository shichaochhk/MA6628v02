#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 19:41:01 2018

@author: songqsh
"""

import numpy as np
import matplotlib.pyplot as plt

class SDE:
    def __init__(self, Mu, Sigma, InitState):
        self.Mu = Mu
        self.Sigma = Sigma
        self.InitState = InitState
        
    def PrtCoef(self, x, t):
        print('At state x = ' + str(x) + ' time t = ' + str(t) + '\n')
        print('Mu ' + str(self.Mu(x, t)) + '\n')
        print('Sigma = ' + str(self.Sigma(x, t)) + '\n')
        
    def PrtInitState(self):
        print('Initial state is ' + str(self.InitState) + '\n')
        
    def Euler(self, T, N):
        x0 = self.InitState
        Mu = self.Mu
        Sigma = self.Sigma       
        t = np.linspace(0, T, N+1)
        
        Wh = np.zeros(N+1) #init BM
        Xh = x0 + np.zeros(N+1) #init Xh
        
        for i in range(N): #run EM
            DeltaT = t[i+1] - t[i]
            DeltaW = np.sqrt(t[i+1] - t[i]) * np.random.normal()
            Wh[i+1] = Wh[i] + DeltaW
            Xh[i+1] = Xh[i] + Mu(Xh[i], t[i]) * DeltaT + \
            Sigma(Xh[i], t[i])* DeltaW
            
        return t, Xh, Wh
        
if __name__ == '__main__':
    
    #std BM
    b = lambda x, t: 0.
    sigma = lambda x, t: 1.
    x0 = 0.
    iSDE = SDE(b, sigma, x0)
    iSDE.PrtInitState()
    iSDE.PrtCoef(1., 5.)
    
    for i in range(10): 
        [t, Y, W] = iSDE.Euler(2., 4000); 
        plt.plot(t,Y);
    
    