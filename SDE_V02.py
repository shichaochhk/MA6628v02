#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 19:41:01 2018

@author: songqsh

One dimensional SDE class


"""

import numpy as np
import matplotlib.pyplot as plt

class SDE:
    
    """
    Initialize
    """
    def __init__(self, Mu, Sigma, InitState):
        self.Mu = Mu
        self.Sigma = Sigma
        self.InitState = InitState
        
        self.MuP = lambda x, t: 0 #first order derivative of Mu
                                #used for Milstein
        
    """
    Euler method
    """
    def Euler(self, T, N):
        x0 = self.InitState
        Mu = self.Mu
        Sigma = self.Sigma       
        t = np.linspace(0, T, N+1)
        DeltaT = T/N
        
        Wh = np.zeros(N+1) #init BM
        Xh = x0 + np.zeros(N+1) #init Xh
        
        for i in range(N): #run EM            
            DeltaW = np.sqrt(t[i+1] - t[i]) * np.random.normal()
            Wh[i+1] = Wh[i] + DeltaW
            Xh[i+1] = Xh[i] + \
                Mu(Xh[i], t[i]) * DeltaT + \
                Sigma(Xh[i], t[i])* DeltaW
            
        return t, Xh, Wh


    """
    Milstein -- Later
    """
    
    def Milstein(self, T, N):
        x0 = self.InitState
        Mu = self.Mu
        MuP = self.MuP
        
        Sigma = self.Sigma
        t = np.linspace(0, T, N+1)
        DeltaT = T/N
        
        Wh = np.zeros(N+1) #init BM
        Xh = x0 + np.zeros(N+1) #init Xh
        
        for i in range(N):
            DeltaW = np.sqrt(t[i+1] - t[i]) * np.random.normal()
            Wh[i+1] = Wh[i] + DeltaW
            Xh[i+1] = Xh[i] + \
                Mu(Xh[i], t[i]) * DeltaT + \
                Sigma(Xh[i], t[i])* DeltaW #Euler
            Xh[i+1] = Xh[i+1] + \
                0.5 * Mu(Xh[i], t[i]) * MuP(Xh[i], t[i]) * (DeltaW**2 - DeltaT)
                
        return t, Xh, Wh
         
    
    

        
if __name__ == '__main__':
    
    #OU process
    b = lambda x, t: - x
    sigma = lambda x, t: 1.
    x0 = 1.
    iSDE = SDE(b, sigma, x0)
    
    plt.figure()
    for i in range(10): 
        [t, Y, W] = iSDE.Euler(20., 100); 
        plt.plot(t,Y);
    
    plt.figure()
    for i in range(10): 
        [t, Y, W] = iSDE.Milstein(20., 100); 
        plt.plot(t,Y);
    