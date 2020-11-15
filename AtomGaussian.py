# -*- coding: utf-8 -*-

import numpy as np
#from openbabel import openbabel
from rdkit import Chem

class AtomGaussian(): # give a system which incoulde the initial condition
    
    def __init__(self, centre= np.array([0.0, 0.0, 0.0]), alpha=0.0, volume=0, Gaussian_weight=0,number=0):
        self.centre= centre
        self.alpha = alpha  #Gaussion paprameter
        self.volume = volume
        self.weight = Gaussian_weight  
        self.n = number # number of parents gaussians for this guassian function, 
                        # used for recording overlap gaussian information

def atomIntersection(a = AtomGaussian,b = AtomGaussian):
    
    c = AtomGaussian()
    c.alpha = a.alpha + b.alpha

    #centre 
    c.centre = (a.alpha * a.centre + b.alpha * b.centre)/c.alpha; 
    
    #intersection_volume
    d = 0 #The distance squared between two gaussians
    for i in a.centre - b.centre:
        d+= i**2
   
    c.weight = a.weight * b.weight * np.exp(- a.alpha * b.alpha/c.alpha * d)  
    scale = np.pi/(c.alpha)
    c.volume = c.weight * scale ** (3/2)
    
    # Set the numer of atom of the overlap gaussian
    c.n = a.n + b.n
    
    return  c

               
            
            
    
    
    
    

            
        
    
        
            
            
    
