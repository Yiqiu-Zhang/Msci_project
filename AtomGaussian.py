# -*- coding: utf-8 -*-

import numpy as np
#from openbabel import openbabel


class AtomGaussian(): # give a system which incoulde the initial condition
    
    def __init__(self, centre= np.array([0.0, 0.0, 0.0]), alpha=0.0, volume=0, Gaussian_weight=2.7,number=0):
        self.centre= centre
        self.alpha = alpha  #Gaussion paprameter
        self.volume = volume
        self.weight = Gaussian_weight  
        self.n = number # number of parents gaussians for this guassian function, 
                        # used for recording overlap gaussian information

def atomIntersection(a = AtomGaussian(),b = AtomGaussian()):
    
    c = AtomGaussian()
    c.alpha = a.alpha + b.alpha

    #centre 
    c.centre = (a.alpha * a.centre + b.alpha * b.centre)/c.alpha; 
    
    #intersection_volume
    d = a.centre - b.centre
    d_sqr = d.dot(d)  #The distance squared between two gaussians
     
    c.weight = a.weight * b.weight * np.exp(- a.alpha * b.alpha/c.alpha * d_sqr)  
    scale = np.pi/c.alpha
    c.volume = c.weight * scale ** 1.5
    
    # Set the numer of atom of the overlap gaussian
    c.n = a.n + b.n
    
    return  c

               
            
            
    
    
    
    

            
        
    
        
            
            
    
