# -*- coding: utf-8 -*-

import numpy as np
from openbabel import openbabel



class AtomGaussian(): # give a system which incoulde the initial condition
    
    def __init__(self):
        self.centre= np.array([0.0, 0.0, 0.0])
        self.alpha = 0.0  #Gaussion paprameter
        self.volume = 0.0
        self.weight = 2.828427125
        self.n = 0 # number of parents gaussians for this guassian function, 
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
    
    c.volume = c.weight * (np.pi/c.alpha) ** 1.5
    
    # Set the numer of atom of the overlap gaussian
    c.n = a.n + b.n
    
    return  c

               
            
            
    
    
    
    

            
        
    
        
            
            
    
