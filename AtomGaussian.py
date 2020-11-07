# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 15:18:36 2020

@author: THINKPAD
"""

import numpy as np
from openbabel import openbabel



class AtomGaussian(): # give a system which incoulde the initial condition
    
    def __init__(self, centre= np.array([0.0, 0.0, 0.0]), alpha=0.0, atom_volume=0, Gaussian_weight=0,number=0):
        self.centre= centre
        self.alpha = alpha  #Gaussion paprameter
        self.atom_volume = atom_volume
        self.w = Gaussian_weight  
        self.n = number
    
    
class atomIntersection():
    
    def __init__(self,a = AtomGaussian(),b = AtomGaussian()):
        self.a = a
        self.b = b
        self.c = AtomGaussian()
        
    def overlap_gaussian(self):
        # find c.alpha
        self.c.alpha = self.a.alpha + self.b.alpha
    
        #centre 
        self.c.centre = (self.a.alpha * self.a.centre + self.b.alpha * self.b.centre)/self.c.alpha; 
        
        #intersection_volume
        d=(self.a.centre[0] - self.b.centre[0])**2
        + (self.a.centre[1] - self.b.centre[1])**2
        + (self.a.centre[2] - self.b.centre[2])**2
        
        self.c.w = self.a.w * self.b.w * np.exp(- self.a.alpha * self.b.alpha/self.c.alpha * d)
        
        scale = np.pi/(self.c.alpha)
        
        self.c.atom_volume = self.c.w * scale ** (3/2)
        
        return  self.c
    
    
#%%

class GaussianVolume(AtomGaussian):
    
    def __init__(self, volume=0.0, overlap=0.0, centroid=np.array([0.0, 0.0, 0.0]), 
                 rotation=np.array([0.0, 0.0, 0.0]), N=0):
        
        self.volume = volume
        self.overlap = overlap
        self.centroid = centroid
        self.rotation = rotation
        self.gaussians = np.empty((N, 7), dtype=object)
        self.childOverlaps = np.empty(N, dtype=object)
        self.levels = np.empty(N, dtype=object)
        
        # from AtomGaussian
        self.alpha = np.empty((N), dtype=object)
        self.centre = np.empty((N,3), dtype=object)
        self.atom_volume = np.empty((N), dtype=object)
        self.w = np.empty((N), dtype=object)
        self.n = np.empty((N), dtype=object)
        
        
        
        
    def GAlpha(self,atomicnumber): #returns the Alpha value of the atom
        
            switcher={
       1:      
          1.679158285,
         
       3:      
          0.729980658,
         
       5:      
          0.604496983,
         
       6:      
          0.836674025,
         
       7:      
          1.006446589,
         
       8:      
          1.046566798,
         
       9:      
          1.118972618,
         
       11:     
          0.469247983,
         
       12:     
          0.807908026,
         
       14:     
          0.548296583,
         
       15:
          0.746292571,
        
       16:     
          0.746292571,
         
       17:     
          0.789547080,
         
       19:     
          0.319733941,
         
       20:   	 
          0.604496983,
         
       26:   	 
          1.998337133,
         
       29:   	 
          1.233667312,
         
       30:   	 
          1.251481772,
         
       35:   	 
          0.706497569,
         
       53:   	 
           0.616770720,
             }
            return switcher.get(atomicnumber,1.074661303) 
#%%
N=1000
gv = GaussianVolume(volume=0.0, overlap=0.0, centroid=np.array([0.0, 0.0, 0.0]), 
                 rotation=np.array([0.0, 0.0, 0.0]), N=N)

#gv.levels.append(N); 

atomIndex = 0
vecIndex = N
atom_position = np.random.random(size=(N, 3))*2
alphavalue = 1
guassian_weight = 2.70
radius_VDW = 0.8


overlaps_index =[] #keep all the first level overlap information inside these list
overlaps_volume =[]
overlaps_centre =[]
while atomIndex < N:
    gv.centre[atomIndex] = atom_position[atomIndex,:]
    gv.alpha[atomIndex] = alphavalue
    gv.w[atomIndex] = guassian_weight 
    gv.atom_volume[atomIndex] = (4.0 * np.pi/3.0) * radius_VDW **3 # volume of the atom
    gv.n[atomIndex] = 1 # number

    

    '''Update volume and centroid of the molecue'''
    gv.volume = gv.volume + gv.atom_volume[atomIndex]
    gv.centroid = gv.centroid + gv.atom_volume[atomIndex]*gv.centre[atomIndex]
    
    
    i=0
    temp_index =[]
    temp_volume=[]
    temp_centre=[]
    while i < atomIndex:
        
        atom1 = AtomGaussian(centre= gv.centre[i], alpha=gv.alpha[i], atom_volume=gv.atom_volume[i], 
                             Gaussian_weight=gv.w[i], number=gv.n[i])
        
        atom2= AtomGaussian(centre= gv.centre[atomIndex], alpha=gv.alpha[atomIndex],
                             atom_volume=gv.atom_volume[atomIndex], Gaussian_weight=gv.w[atomIndex],
                             number=gv.n[atomIndex])
        
        ga = atomIntersection(a = atom1, b=atom2).overlap_gaussian()
          
 
        EPS = 0.003
        if ga.atom_volume / (gv.atom_volume[i] + gv.atom_volume[atomIndex] - ga.atom_volume) > EPS:
        
            gv.volume = gv.volume - ga.atom_volume
            gv.centroid = gv.centroid - ga.atom_volume*ga.centre
            
            temp_index.append(i)
            temp_volume.append(ga.atom_volume)
            temp_centre.append(ga.centre)
        
        i+=1
    temp_index.reverse()
    temp_volume.reverse()
    temp_centre.reverse()
    overlaps_index.append(temp_index)
    overlaps_volume.append(temp_volume)
    overlaps_centre.append(temp_centre)
    atomIndex += 1


current_overlaps=[]    # a list that stores the overlaps index of current level     
#overlaps_index=[[],[0],[1,0],[2,1],[3,2,0],[4,3,2,0]]
for parents_index in range(N-1,-1,-1):
    for large_index in overlaps_index[parents_index]:
        for small_index in overlaps_index[parents_index]:
            if small_index != large_index:
                if small_index in overlaps_index[large_index]:
                    current_overlaps.append([parents_index,large_index,small_index])

level = 2
final_level = 3 #because we have run pairs before , therefore it is final level - 1
while level < final_level:
    
    i=0
    j=1
    L=level
    next_overlaps = [] # a list that stores the overlaps index of next level
    if len(current_overlaps) == 0:
        print('no higher level overlap')
        
    while i < len(current_overlaps) - 1:
       
        while current_overlaps[i][:L] == current_overlaps[j][:L]:
            small_index = current_overlaps[j][L]
            large_index = current_overlaps[i][L]
            if small_index in overlaps_index[large_index]:
                
                next_overlaps.append(current_overlaps[i][:L]+[large_index,small_index])
            

            j+=1
        else:
            
            i+=1
            j=i+1
    next_overlaps
    
    
            
    current_overlaps = next_overlaps
    
    
    level +=1
            
            
            
#%%
       
current_overlaps=[]    # a list that stores the overlaps index of current level     
overlaps_index=[[],[0],[1,0],[2,1],[3,2,0],[4,3,2,0]]
for parents_index in range(6-1,-1,-1):
    for large_index in overlaps_index[parents_index]:
        for small_index in overlaps_index[parents_index]:
            if small_index != large_index:
                if small_index in overlaps_index[large_index]:
                    current_overlaps.append([parents_index,large_index,small_index])
#%%
level = 2
final_level = 3 #because we have run pairs before , therefore it is final level - 1
while level < final_level:
    
    i=0
    j=1
    L=level
    next_overlaps = [] # a list that stores the overlaps index of next level
    if np.size(current_overlaps) == 0:
        print('no higher level overlap')
        
    while i < len(current_overlaps) - 1:
       
        while current_overlaps[i][:L] == current_overlaps[j][:L]:
            small_index = current_overlaps[j][L]
            large_index = current_overlaps[i][L]
            if small_index in overlaps_index[large_index]:
                
                next_overlaps.append(current_overlaps[i][:L]+[large_index,small_index])
            

            j+=1
        else:
            
            i+=1
            j=i+1
    next_overlaps
    
    
            
    current_overlaps = next_overlaps
    
    
    level +=1
    
    


                    
            
    
    


    
        
        
        


