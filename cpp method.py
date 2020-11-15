# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 19:43:22 2020

@author: THINKPAD
"""

import numpy as np
#from openbabel import openbabel
from rdkit import Chem
import itertools




class AtomGaussian(): # give a system which incoulde the initial condition
    
    def __init__(self, centre= np.array([0.0, 0.0, 0.0]), alpha=0.0, volume=0, Gaussian_weight=0,number=0):
        self.centre= centre
        self.alpha = alpha  #Gaussion paprameter
        self.volume = volume
        self.w = Gaussian_weight  
        self.n = number # number of parents gaussians for this guassian function, 
                        # used for recording overlap gaussian information

def atomIntersection(a = AtomGaussian(),b = AtomGaussian()):
    
    c = AtomGaussian()
    c.alpha = a.alpha + b.alpha

    #centre 
    c.centre = (a.alpha * a.centre + b.alpha * b.centre)/c.alpha; 
    
    #intersection_volume
    d = 0
    for i in a.centre - b.centre:
        d+= i**2
   
    c.w = a.w * b.w * np.exp(- a.alpha * b.alpha/c.alpha * d)
    
    scale = np.pi/(c.alpha)
    
    c.volume = c.w * scale ** (3/2)
    c.n = a.n + b.n
    
    return  c

#%%

class GaussianVolume(AtomGaussian):
    
    def __init__(self, volume=0.0, overlap=0.0, centroid=np.array([0.0, 0.0, 0.0]), 
                 rotation=np.array([0.0, 0.0, 0.0]), N=0):
        
        self.volume = volume
        self.overlap = overlap
        self.centroid = centroid
        self.rotation = rotation
        self.gaussians = [AtomGaussian() for i in range(0,N)]
        self.childOverlaps = [[] for i in range(N)] # store the overlap tree used for molecue molecue intersection
        self.levels = [N]
        
        self.parents = []
        self.processQueue = []
                    
def GAlpha(atomicnumber): #returns the Alpha value of the atom
    
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
N=10  # !!!N would need to redefine inside the function Molecue_volume
gv = GaussianVolume(volume=0.0, overlap=0.0, centroid=np.array([0.0, 0.0, 0.0]), 
                 rotation=np.array([0.0, 0.0, 0.0]), N=N)

#gv.levels.append(N)

def Molecue_volume(gv):

    parents=[[] for i in range(N)]
    overlaps=[[] for i in range(N)]
    atomIndex = 0
    vecIndex = N
    
     # !!!this part need redefine after import RDkit or openbabel
    atom_position = np.random.random(size=(N, 3))*5
    radius_VDW = 0.8
    alphavalue = 2.344/(radius_VDW**2)
    guassian_weight = 2.70
    
    while atomIndex < N:
        gv.gaussians[atomIndex].centre = atom_position[atomIndex,:]
        gv.gaussians[atomIndex].alpha = alphavalue
        gv.gaussians[atomIndex].w = guassian_weight 
        gv.gaussians[atomIndex].volume = (4.0 * np.pi/3.0) * radius_VDW **3 # volume of the atom
        gv.gaussians[atomIndex].n = 1 
    
     
        '''Update volume and centroid of the molecue'''
        gv.volume += gv.gaussians[atomIndex].volume
        gv.centroid += gv.gaussians[atomIndex].volume*gv.gaussians[atomIndex].centre
        
        i=0
        while i < atomIndex:
        
            atom1 = gv.gaussians[i]
            atom2= gv.gaussians[atomIndex]
            
            ga = atomIntersection(a = atom1, b=atom2)
          
     
            EPS = 0.003
            if ga.volume / (gv.gaussians[i].volume + gv.gaussians[atomIndex].volume - ga.volume) > EPS:
        
                gv.volume = gv.volume - ga.volume
                gv.centroid = gv.centroid - ga.volume*ga.centre
                
                gv.gaussians.append(ga) 
                gv.childOverlaps.append([]) #append a empty list in the end to store child of child
                overlaps.append([])
                parents.append([i,atomIndex])
                
                gv.childOverlaps[i].append(vecIndex) # store the position of the child (vecIndex) in the root (i)   
                overlaps[i].append(atomIndex)
                vecIndex+=1
            
            i+=1
            
        atomIndex += 1
        
    startLevel = N
    nextLevel = len(gv.gaussians)
    gv.levels.append(nextLevel)
        
    level = 2
    final_level = 7 

    while level < final_level:
        
        i = startLevel
        while i < nextLevel:
            a1 = parents[i][0]
            a2 = parents[i][1]
            overlaps[i] = list(set(overlaps[a1]) & set(overlaps[a2]))
            
            
            if len(overlaps[i]) != 0:
                for elements in overlaps[i]:
                    if elements >= a2:
                        old_overlap = gv.gaussians[i]
                        added_overlap = gv.gaussians[elements]
                        ga = atomIntersection(old_overlap,added_overlap)
                        
                        EPS = 0.003
                        if ga.volume / (old_overlap.volume + added_overlap.volume - ga.volume) > EPS:
                        
                            if ga.n%2 ==0:# even number overlaps give positive contribution
                                gv.volume = gv.volume - ga.volume
                                gv.centroid = gv.centroid - ga.volume*ga.centre
                            else:
                                gv.volume = gv.volume + ga.volume
                                gv.centroid = gv.centroid + ga.volume*ga.centre
                            
                            gv.gaussians.append(ga)
                            gv.childOverlaps.append([]) #append a empty list in the end to store child of child
                            parents.append([i,elements])
                            overlaps.append([])
                            gv.childOverlaps[i].append(vecIndex) # store the position of the child (vecIndex) in the root (i)
                            vecIndex+=1
                        
            i+=1
            
        startLevel = nextLevel
        nextLevel = len(gv.gaussians)
        gv.levels.append(nextLevel)
        
        level +=1
    
    gv.centroid/=gv.volume #normalise the centroid
    gv.overlap, gv.processQueue = Molecue_overlap(gv,gv)
    gv.parents = parents
    

    
    return gv

#%%
def initOrientation(gv):
    mass_matrix = np.zeros(shape=(3,3))
    
    for i in gv.gaussians:
        i.centre -=gv.centroid
        if i.n % 2 == 0: # for even number of atom, negative contribution
        
            mass_matrix[0][0] -= i.volume * i.centre[0] * i.centre[0]
            mass_matrix[0][1] -= i.volume * i.centre[0] * i.centre[1]
            mass_matrix[0][2] -= i.volume * i.centre[0] * i.centre[2]
            mass_matrix[1][1] -= i.volume * i.centre[1] * i.centre[1]
            mass_matrix[1][2] -= i.volume * i.centre[1] * i.centre[2]
            mass_matrix[2][2] -= i.volume * i.centre[2] * i.centre[2]
        else:
            mass_matrix[0][0] += i.volume * i.centre[0] * i.centre[0]
            mass_matrix[0][1] += i.volume * i.centre[0] * i.centre[1]
            mass_matrix[0][2] += i.volume * i.centre[0] * i.centre[2]
            mass_matrix[1][1] += i.volume * i.centre[1] * i.centre[1]
            mass_matrix[1][2] += i.volume * i.centre[1] * i.centre[2]
            mass_matrix[2][2] += i.volume * i.centre[2] * i.centre[2]
        
    # set lower triangle       	
    mass_matrix[1][0] = mass_matrix[0][1]
    mass_matrix[2][0] = mass_matrix[0][2]
    mass_matrix[2][1] = mass_matrix[1][2]
    
    #normalise
    mass_matrix /= gv.volume
      
    gv.rotation, s, vh = np.linalg.svd(mass_matrix)
        
    if np.linalg.det(gv.rotation) < 0:
        gv.rotation[:][2] = - gv.rotation[:][2]
        
    for i in gv.gaussians:
        i.centre = gv.rotation.dot(i.centre)
              
    return gv


#%%

def Molecue_overlap(gRef = GaussianVolume, gDb = GaussianVolume):
    processQueue=[]
    overlap_volume = 0
    N1 = gRef.levels[0]
    N2 = gDb.levels[0]
    i=0
    EPS = 0.03
    while i < N1:
        j=0
        while j < N2:
           g_ij = atomIntersection(a = gRef.gaussians[i], b=gDb.gaussians[j])
           V_ij = g_ij.volume
           
           if V_ij / (gRef.gaussians[i].volume + gDb.gaussians[j].volume - V_ij) > EPS:
               
               overlap_volume += V_ij
               
               d1 = gRef.childOverlaps[i]
               d2 = gDb.childOverlaps[j]
               
               if d2:
                   for it1 in d2:
                       processQueue.append([i,it1])
               if d1:
                   for it1 in d1:
                       processQueue.append([it1,j])
                      
           j+=1

        i+=1
    
    for pair in processQueue:
               
        i = pair[0]
        j = pair[1]
        g_ij = atomIntersection(a = gRef.gaussians[i], b=gDb.gaussians[j])
        V_ij = g_ij.volume
        
        if V_ij / (gRef.gaussians[i].volume + gDb.gaussians[j].volume - V_ij) > EPS:
                           
            if (g_ij.n)%2 ==0:
                
                overlap_volume += V_ij
            else:
                overlap_volume -= V_ij
                
                
            d1 = gRef.childOverlaps[i]
            d2 = gDb.childOverlaps[j]
        
            if d1 and gRef.gaussians[i].n >  gDb.gaussians[j].n:
                for it1 in d1:
                    if [it1,j] not in processQueue:
                        processQueue.append([it1,j])
            else:
                if d2:
                    for it1 in d2:
                        if [i,it1] not in processQueue:
                            processQueue.append([i,it1])
                if d1 and gDb.gaussians[j].n - gRef.gaussians[i].n < 2:
                    for it1 in d1:
                        if [it1,j] not in processQueue:
                            processQueue.append([it1,j])

    return overlap_volume, processQueue
                    

#%%
gv =  Molecue_volume(gv)



                
        
            
        
        
        
                       
                   
                    
                   
               