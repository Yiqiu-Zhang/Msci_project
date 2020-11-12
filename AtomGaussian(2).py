# -*- coding: utf-8 -*-

import numpy as np
#from openbabel import openbabel
from rdkit import Chem



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
N=200
gv = GaussianVolume(volume=0.0, overlap=0.0, centroid=np.array([0.0, 0.0, 0.0]), 
                 rotation=np.array([0.0, 0.0, 0.0]), N=N)

#gv.levels.append(N); 

atomIndex = 0
vecIndex = N
alphavalue = 1
atom_position = np.random.random(size=(N, 3))
radius_VDW = 1.1
alphavalue = 2.344/(radius_VDW**2)
guassian_weight = 2.70


#def Molecue_volume:
current_index=[]
pair_index =[] #keep all the first level overlap information inside these list
current_gaussian = []
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
            current_index.append([atomIndex,i])
            current_gaussian.append(ga)
    
        i+=1
    temp_index.reverse()
    pair_index.append(temp_index)
    atomIndex += 1

current_index.reverse()
current_gaussian.reverse()
    #return current_gaussian, current_index
#%%
level = 2
final_level = 4 
next_index = [] # a list that stores the overlaps index of next level
next_gaussian=[]
while level < final_level:
    
    i=0
    j=1
    L=level-1 # since python index start from zero
        
    while i < len(current_index) - 1:
        
        while current_index[i][:L] == current_index[j][:L]:
            
            small_index = current_index[j][L]
            large_index = current_index[i][L]
            
            if small_index in pair_index[large_index]:
                               
                old_overlap= current_gaussian[i]
                added_atom= AtomGaussian(centre= gv.centre[small_index], alpha=gv.alpha[small_index],
                             atom_volume=gv.atom_volume[small_index], Gaussian_weight=gv.w[small_index],
                             number=gv.n[small_index])
        
                ga = atomIntersection(a = old_overlap, b=added_atom).overlap_gaussian()
                EPS = 0.003
                '''
                if ga.atom_volume / (old_overlap.atom_volume + added_atom.atom_volume - ga.atom_volume) > EPS:
        
                    gv.volume = gv.volume - (-1)**L*ga.atom_volume
                    gv.centroid = gv.centroid - (-1)**L*ga.atom_volume*ga.centre
                    '''
            
                next_index.append(current_index[i][:L]+[large_index,small_index])
                    
                next_gaussian.append(ga)
                    
            if j < len(current_index)-1: 
                j+=1
            else:
                i+=1
                break
            
        else:
            i+=1
            j=i+1
    
    
    if len(next_index) == 0:
        print('no overlap at level_'+ str(L+2))
        
    else:
        current_index = next_index
        current_gaussian = next_gaussian
        print('number of overlaps at level_'+str(L+2) +'= ' + str(len(current_index)))
        
    next_index = [] 
    next_gaussian=[]
    
    level +=1
            
            
    
