# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 19:43:22 2020

@author: THINKPAD
"""

import numpy as np
#from openbabel import openbabel
from rdkit import Chem
from AlignmentInfo import AlignmentInfo
from AtomGaussian import AtomGaussian, atomIntersection



class GaussianVolume(AtomGaussian):
    
    def __init__(self, volume=0.0, overlap=0.0, centroid=np.array([0.0, 0.0, 0.0]), 
                 rotation=np.array([0.0, 0.0, 0.0]), N=0):
        
        self.volume = volume
        self.overlap = overlap
        self.centroid = centroid
        self.rotation = rotation
        
        # Store the original atom gaussians and the overlap gaussians calculated later
        self.gaussians = [AtomGaussian() for i in range(0,N)]
        
        # store the overlap tree used for molecule molecule intersection
        self.childOverlaps = [[] for i in range(N)] 
        
        # Store the number of gaussians for each level
        self.levels = [N]
        
        #self.parents = []
        #self.processQueue = []
                    
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
N=20  # !!!N would need to redefine inside the function Molecule_volume
gv = GaussianVolume(volume=0.0, overlap=0.0, centroid=np.array([0.0, 0.0, 0.0]), 
                 rotation=np.array([0.0, 0.0, 0.0]), N=N)

def Molecule_volume(gv = GaussianVolume):
    
    # Stores the parents of gv.gaussians[i] inside parents[i]
    parents=[[] for i in range(N)]  
    
    # Stores the atom index that have intersection with i_th gaussians inside overlaps[i]
    overlaps=[[] for i in range(N)] 
    
    atomIndex = 0
    vecIndex = N #Used to indicated the initial position of the child gaussian
    
     # !!!this part need redefine after import RDkit or openbabel
    atom_position = np.random.random(size=(N, 3))*3
    radius_VDW = 0.8
    alphavalue = 2.344/(radius_VDW**2)
    guassian_weight = 2.70
    
    while atomIndex < N:
        gv.gaussians[atomIndex].centre = atom_position[atomIndex,:]
        gv.gaussians[atomIndex].alpha = alphavalue
        gv.gaussians[atomIndex].weight = guassian_weight 
        gv.gaussians[atomIndex].volume = (4.0 * np.pi/3.0) * radius_VDW **3 # volume of the atom
        gv.gaussians[atomIndex].n = 1 
    
     
        '''Update volume and centroid of the Molecule'''
        gv.volume += gv.gaussians[atomIndex].volume
        gv.centroid += gv.gaussians[atomIndex].volume*gv.gaussians[atomIndex].centre
        
        '''loop over every atom to find the second level overlap'''
        i=0
        while i < atomIndex:
        
            atom1 = gv.gaussians[i]
            atom2= gv.gaussians[atomIndex]
            
            ga = atomIntersection(a = atom1, b=atom2)
          
     
            EPS = 0.003
            # Check if overlap is sufficient enough
            if ga.volume / (gv.gaussians[i].volume + gv.gaussians[atomIndex].volume - ga.volume) > EPS:
        
                gv.volume = gv.volume - ga.volume
                gv.centroid = gv.centroid - ga.volume*ga.centre
                
                gv.gaussians.append(ga) 
                
                #append a empty list in the end to store child of this overlap gaussian
                overlaps.append([])
                
                gv.childOverlaps.append([]) 
                parents.append([i,atomIndex])
                
                # store the position of the child (vecIndex) in the root (i)   
                gv.childOverlaps[i].append(vecIndex) 
                
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
            
            # parents[i] is a pair list e.g.[a1,a2]
            a1 = parents[i][0]
            a2 = parents[i][1]
            
            # find elements that overlaps with both gaussians(a1 and a2)
            overlaps[i] = list(set(overlaps[a1]) & set(overlaps[a2]))
            
            
            if len(overlaps[i]) != 0:
                for elements in overlaps[i]:
                    
                    # check if there is a wrong index
                    if elements >= a2:
                        old_overlap = gv.gaussians[i]
                        added_overlap = gv.gaussians[elements]
                        ga = atomIntersection(old_overlap,added_overlap)
                        
                        EPS = 0.003
                        if ga.volume / (old_overlap.volume + added_overlap.volume - ga.volume) > EPS:
                        
                            if ga.n%2 ==0:# even number overlaps give positive contribution
                                gv.volume = gv.volume - ga.volume
                                gv.centroid = gv.centroid - ga.volume*ga.centre
                            else:         # odd number overlaps give negative contribution
                                gv.volume = gv.volume + ga.volume
                                gv.centroid = gv.centroid + ga.volume*ga.centre
                            
                            gv.gaussians.append(ga)
                            
                            #append a empty list in the end to store child of this overlap gaussian
                            gv.childOverlaps.append([]) 
                            
                            parents.append([i,elements])
                            overlaps.append([])
                            
                            # store the position of the child (vecIndex) in the root (i)
                            gv.childOverlaps[i].append(vecIndex) 
                            
                            vecIndex+=1
                        
            i+=1
            
        startLevel = nextLevel
        nextLevel = len(gv.gaussians)
        gv.levels.append(nextLevel)
        
        level +=1
    
    gv.centroid/=gv.volume #normalise the centroid
    gv.overlap, gv.processQueue = Molecule_overlap(gv,gv)
    gv.parents = parents
    

    
    return gv

#%%
    '''Build up the mass matrix'''
def initOrientation(gv = GaussianVolume):
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
        else:            # for odd number of atom, positive contribution
            mass_matrix[0][0] += i.volume * i.centre[0] * i.centre[0]
            mass_matrix[0][1] += i.volume * i.centre[0] * i.centre[1]
            mass_matrix[0][2] += i.volume * i.centre[0] * i.centre[2]
            mass_matrix[1][1] += i.volume * i.centre[1] * i.centre[1]
            mass_matrix[1][2] += i.volume * i.centre[1] * i.centre[2]
            mass_matrix[2][2] += i.volume * i.centre[2] * i.centre[2]
        
    # set lower triangle due to its sysmetry       	
    mass_matrix[1][0] = mass_matrix[0][1]
    mass_matrix[2][0] = mass_matrix[0][2]
    mass_matrix[2][1] = mass_matrix[1][2]
    
    #normalise
    mass_matrix /= gv.volume
    
    #singular value decomposition
    gv.rotation, s, vh = np.linalg.svd(mass_matrix)
    
    #project the atoms' coordinates onto the principle axes
    if np.linalg.det(gv.rotation) < 0:
        gv.rotation[:][2] = - gv.rotation[:][2]
        
    for i in gv.gaussians:
        i.centre = gv.rotation.dot(i.centre)
              
    return gv


#%%

def Molecule_overlap(gRef = GaussianVolume, gDb = GaussianVolume):
    processQueue=[] #Stores the pair of gaussians that needed to calculate overlap
    overlap_volume = 0
    N1 = gRef.levels[0] #Reference molecule
    N2 = gDb.levels[0] #Database molecule
    
    '''loop over the atoms in both molecules'''
    i=0
    EPS = 0.03
    while i < N1:
        j=0
        while j < N2:
           g_ij = atomIntersection(a = gRef.gaussians[i], b=gDb.gaussians[j])
           V_ij = g_ij.volume
           
           if V_ij / (gRef.gaussians[i].volume + gDb.gaussians[j].volume - V_ij) > EPS:
               
               overlap_volume += V_ij
               
               # Loop over child nodes and add to queue
               d1 = gRef.childOverlaps[i]
               d2 = gDb.childOverlaps[j]
               
               # First add (i,child(j))
               if d2:
                   for it1 in d2:
                       processQueue.append([i,it1])
            
            #Second add (child(i),j)
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
                 #Add (child(i),j)
                for it1 in d1:
                    if [it1,j] not in processQueue:
                        processQueue.append([it1,j])
            else:
                
                if d2:
                    # add (i,child(j))
                    for it1 in d2:
                        if [i,it1] not in processQueue:
                            processQueue.append([i,it1])
                if d1 and gDb.gaussians[j].n - gRef.gaussians[i].n < 2:
                    # add (child(i),j)
                    for it1 in d1:
                        if [it1,j] not in processQueue:
                            processQueue.append([it1,j])

    return overlap_volume, processQueue
                    

def getScore(name, Voa, Vra, Vda):
    
    if name == 'tanimoto':
        return Voa/(Vra+Vda-Voa)
    elif name == 'tversky_ref':
        return Voa / (0.95*Vra + 0.05*Vda)
    elif name == 'tversky_db':
        return Voa/(0.05*Vra+0.95*Vda)
    
    return 0.0

def checkVolumes(gRef = GaussianVolume, gDb = GaussianVolume,
                 res = AlignmentInfo()):
    
    if res.overlap > gRef.overlap:
        res.overlap = gRef.overlap
        
    if res.overlap > gDb.overlap:
        res.overlap = gDb.overlap
        
    return 

    
    
    
#%%
gv =  Molecule_volume(gv)


            
        
        
        
        
                        
                    
                            
                            
                    
                    
            
        
    
    
    
        
        


                
        
            
        
        
        
                       
                   
                    
                   
               