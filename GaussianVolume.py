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
        self.gaussians = []
        
        # store the overlap tree used for molecule molecule intersection
        self.childOverlaps = [] 
        
        # Store the number of gaussians for each level
        self.levels = []
                
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
#%%'

def Molecule_volume(mol = Chem.rdchem.Mol(),  gv = GaussianVolume()):
    
    N = 0
    for atom in mol.GetAtoms():
        #if atom.GetAtomicNum() != 1: # ignore hydrogen atom
        N+=1
        gv.childOverlaps.append([])
        gv.gaussians.append(AtomGaussian())
            
    gv.levels.append(N)
    gv.volume = 0.0
    gv.centroid = np.array([0.0, 0.0, 0.0])
    
    # Stores the parents of gv.gaussians[i] inside parents[i]
    parents=[[] for i in range(N)]  
    
    # Stores the atom index that have intersection with i_th gaussians inside overlaps[i]
    overlaps=[[] for i in range(N)] 
    
    atomIndex = 0
    vecIndex = N #Used to indicated the initial position of the child gaussian
    
    guassian_weight = 2.70
    
    AtomID = 0  
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        
        gv.gaussians[atomIndex].centre = np.array(conf.GetAtomPosition(AtomID))
        gv.gaussians[atomIndex].alpha = GAlpha(atom.GetAtomicNum())
        gv.gaussians[atomIndex].weight = guassian_weight 
        radius_VDW = Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())
        gv.gaussians[atomIndex].volume = (4.0 * np.pi/3.0) * radius_VDW **3 
        gv.gaussians[atomIndex].n = 1 
           
        '''Update volume and centroid of the Molecule'''
        gv.volume += gv.gaussians[atomIndex].volume
        gv.centroid += gv.gaussians[atomIndex].volume*gv.gaussians[atomIndex].centre
        
        '''loop over every atom to find the second level overlap'''
        i=0
        while i < atomIndex:
        
            atom1 = gv.gaussians[i]
            atom2= gv.gaussians[atomIndex]
            
            ga = atomIntersection(a=atom1, b=atom2)
          
     
            EPS = 0.003
            # Check if overlap is sufficient enough
            if ga.volume / (gv.gaussians[i].volume + gv.gaussians[atomIndex].volume - ga.volume) > EPS:
        
                gv.volume -=  ga.volume
                gv.centroid -=   ga.volume*ga.centre
                
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
        AtomID +=1
        
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
                                gv.volume -=  ga.volume
                                gv.centroid -=   ga.volume*ga.centre
                            else:         # odd number overlaps give negative contribution
                                gv.volume +=  ga.volume
                                gv.centroid +=  ga.volume*ga.centre
                            
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
    
     
    gv.overlap = Molecule_overlap(gv,gv)
    gv.parents = parents
    

    
    return gv

#%%
    '''Build up the mass matrix'''
def initOrientation(gv = GaussianVolume()):
    mass_matrix = np.zeros(shape=(3,3))
    
    gv.centroid /= gv.volume #normalise the centroid

    for i in gv.gaussians:
        i.centre -= gv.centroid
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

def Molecule_overlap(gRef = GaussianVolume(), gDb = GaussianVolume()):
    processQueue=[] #Stores the pair of gaussians that needed to calculate overlap
    overlap_volume = 0
    N1 = gRef.levels[0] #Reference molecule
    N2 = gDb.levels[0] #Database molecule
    
    '''loop over the atoms in both molecules'''
    EPS = 0.03
    for i in range(0,N1):
        for j in range(0,N2):
            
            Cij = gRef.gaussians[i].alpha * gDb.gaussians[j].alpha / (gRef.gaussians[i].alpha + gDb.gaussians[j].alpha)
            			
            # Variables to store sum and difference of components
            d = (gRef.gaussians[i].centre - gDb.gaussians[j].centre)
            d_sqr = sum(d**2)
            			
            # Compute overlap volume
            V_ij = gRef.gaussians[i].weight * gDb.gaussians[j].weight * np.exp(- Cij * (d_sqr )) * (np.pi/(gRef.gaussians[i].alpha + gDb.gaussians[j].alpha))**1.5 

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
    
                      
    
    while len(processQueue) != 0: # loop when processQueue is not empty
    
        pair = processQueue.pop(0)
               
        i = pair[0]
        j = pair[1]
        
        Cij = gRef.gaussians[i].alpha * gDb.gaussians[j].alpha / (gRef.gaussians[i].alpha + gDb.gaussians[j].alpha)
            			
        # Variables to store sum and difference of components
        d = (gRef.gaussians[i].centre - gDb.gaussians[j].centre)
        d_sqr = sum(d**2)
        			
        # Compute overlap volume
        V_ij = gRef.gaussians[i].weight * gDb.gaussians[j].weight * np.exp(- Cij * (d_sqr )) * (np.pi/(gRef.gaussians[i].alpha + gDb.gaussians[j].alpha))**1.5 
      
        if V_ij / (gRef.gaussians[i].volume + gDb.gaussians[j].volume - V_ij) > EPS:
                           
            if (gRef.gaussians[i].n + gDb.gaussians[j].n)%2 ==0: 
                
                overlap_volume += V_ij
            else:
                overlap_volume -= V_ij
                
                
            d1 = gRef.childOverlaps[i]
            d2 = gDb.childOverlaps[j]
            
            
            if d1 and gRef.gaussians[i].n >  gDb.gaussians[j].n:
                 #Add (child(i),j)
                for it1 in d1:
  
                    processQueue.append([it1,j])
            else:
                
                if d2:
                    # add (i,child(j))
                    for it1 in d2:
                        processQueue.append([i,it1])
                if d1 and gDb.gaussians[j].n - gRef.gaussians[i].n < 2:
                    # add (child(i),j)
                    for it1 in d1:           
                        processQueue.append([it1,j])

    return overlap_volume
                    

def getScore(name, Voa, Vra, Vda):
    
    if name == 'tanimoto':
        #print('Voa_' + str(Voa) +'Vra_' + str(Vra) + 'Vda_' + str(Vda))
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
#gv =  Molecule_volume(gv)


            
        
        
        
        
                        
                    
                            
                            
                    
                    
            
        
    
    
    
        
        


                
        
            
        
        
        
                       
                   
                    
                   
               