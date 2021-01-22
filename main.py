# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:12:45 2020

@author: THINKPAD
"""

import numpy as np
#from openbabel import openbabel
from rdkit import Chem
from AlignmentInfo import AlignmentInfo
from GaussianVolume import GaussianVolume, Molecule_volume, initOrientation, getScore, checkVolumes
from ShapeAlignment import ShapeAlignment
from SolutionInfo import SolutionInfo, updateSolutionInfo
from rdkit.Chem import AllChem
import moleculeRotation #import positionMolecule, repositionMolecule, rotateMolecule
maxIter = 0
# write from main.cpp line 124, need to add molecule information from rdkit
#refMol = Chem.MolFromSmiles('NS(=O)(=O)c1ccc(C(=O)N2Cc3ccccc3C(c3ccccc3)C2)cc1')
refMol = Chem.MolFromMolFile('ref.mol')
#refMol = Chem.MolFromMolFile('sangetan.mol')

#pre_refMol = Chem.MolFromSmiles('COc1ccc(-c2nc3c4ccccc4ccc3n2C(C)C)cc1')   
#pre_refMol = Chem.MolFromMolFile('AAR.mol')
#pre_refMol_H=Chem.AddHs(pre_refMol)
#AllChem.EmbedMolecule(pre_refMol_H) 
#AllChem.MMFFOptimizeMolecule(pre_refMol_H)
#refMol = Chem.RemoveHs(pre_refMol_H)


refVolume = GaussianVolume()

# List all Gaussians and their respective intersections
Molecule_volume(refMol,refVolume)
print(refVolume.volume)
print(refVolume.overlap)
#%%
# 	Move the Gaussian towards its center of geometry and align with principal axes
initOrientation(refVolume)
print(refVolume.overlap)

#%%

# dodge 132-136 , 146 - 172 read database, dodge all scoreonly process

#Create a class to hold the best solution of an iteration
bestSolution = SolutionInfo()
bestSolution.refAtomVolume = refVolume.overlap
bestSolution.refCenter = refVolume.centroid
bestSolution.refRotation = refVolume.rotation

# Create the set of Gaussians of database molecule
inf = open('DK+clean.sdf','rb')#
fsuppl = Chem.ForwardSDMolSupplier(inf)
Molcount = 0
SolutionTable = np.zeros([101,4])
for dbMol in fsuppl: 
    #if dbMol.GetProp('zinc_id') != 'ZINC000000017630': continue
    if Molcount >= 100: continue
    if dbMol is None: continue

    dbName = dbMol.GetProp('zinc_id')
    
    Molcount+=1

    #pre_dbMol = Chem.MolFromMolFile('GAR.mol')
    #pre_dbMol_H=Chem.AddHs(pre_dbMol)
    #AllChem.EmbedMolecule(pre_dbMol_H) 
    #AllChem.MMFFOptimizeMolecule(pre_dbMol_H)
    #dbMol = Chem.RemoveHs(pre_dbMol_H) 
    #dbMol = Chem.MolFromMolFile('AAR.mol')
    #dbMol = Chem.MolFromMolFile('sangetan.mol')

    dbVolume = GaussianVolume()
    Molecule_volume(dbMol,dbVolume)
    
    res = AlignmentInfo()
    bestScore = 0.0
    
    initOrientation(dbVolume)
    aligner = ShapeAlignment(refVolume,dbVolume)
    aligner.setMaxIterations(maxIter) # !!!manully setted here
    aligner._maxIter
    for l in range(0,4):
        quat = np.zeros(4)
        quat[l] = 1.0
        
        nextRes = aligner.gradientAscent(quat)
        checkVolumes(refVolume, dbVolume, nextRes)
        ss = getScore('tanimoto', nextRes.overlap, refVolume.overlap, dbVolume.overlap)
        
        if ss > bestScore:
            
            res = nextRes
            bestScore = ss
                   
        if bestScore > 0.98: # 
            break

    if maxIter > 0: #!!!
        nextRes = aligner.simulatedAnnealing(res.rotor)
        checkVolumes(refVolume, dbVolume, nextRes)
        ss = getScore('tanimoto', nextRes.overlap, refVolume.overlap, dbVolume.overlap)
        if (ss > bestScore):
            bestScore = ss
            res = nextRes
                 
    updateSolutionInfo(bestSolution, res, bestScore, dbVolume)
    bestSolution.dbMol = dbMol
    bestSolution.dbName = dbName  # need to use rdkit
    
    if bestSolution.score > 0.7: #!!! the cutoff value
        
        '''post-process molecules'''
        
        # Translate and rotate the molecule towards its centroid and inertia axes
        moleculeRotation.positionMolecule(bestSolution.dbMol, bestSolution.dbCenter, bestSolution.dbRotation)
                
       	# Rotate molecule with the optimal
        moleculeRotation.rotateMolecule(bestSolution.dbMol, bestSolution.rotor)
       
       	# Rotate and translate the molecule with the inverse rotation and translation of the reference molecule
        moleculeRotation.repositionMolecule(bestSolution.dbMol,refVolume.centroid, refVolume.rotation )
           

    SolutionTable[Molcount] = np.array([bestSolution.score,bestSolution.atomOverlap,bestSolution.refAtomVolume,bestSolution.dbAtomVolume])
#%%
np.savetxt("foo.csv", SolutionTable, delimiter=",")
#%%
imported_db =  Chem.MolToMolBlock(bestSolution.dbMol,confId=-1)    
imported_ref = Chem.MolToMolBlock(refMol,confId=-1) 
with open(r"C:\Users\THINKPAD\Desktop\Msci project\Msci_project_code\Msci_project\Data\db.mol", "w") as newfile:
       newfile.write(imported_db)
       
with open(r"C:\Users\THINKPAD\Desktop\Msci project\Msci_project_code\Msci_project\Data\ref.mol", "w") as newfile:
       newfile.write(imported_ref)
    
        

