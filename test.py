# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 12:25:56 2020

@author: THINKPAD
"""


import numpy as np
#from openbabel import openbabel
from rdkit import Chem
from AlignmentInfo import AlignmentInfo
from GaussianVolume import GaussianVolume, Molecule_volume, initOrientation, getScore, checkVolumes
from ShapeAlignment import ShapeAlignment
from SolutionInfo import SolutionInfo, updateSolutionInfo

# write from main.cpp line 124, need to add molecule information from rdkit
#refMol = Chem.MolFromSmiles('NS(=O)(=O)c1ccc(C(=O)N2Cc3ccccc3C(c3ccccc3)C2)cc1')
refMol = Chem.MolFromMolFile('GAR.mol')
#refMol = Chem.MolFromMolFile('sangetan.mol')

refVolume = GaussianVolume()

# List all Gaussians and their respective intersections
Molecule_volume(refMol,refVolume)
#%%
# 	Move the Gaussian towards its center of geometry and align with principal axes
initOrientation(refVolume)


# dodge 132-136 , 146 - 172 read database, dodge all scoreonly process

#Create a class to hold the best solution of an iteration
bestSolution = SolutionInfo()
bestSolution.refAtomVolume = refVolume.overlap
bestSolution.refCenter = refVolume.centroid
bestSolution.refRotation = refVolume.rotation

# Create the set of Gaussians of database molecule

#dbMol  = Chem.MolFromSmiles('O=C(CCl)N1Cc2ccccc2C(c2ccccc2)C1')
dbMol = Chem.MolFromMolFile('AAR.mol')
#dbMol = Chem.MolFromMolFile('sangetan.mol')

dbVolume = GaussianVolume()
Molecule_volume(dbMol,dbVolume)


res = AlignmentInfo()
bestScore = 0.0

initOrientation(dbVolume)
aligner = ShapeAlignment(refVolume,dbVolume)
aligner.setMaxIterations(20) # !!!manully setted here

for l in range(0,4):
    quat = np.zeros(4)
    quat[l] = 1.0
    
    nextRes = aligner.gradientAscent(quat)
    checkVolumes(refVolume, dbVolume, nextRes)
    ss = getScore('tanimoto', nextRes.overlap, refVolume.overlap, dbVolume.overlap)
    
    if ss > bestScore:
        
        res = nextRes
        bestScore = ss
    if bestScore > 0.98:
        break
    # line 209

 
updateSolutionInfo(bestSolution, res, bestScore, dbVolume)
#bestSolution.dbMol = dbMol
#bestSolution.dbName = dbName  # need to use rdkit
