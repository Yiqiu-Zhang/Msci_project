# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:12:45 2020

@author: THINKPAD
"""

import numpy as np
#from openbabel import openbabel
from rdkit import Chem
from AlignmentInfo import AlignmentInfo
from AtomGaussian import AtomGaussian, atomIntersection
from GaussianVolume import GaussianVolume, Molecule_volume, Molecule_overlap, initOrientation, getScore, checkVolumes
from ShapeAlignment import ShapeAlignment
from SolutionInfo import SolutionInfo, updateSolutionInfo, setAllScores

# write from main.cpp line 124, need to add molecule information from rdkit

refVolume = GaussianVolume

# List all Gaussians and their respective intersections
Molecule_volume(refVolume)

# 	Move the Gaussian towards its center of geometry and align with principal axes
initOrientation(refVolume)

# dodge 132-136 , 146 - 172 read database, dodge all scoreonly process

#Create a class to hold the best solution of an iteration
bestSolution = SolutionInfo
bestSolution.refAtomVolume = refVolume.overlap
bestSolution.refCenter = refVolume.centroid
bestSolution.refRotation = refVolume.rotation

# Create the set of Gaussians of database molecule
dbVolume = GaussianVolume
Molecule_volume(dbVolume)

res = AlignmentInfo
bestScore = 0.0

initOrientation(dbVolume)
aligner = ShapeAlignment(refVolume,dbVolume)
aligner.setMaxIterations(20) # !!!manully setted here

for l in range(0,4):
    quat = np.zeros(4)
    quat[l] = 1.0
    
    nextRes = aligner.gradientAscent(quat)
    checkVolumes(refVolume, dbVolume, nextRes)
    ss = getScore('no_name', nextRes.overlap, refVolume.overlap, dbVolume.overlap)
    
    if ss > bestScore:
        res = nextRes
        bestScore = ss
    if bestScore > 0.98:
        break
    # line 209
        

