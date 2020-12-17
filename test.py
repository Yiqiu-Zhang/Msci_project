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
