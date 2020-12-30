# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 12:25:56 2020

@author: THINKPAD
"""

import py3Dmol
import numpy as np
#from openbabel import openbabel
from rdkit import Chem
from AlignmentInfo import AlignmentInfo
from GaussianVolume import GaussianVolume, Molecule_volume, initOrientation, getScore, checkVolumes
from ShapeAlignment import ShapeAlignment
from SolutionInfo import SolutionInfo, updateSolutionInfo

inf = open('DK+clean.sdf','rb')#
fsuppl = Chem.ForwardSDMolSupplier(inf)
Molcount = 0
for dbMol in fsuppl:
    if Molcount >= 1: continue
    if dbMol is None: continue

    dbName = dbMol.GetProp('zinc_id')