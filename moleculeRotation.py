# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 14:39:34 2020

@author: THINKPAD
"""
from rdkit import Chem
import numpy as np
from rdkit.Geometry import Point3D
from SolutionInfo import SolutionInfo
from GaussianVolume import GaussianVolume

def positionMolecule(mol = Chem.rdchem.Mol(), centroid=SolutionInfo().dbCenter, rotation = SolutionInfo().dbRotation):
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        
        x,y,z = rotation.dot(np.array(conf.GetAtomPosition(i)) - centroid)
        conf.SetAtomPosition(i,Point3D(x,y,z))
        
    return

def repositionMolecule(mol = Chem.rdchem.Mol(), centroid=GaussianVolume().centroid, rotation = GaussianVolume().rotation):
    conf = mol.GetConformer()
    
    for i in range(mol.GetNumAtoms()):
        #  Get coordinates
        x, y, z = rotation.dot(np.array(conf.GetAtomPosition(i))) + centroid
        conf.SetAtomPosition(i,Point3D(x,y,z))
    return

def rotateMolecule(mol = Chem.rdchem.Mol(), rotor = SolutionInfo().rotor):
    
    rot = np.zeros(9).reshape(3,3)
    r1 = rotor[1] * rotor[1]
    r2 = rotor[2] * rotor[2]
    r3 = rotor[3] * rotor[3]
	
    rot[0][0] = 1.0 - 2.0*r2 - 2.0*r3
    rot[0][1] = 2.0 * (rotor[1]*rotor[2] - rotor[0]*rotor[3])
    rot[0][2] = 2.0 * (rotor[1]*rotor[3] + rotor[0]*rotor[2])
    rot[1][0] = 2.0 * (rotor[1]*rotor[2] + rotor[0]*rotor[3])
    rot[1][1] = 1.0 - 2*r3 - 2*r1
    rot[1][2] = 2.0 * (rotor[2]*rotor[3] - rotor[0]*rotor[1])
    rot[2][0] = 2.0 * (rotor[1]*rotor[3] - rotor[0]*rotor[2])
    rot[2][1] = 2.0 * (rotor[2]*rotor[3] + rotor[0]*rotor[1])
    rot[2][2] = 1.0 - 2*r2 - 2*r1
    
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
         x,y,z = rot.dot(np.array(conf.GetAtomPosition(i)))
         conf.SetAtomPosition(i,Point3D(x,y,z))
         
