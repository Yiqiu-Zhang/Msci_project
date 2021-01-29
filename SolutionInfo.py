# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:13:36 2020

@author: THINKPAD
"""

import numpy as np
import AlignmentInfo
import GaussianVolume

class SolutionInfo():
    def __init__(self):
        self.refName = 'name'
        self.refAtomVolume = 0.0
        self.refCenter = [0,0,0]
        self.refRotation = 0
        self.dbName =""
        self.dbAtomVolume =0.0
        self.dbMol = 0
        self.dbCenter =[0,0,0]
        self.dbRotation = 0 
        self.atomOverlap = 0.0
        self.score = 0.0
        self.rotor = np.zeros((4))
        self.rotor[0] = 1.0
        
#%%
def updateSolutionInfo(s = SolutionInfo, res = AlignmentInfo, score = 0, gv = GaussianVolume):
    s.dbAtomVolume = gv.overlap
    s.dbCenter = gv.centroid
    s.dbRotation = gv.rotation
    s.atomOverlap = res.overlap
    s.score = score
    s.rotor = res.rotor   
    return 

def setAllScores(res = SolutionInfo):
    
    res.atomOverlap / (res.refAtomVolume + res.dbAtomVolume - res.atomOverlap)
    
    return
                      
    