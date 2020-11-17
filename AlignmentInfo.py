# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:57:05 2020

@author: THINKPAD
"""
import numpy as np
class AlignmentInfo():
    def __init__(self):
        self.overlap = 0
        self.rotor = np.array((1,0,0,0))