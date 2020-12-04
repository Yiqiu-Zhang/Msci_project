# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:53:39 2020

@author: THINKPAD
"""
import numpy as np
from collections import deque

from AlignmentInfo import AlignmentInfo
from AtomGaussian import AtomGaussian
from GaussianVolume import GaussianVolume

class ShapeAlignment(GaussianVolume):
    
    def __init__(self, gRef = GaussianVolume(),gDb = GaussianVolume()):
        
        self._gRef = gRef
        self._gDb = gDb
        self._rAtom = gRef.levels[0]
        self._rGauss = len(gRef.gaussians)
        self._dAtom = gDb.levels[0]
        self._dGauss = len(gDb.gaussians)
        self._maxSize = self._rGauss * self._dGauss + 1
        self._maxIter = 15
        self._matrixMap = [[] for i in range(0,self._rGauss) for j in range(0,self._dGauss)]
        
    def gradientAscent(self, rotor):
        
        
        processQueue= deque()
        
        Aij = 0
        Aq = np.zeros(4)
        
        Vij = 0
        qAq = 0
        
        d1 = []
        d2 = []
        
        overGrad = np.zeros(4)
        overHessian = np.zeros((4,4))
        
        atomOverlap = 0
        # pharmOverlap = 0
        
        res = AlignmentInfo()
        
        oldVolume = 0
        iterations = 0
        mapIndex = 0
        
        while iterations <20:
            atomOverlap = 0
            # pharmOverlap = 0
            iterations+=1
            
            overGrad = np.zeros(4)
            overHessian = np.zeros((4,4))
            
            xlambda = 0
            
            for i in range(0,self._rAtom):
                for j in range(0,self._dAtom):
                    mapIndex = (i * self._dGauss) + j
                    
                    #Sub in the Aij to corresponding location, or calculate& sub in if empty initially
                    
                    if len(self._matrixMap[mapIndex]) == 0:
                        
                        Aij = self._updateMatrixMap(self._gRef.gaussians[i], self._gDb.gaussians[j])
                        
                        
                        self._matrixMap[mapIndex] = Aij
                        
                    else:
                        Aij = self._matrixMap[mapIndex]
                    
                    
                    #Calculation of q′Aq, rotor product
                    Aq[0] =  Aij[0] * rotor[0] +  Aij[1] * rotor[1] +  Aij[2] * rotor[2] +  Aij[3] * rotor[3]
                    Aq[1] =  Aij[4] * rotor[0] +  Aij[5] * rotor[1] +  Aij[6] * rotor[2] +  Aij[7] * rotor[3]
                    Aq[2] =  Aij[8] * rotor[0] +  Aij[9] * rotor[1] + Aij[10] * rotor[2] + Aij[11] * rotor[3]
                    Aq[3] = Aij[12] * rotor[0] + Aij[13] * rotor[1] + Aij[14] * rotor[2] + Aij[15] * rotor[3] 
                        
                    qAq = rotor[0] * Aq[0] + rotor[1]*Aq[1] + rotor[2]*Aq[2] + rotor[3]*Aq[3]      
                    
                    #Volume overlap rewritten
                    
                    Vij = Aij[16] * np.exp( -qAq )
                    
                    EPS = 0.03
                    if  Vij/(self._gRef.gaussians[i].volume + self._gDb.gaussians[j].volume - Vij ) > EPS:
                        
                        atomOverlap += Vij
                        
                        v2 = 2.0 * Vij #simply Vij doubled, for lateral use
                        
                        #Lagrange coefficient lambda
                        xlambda -= v2 * qAq
                        
                        #overGrad calculated, gradient minus since
                        overGrad -= v2 * Aq
                        
                        # overHessian += 2*Vij(2*Aijq'qAij-Aij); (only upper triangular part)
                        overHessian[0][0] += v2 * (2.0 * Aq[0]*Aq[0] - Aij[0])
                        overHessian[0][1] += v2 * (2.0 * Aq[0]*Aq[1] - Aij[1])
                        overHessian[0][2] += v2 * (2.0 * Aq[0]*Aq[2] - Aij[2])
                        overHessian[0][3] += v2 * (2.0 * Aq[0]*Aq[3] - Aij[3])
                        overHessian[1][1] += v2 * (2.0 * Aq[1]*Aq[1] - Aij[5])
                        overHessian[1][2] += v2 * (2.0 * Aq[1]*Aq[2] - Aij[6])
                        overHessian[1][3] += v2 * (2.0 * Aq[1]*Aq[3] - Aij[7])
                        overHessian[2][2] += v2 * (2.0 * Aq[2]*Aq[2] - Aij[10])
                        overHessian[2][3] += v2 * (2.0 * Aq[2]*Aq[3] - Aij[11])
                        overHessian[3][3] += v2 * (2.0 * Aq[3]*Aq[3] - Aij[15])
                        
                        #loop over child nodes and add to queue
                        d1 = self._gRef.childOverlaps[i]
                        d2 = self._gDb.childOverlaps[j]
                        
                        if d2:
                            for it1 in d2:
                                processQueue.append([i,it1])
                                
                        if d1:
                            for it1 in d1:
                                processQueue.append([it1,j])
                                
                           
            
            while len(processQueue) != 0: # processQueue is not empty
                pair = processQueue.popleft()               
                 
                i = pair[0]
                j = pair[1]
                 
                mapIndex = (i*self._dGauss) + j
                
                #Sub in the Aij to corresponding location, or calculate& sub in if empty initially 
                if len(self._matrixMap[mapIndex]) == 0:
                        
                    Aij = self._updateMatrixMap(a = self._gRef.gaussians[i], b = self._gDb.gaussians[j])
                    
                    self._matrixMap[mapIndex] = Aij
                    
                else:
                    Aij = self._matrixMap[mapIndex]
                    
                #rotor product        
                Aq[0] =  Aij[0] * rotor[0] +  Aij[1] * rotor[1] +  Aij[2] * rotor[2] +  Aij[3] * rotor[3]
                Aq[1] =  Aij[4] * rotor[0] +  Aij[5] * rotor[1] +  Aij[6] * rotor[2] +  Aij[7] * rotor[3]
                Aq[2] =  Aij[8] * rotor[0] +  Aij[9] * rotor[1] + Aij[10] * rotor[2] + Aij[11] * rotor[3]
                Aq[3] = Aij[12] * rotor[0] + Aij[13] * rotor[1] + Aij[14] * rotor[2] + Aij[15] * rotor[3] 
                    
                qAq = rotor[0] * Aq[0] + rotor[1]*Aq[1] + rotor[2]*Aq[2] + rotor[3]*Aq[3]      
                    
                Vij = Aij[16] * np.exp( -qAq )
                
                
                EPS = 0.03
                if  abs(Vij)/(self._gRef.gaussians[i].volume + self._gDb.gaussians[j].volume - abs(Vij) ) > EPS:
                    
                    atomOverlap += Vij
                    
                    
                    v2 = 2.0 * Vij
                    xlambda -= v2 * qAq
                    
                    overGrad -= v2 * Aq
                    
                    # overHessian += 2*Vij(2*Aijq'qAij-Aij); (only upper triangular part)
                    overHessian[0][0] += v2 * (2.0 * Aq[0]*Aq[0] - Aij[0])
                    overHessian[0][1] += v2 * (2.0 * Aq[0]*Aq[1] - Aij[1])
                    overHessian[0][2] += v2 * (2.0 * Aq[0]*Aq[2] - Aij[2])
                    overHessian[0][3] += v2 * (2.0 * Aq[0]*Aq[3] - Aij[3])
                    overHessian[1][1] += v2 * (2.0 * Aq[1]*Aq[1] - Aij[5])
                    overHessian[1][2] += v2 * (2.0 * Aq[1]*Aq[2] - Aij[6])
                    overHessian[1][3] += v2 * (2.0 * Aq[1]*Aq[3] - Aij[7])
                    overHessian[2][2] += v2 * (2.0 * Aq[2]*Aq[2] - Aij[10])
                    overHessian[2][3] += v2 * (2.0 * Aq[2]*Aq[3] - Aij[11])
                    overHessian[3][3] += v2 * (2.0 * Aq[3]*Aq[3] - Aij[15])
                    
                    #loop over child nodes and add to queue
                    d1 = self._gRef.childOverlaps[i]
                    d2 = self._gDb.childOverlaps[j]
                    
                    if d1 and self._gRef.gaussians[i].n >self._gDb.gaussians[j].n:
                        for it1 in d1:
                            processQueue.append([it1,j])
                    else:     
                        if d2:
                            for it1 in d2:
                                processQueue.append([i,it1])
                                
                        if d1 and self._gDb.gaussians[j].n - self._gRef.gaussians[i].n <2:
                            for it1 in d1:        
                                processQueue.append([it1,j])
                      
            
            
            #check if the new volume is better than the previously found one
            #if not quit the loop
            if iterations > 6 and atomOverlap < oldVolume + 0.0001:
                break
                
            oldVolume = atomOverlap #store latest volume found
            
            #no measurable overlap between two volumes
            if np.isnan(xlambda) or np.isnan(oldVolume) or oldVolume == 0:
                break
            
            #update solution 
            if oldVolume > res.overlap:
                
                res.overlap = atomOverlap
                res.rotor = rotor
                
                if  res.overlap/(self._gRef.overlap + self._gDb.overlap - res.overlap)  > 0.99 :
                    
                    break
                    
            overHessian -= xlambda #update the gradient and hessian 
            
            #fill lower triangular of the hessian matrix
            overHessian[1][0] = overHessian[0][1]
            overHessian[2][0] = overHessian[0][2]
            overHessian[2][1] = overHessian[1][2]
            overHessian[3][0] = overHessian[0][3]
            overHessian[3][1] = overHessian[1][3]
            overHessian[3][2] = overHessian[2][3]
            
            #update gradient to make h
            overGrad[0] -= xlambda * rotor[0]
            overGrad[1] -= xlambda * rotor[1]
            overGrad[2] -= xlambda * rotor[2]
            overGrad[3] -= xlambda * rotor[3]
            
            #line 354
            #update gradient based on inverse hessian
            temp = overHessian.dot(overGrad)
            h = overGrad* temp
            
            #small scaling of the gradient
            for i in range(0,len(h)):
                if h[i] != 0:
                    h[i] = 1/h[i] * (sum(i**2 for i in overGrad))
                    
            overGrad *= h
            
            # update rotor based on gradient information
            rotor -= overGrad
            
            #normalise rotor such that it has unit norm
            nr = np.sqrt(sum(i**2 for i in rotor))
            
            rotor /= nr
            
        return res
    
    def simulatedAnnealing(self, rotor):
        
        rotor = np.zeros(4)
        processQueue= deque() #create a queue to hold the pairs to process
        
        Aij = 0
        Aq = np.zeros(4)
        
        Vij = 0
        qAq = 0
        
        d1 = []
        d2 = []
        
        dTemperature = 1.1
        
        res = AlignmentInfo()
        
        oldVolume = 0
        # bestVolume = 0
        iterations = 0
        sameCount = 0
        mapIndex = 0
        
        while iterations < self._maxIter:
            
            #reset volume
            atomOverlap = 0
            # pharmOverlap = 0 
            iterations += 1
            
            #temperature of the simulated annealing step
            T = np.sqrt((1.0 + iterations)/dTemperature)
            
            #create atom-atom overlaps 
            for i in range(0,self._rAtom):
                for j in range(0,self._dAtom):
                    
                    mapIndex = (i * self._dGauss) + j
                    
                    #Sub in the Aij to corresponding location, or calculate& sub in if empty initially
                    if len(self._matrixMap[mapIndex]) == 0:
                        
                        Aij = self._updateMatrixMap(self._gRef.gaussians[i], self._gDb.gaussians[j])
                        
                        self._matrixMap[mapIndex] = Aij
                        
                    else:
                        Aij = self._matrixMap[mapIndex]

                        
                    Aq[0] =  Aij[0] * rotor[0] +  Aij[1] * rotor[1] +  Aij[2] * rotor[2] +  Aij[3] * rotor[3]
                    Aq[1] =  Aij[4] * rotor[0] +  Aij[5] * rotor[1] +  Aij[6] * rotor[2] +  Aij[7] * rotor[3]
                    Aq[2] =  Aij[8] * rotor[0] +  Aij[9] * rotor[1] + Aij[10] * rotor[2] + Aij[11] * rotor[3]
                    Aq[3] = Aij[12] * rotor[0] + Aij[13] * rotor[1] + Aij[14] * rotor[2] + Aij[15] * rotor[3] 
                        
                    qAq = rotor[0] * Aq[0] + rotor[1]*Aq[1] + rotor[2]*Aq[2] + rotor[3]*Aq[3]      
                        
                    Vij = Aij[16] * np.exp( -qAq )
                    
                    EPS = 0.03
                    if  Vij/(self._gRef.gaussians[i].volume + self._gDb.gaussians[j].volume - Vij ) > EPS:
                        
                        atomOverlap += Vij
                        
                        d1 = self._gRef.childOverlaps[i]
                        d2 = self._gDb.childOverlaps[j]
                        
                        if d2:
                            for it1 in d2:
                                processQueue.append([i,it1])
                        
                        if d1:
                            for it1 in d1:
                                processQueue.append([it1,j])
                                
                                
            while len(processQueue) != 0: # processQueue is not empty
                pair = processQueue.popleft()
                
                i = pair[0]
                j = pair[1]
                
                mapIndex = (i * self._dGauss) + j
                
                #Sub in the Aij to corresponding location, or calculate& sub in if empty initially    
                # matIter = self._matrixMap[mapIndex]
                if len(self._matrixMap[mapIndex]) == 0:
                        
                    Aij = self._updateMatrixMap(self._gRef.gaussians[i], self._gDb.gaussians[j])
                    
                    self._matrixMap[mapIndex] = Aij
                    
                else:
                    Aij = self._matrixMap[mapIndex]
                    
                Aq[0] =  Aij[0] * rotor[0] +  Aij[1] * rotor[1] +  Aij[2] * rotor[2] +  Aij[3] * rotor[3]
                Aq[1] =  Aij[4] * rotor[0] +  Aij[5] * rotor[1] +  Aij[6] * rotor[2] +  Aij[7] * rotor[3]
                Aq[2] =  Aij[8] * rotor[0] +  Aij[9] * rotor[1] + Aij[10] * rotor[2] + Aij[11] * rotor[3]
                Aq[3] = Aij[12] * rotor[0] + Aij[13] * rotor[1] + Aij[14] * rotor[2] + Aij[15] * rotor[3] 
                    
                qAq = rotor[0] * Aq[0] + rotor[1]*Aq[1] + rotor[2]*Aq[2] + rotor[3]*Aq[3]      
                    
                Vij = Aij[16] * np.exp( -qAq )
                
                EPS = 0.03
                if  abs(Vij)/(self._gRef.gaussians[i].volume + self._gDb.gaussians[j].volume - abs(Vij) ) > EPS:
                
                    atomOverlap += Vij
                    
                    d1 = self._gRef.childOverlaps[i]
                    d2 = self._gDb.childOverlaps[j]
                
                    if d1 and self._gRef.gaussians[i].n >self._gDb.gaussians[j].n:
                        for it1 in d1:
                            processQueue.append([it1,j])
                    else:     
                        if d2:
                            for it1 in d2:
                                processQueue.append([i,it1])
                                
                        if d1 and self._gDb.gaussians[j].n - self._gRef.gaussians[i].n <2:
                            for it1 in d1:        
                                processQueue.append([it1,j])
            
            overlapVol = atomOverlap
            
            #check if the new volume is better than the previously found one
            if overlapVol < oldVolume:
                
                D = np.exp(-np.sqrt(oldVolume - overlapVol))/T
                
                if D > np.random.random():
                    
                    oldRotor = rotor
                    oldVolume = overlapVol
                    sameCount = 0
                else:
                    sameCount +=1
                    if sameCount == 30:
                        iterations = self._maxIter
                        
            else:
                #store latest volume found
                oldVolume = overlapVol
                oldRotor = rotor
                
                #update best found so far
                # bestRotor = rotor
                # bestVolume = overlapVol
                sameCount = 0
                
                #check if it is better than the best solution found so far
                if overlapVol > res.overlap:
                    res.overlap = atomOverlap
                    res.rotor = rotor
                    
                    if (res.overlap / (self._gRef.overlap + self._gDb.overlap - res.overlap )) > 0.99:
                        break
               
            #make random permutation & double range = 0.05		
            ranPermutation = 0.1/T
            rotor = oldRotor - ranPermutation + 2* ranPermutation*np.random.random()
            
            #normalise rotor such that it has unit norm
            nr = np.sqrt(sum(i**2 for i in rotor))
            rotor /= nr
            
        return res
    
    def setMaxIterations(self, i):
        
        self._maxIter = i
        return
                               
    def _updateMatrixMap(self, a= AtomGaussian(), b = AtomGaussian()):
        
        #A is a matrix thatsolely depends on the position of the two centres of Gaussians a and b
        A = np.zeros(17)
        
        d = (a.centre - b.centre)
        s = (a.centre + b.centre)
        d2 = sum(i**2 for i in d) # The distance squared between two gaussians
        # s2 = sum(i**2 for i in s)
    
        weight = a.alpha * b.alpha /(a.alpha + b.alpha)
        
        A[0] = d2
        A[1] = d[1]*s[2] - d[2]*s[1]
        A[2] = d[2]*s[0] - d[0]*s[2]
        A[3] = d[0]*s[1] - d[1]*s[0]
        A[4] = A[1]
        A[5] = d[0]**2 + s[1]**2 + s[2]**2
        A[6] = d[0]*d[1] - s[0]*s[1]
        A[7] = d[0]*d[2] - s[0]*s[2]
        A[8] = A[2]
        A[9] = A[6]
        A[10] = s[0]**2 + d[1]**2 + s[2]**2
        A[11] = d[1]*d[2] - s[1]*s[2]
        A[12] = A[3]
        A[13] = A[7]
        A[14] = A[11]
        A[15] = s[0]**2 + s[1]**2 + d[2]**2
        
        A = A*weight
        
        if ((a.n + b.n) %2) == 0:
            A[16] = a.weight * b.weight * (np.pi/(a.alpha + b.alpha))**1.5      
        else:
            A[16] = - a.weight * b.weight * (np.pi/(a.alpha + b.alpha))**1.5
        return A    
                    
                    
                    
                    
                
                
                    
                    
            
            
            