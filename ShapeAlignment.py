# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:53:39 2020

@author: THINKPAD
"""
import numpy as np
import GaussianVolume
import AlignmentInfo
import AtomGaussian

class ShapeAlignment(GaussianVolume):
    
    def __init__(self, gRef = GaussianVolume(),gDb = GaussianVolume()):
        
        self._gRef = gRef
        self.gDb = gDb
        self._rAtom = gRef.levels[0]
        self._rGauss = len(gRef.gaussians)
        self._dAtom = gDb.levels[0]
        self._dGauss = len(gDb.gaussians)
        self._maxSize = self._rGauss * self._dGauss + 1
        self._maxIter = 50
        self._matrixMap = np.ndarray((3,2))
        
    def gradientAscent(self,AlignmentInfo):
        
        rotor = np.zeros(4)
        processQueue=[]
        
        Aij = 0
        Aq = np.zeros(4)
        
        Vij = 0
        qAq = 0
        
        d1 = []
        d2 = []
        
        overGrad = np.zeros(4)
        overHessian = np.zeros((4,4))
        
        atomOverlap = 0
        pharmOverlap = 0
        
        res = AlignmentInfo()
        
        oldVolume = 0
        iterations = 0
        mapIndex = 0
        
        while iterations <20:
            atomOverlap = 0
            pharmOverlap = 0
            iterations+=1
            
            overGrad = 0
            overHessian = 0
            
            xlambda = 0
            
            for i in range(0,self._rAtom):
                for j in range(0,self._dAtom):
                    mapIndex = (i * self._dGauss) + j
                    
                    matIter = self._matrixMap[mapIndex]
                    if matIter == self._matrixMap[-1]:
                        
                        Aij = self._updateMatrixMap(self._gRef.gaussians[i], self._gDb.gaussians[j])
                        
                        self._matrixMap[mapIndex] = Aij
                        
                    else:#!!!!
                        Aij = 0
                        
                    Aq[0] =  Aij[0] * rotor[0] +  Aij[1] * rotor[1] +  Aij[2] * rotor[2] +  Aij[3] * rotor[3]
                    Aq[1] =  Aij[4] * rotor[0] +  Aij[5] * rotor[1] +  Aij[6] * rotor[2] +  Aij[7] * rotor[3]
                    Aq[2] =  Aij[8] * rotor[0] +  Aij[9] * rotor[1] + Aij[10] * rotor[2] + Aij[11] * rotor[3]
                    Aq[3] = Aij[12] * rotor[0] + Aij[13] * rotor[1] + Aij[14] * rotor[2] + Aij[15] * rotor[3] 
                        
                    qAq = rotor[0] * Aq[0] + rotor[1]*Aq[1] + rotor[2]*Aq[2] + rotor[3]*Aq[3]      
                        
                    Vij = Aij[16] * np.exp( -qAq )
                    
                    EPS = 0.03
                    if  Vij/(self._gRef.gaussians[i].volume + self._gDb.gaussians[j].volume - Vij ) > EPS:
                        
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
                        
                        d1 = self._gRef.childOverlaps[i]
                        d2 = self._gDb.childOverlaps[j]
                        
                        if d2:
                            for it1 in d2:
                                processQueue.append([i,it1])
                                
                        if d1:
                            for it1 in d1:
                                processQueue.append([it1,j])
                                
                            
                                    
            for item in processQueue:
                 
                i = item[0]
                j = item[1]
                 
                mapIndex = (i*self._dGauss) + j
                 
                matIter = self._matrixMap[mapIndex]
                if matIter == self._matrixMap[-1]:
                        
                    Aij = self._updateMatrixMap(self._gRef.gaussians[i], self._gDb.gaussians[j])
                        
                    self._matrixMap[mapIndex] = Aij
                        
                else:
                    Aij = matIter #????
                        
                Aq[0] =  Aij[0] * rotor[0] +  Aij[1] * rotor[1] +  Aij[2] * rotor[2] +  Aij[3] * rotor[3]
                Aq[1] =  Aij[4] * rotor[0] +  Aij[5] * rotor[1] +  Aij[6] * rotor[2] +  Aij[7] * rotor[3]
                Aq[2] =  Aij[8] * rotor[0] +  Aij[9] * rotor[1] + Aij[10] * rotor[2] + Aij[11] * rotor[3]
                Aq[3] = Aij[12] * rotor[0] + Aij[13] * rotor[1] + Aij[14] * rotor[2] + Aij[15] * rotor[3] 
                    
                qAq = rotor[0] * Aq[0] + rotor[1]*Aq[1] + rotor[2]*Aq[2] + rotor[3]*Aq[3]      
                    
                Vij = Aij[16] * np.exp( -qAq )
                    
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
                      
                        
            if iterations > 6 and atomOverlap < oldVolume + 0.0001:
                return
                
            oldVolume = atomOverlap
            
            if np.isnan(xlambda) or np.isnan(oldVolume) or oldVolume == 0:
                return
            
            if oldVolume > res.overlap:
                res.overlap = atomOverlap
                res.rotor = rotor
                
                if  res.overlap/(self.gRef.overlap + self.gDb.overlap - res.overlap)  > 0.99 :
                    
                    return
                    
            overHessian -= xlambda
            overHessian[1][0] = overHessian[0][1]
            overHessian[2][0] = overHessian[0][2]
            overHessian[2][1] = overHessian[1][2]
            overHessian[3][0] = overHessian[0][3]
            overHessian[3][1] = overHessian[1][3]
            overHessian[3][2] = overHessian[2][3]
            
            overGrad[0] -= xlambda * rotor[0]
            overGrad[1] -= xlambda * rotor[1]
            overGrad[2] -= xlambda * rotor[2]
            overGrad[3] -= xlambda * rotor[3]
            
            #line 354
            
            temp = overHessian.dot(overGrad)
            h = overGrad* temp
            
            h = 1/h*(sum(i**2 for i in overGrad))
            overGrad *= h
            
            rotor -= overGrad
            
            nr = np.sqrt(sum(i**2 for i in rotor))
            
            rotor /= nr
            
             
        
        return res
    
    def simulatedAnnealing(self):
        
        rotor = np.zeros(4)
        processQueue=[]
        
        Aij = 0
        Aq = np.zeros(4)
        
        Vij = 0
        qAq = 0
        
        d1 = []
        d2 = []
        
        dTemperature = 1.1
        
        res = AlignmentInfo()
        
        oldVolume = 0
        bestVolume = 0
        iterations = 0
        sameCount = 0
        mapIndex = 0
        
        while iterations < self._maxIter:
            
            atomOverlap = 0
            pharmOverlap = 0 
            iterations += 1
            
            T = np.sqrt((1.0 + iterations)/dTemperature)
            
            for i in self._rAtom:
                for j in self._dAtom:
                    
                    mapIndex = (i * self._dGauss) + j
                    
                    matIter = self._matrixMap[mapIndex]
                    if matIter == self._matrixMap[-1]:
                        
                        Aij = self._updateMatrixMap(self._gRef.gaussians[i], self._gDb.gaussians[j])
                        
                        self._matrixMap[mapIndex] = Aij
                        
                    else:
                        Aij = 0
                        
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
                                
                                
            for pair in processQueue:
                
                i = pair[0]
                j = pair[1]
                
                mapIndex = (i * self._dGauss) + j
                    
                matIter = self._matrixMap[mapIndex]
                if matIter == self._matrixMap[-1]:
                    
                    Aij = self._updateMatrixMap(self._gRef.gaussians[i], self._gDb.gaussians[j])
                    
                    self._matrixMap[mapIndex] = Aij
                    
                else:
                    Aij = 0
                    
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
                oldVolume = overlapVol
                oldRotor = rotor
                
                bestRotor = rotor
                bestVolume = overlapVol
                sameCount = 0
                
                if overlapVol > res.overlap:
                    res.overlap = atomOverlap
                    res.rotor = rotor
                    
                    if (res.overlap / (self._gRef.overlap + self._gDb.overlap - res.overlap )) > 0.99:
                        return
               
            
            ranPermutation = 0.1/T
            rotor = oldRotor - ranPermutation + 2* ranPermutation*np.random.random()
            
            nr = np.sqrt(sum(i**2 for i in rotor))
            rotor /= nr
            
        return res
    
    def setMaxIterations(self, i):
        
        self._maxIter = i
        return
                               
    def _updateMatrixMap(self, a= AtomGaussian, b = AtomGaussian):
        
        A = np.zeros(17)
        
        d = (a.center - b.center)
        s = (a.center + b.center)
        d = 0 #The distance squared between two gaussians
        d2 = sum(i**2 for i in d)
        s2 = sum(i**2 for i in s)
    
        C = a.alpha * b.alpha /(a.alpha + b.alpha)
        
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
        
        A = A*C
        
        if (a.n + b.n %2) == 0:
            A[16] = a.C * b.C * (np.pi/(a.alpha + b.alpha))**1.5
        else:
            A[16] = - a.C * b.C * (np.pi/(a.alpha + b.alpha))**1.5
            
        return A    
                    
                    
                    
                    
                
                
                    
                    
            
            
            