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
import pandas as pd

    
def func_qAq(Aij, rotor):
    Aq = np.matmul(Aij, rotor)
    qAq = np.matmul(rotor, Aq)

    return Aq, qAq

def Grad(overHessian,overGrad):
                
        temp = np.matmul(overHessian,overGrad) #overHessian.dot(overGrad)
        h = np.matmul(overGrad, temp)#overGrad* temp
        #h = np.einsum('i,ij,j',overGrad,overHessian,overGrad)
        #small scaling of the gradient
        
        h = np.matmul(overGrad,overGrad)/h  
        overGrad *= h
        return overGrad

iu = np.triu_indices(4)
il2 = np.tril_indices(4, k=-1)

def overH(v2, Aq, Aij, overHessian):
                        
    Aq1 = Aq[:,None]
    outer = np.matmul(Aq1,Aq1.T)
    #R,C = np.triu_indices(N)
    #outer = np.einsum('i,j->ij',Aq,Aq) Takes more time as calculated the whole matrix
    overHessian[iu] += v2 * (2.0 * outer[iu]- Aij[iu])
    return
'''14.2 µs ± 822 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)'''

class ShapeAlignment(GaussianVolume):
    
    def __init__(self, gRef = GaussianVolume(),gDb = GaussianVolume()):
        
        self._gRef = gRef
        self._gDb = gDb
        self._rAtom = gRef.levels[0]
        self._rGauss = len(gRef.gaussians)
        self._dAtom = gDb.levels[0]
        self._dGauss = len(gDb.gaussians)
        self._maxSize = self._rGauss * self._dGauss + 1
        self._maxIter = 50
        #self._matrixMap = [[] for i in range(self._maxSize-1)]
        self._matrixMap = np.nan * np.ndarray([self._maxSize-1,4,4])
        #self._matrixMap = pd.Series(index = np.arange(self._maxSize-1),dtype = float)
        self._map16 = np.zeros(self._maxSize-1)
        
    def __del__(self): 
        self._gRef = None
        self._gDb = None
        # clear the matirxmap
        self._matrixMap = np.nan * np.ndarray([self._maxSize-1,4,4])
        
    def gradientAscent(self, rotor):
        
        EPS = 0.03
        processQueue = deque()
        
        #Aij = 0
        #Aq = np.zeros(4)
        
        #Vij = 0
        #qAq = 0
        
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
            
            for i in range(self._rAtom):
                for j in range(self._dAtom):
                    
                    mapIndex = (i * self._dGauss) + j
                    #Sub in the Aij to corresponding location, or calculate& sub in if empty initially   
                    if self._map16[mapIndex] == 0: #!!!
                        Aij,A16 = self._updateMatrixMap(self._gRef.gaussians[i], self._gDb.gaussians[j])
                        
                        self._matrixMap[mapIndex] = Aij
                        self._map16[mapIndex] = A16
                    else:           
                        Aij = self._matrixMap[mapIndex] 
                        A16 = self._map16[mapIndex]
                    '''
                    #Sub in the Aij to corresponding location, or calculate& sub in if empty initially   
                    if len(self._matrixMap[mapIndex]) == 0:
                        
                        Aij = self._updateMatrixMap(self._gRef.gaussians[i], self._gDb.gaussians[j])
                        
                        self._matrixMap[mapIndex,:] = Aij
                        
                    else:
                        Aij = self._matrixMap[mapIndex,:]
                    '''
                    
                    #Calculation of q′Aq, rotor product
                    Aq, qAq = func_qAq(Aij, rotor)
                   
                    #Volume overlap rewritten
                    Vij = A16 * np.exp( -qAq )                  
                    
                    if  Vij/(self._gRef.gaussians[i].volume + self._gDb.gaussians[j].volume - Vij ) <EPS: continue
                        
                    atomOverlap += Vij
                        
                    v2 = 2.0 * Vij # simply Vij doubled, for lateral use
                        
                    #Lagrange coefficient lambda
                    xlambda -= v2 * qAq
                        
                    #overGrad calculated, gradient minus since
                    overGrad -= v2 * Aq
                        
                    # overHessian += 2*Vij(2*Aijq'qAij-Aij); (only upper triangular part)
                    overH(v2,Aq, Aij, overHessian)
    
                    # loop over child nodes and add to queue
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
                #if np.isnan(self._matrixMap[mapIndex]):
                if self._map16[mapIndex] == 0:
                    
                    Aij,A16 = self._updateMatrixMap(a = self._gRef.gaussians[i], b = self._gDb.gaussians[j])
                    
                    self._matrixMap[mapIndex] = Aij
                    self._map16[mapIndex] = A16
                    
                else:
                    Aij = self._matrixMap[mapIndex]
                    A16 = self._map16[mapIndex]
                    
                Aq, qAq = func_qAq(Aij, rotor)

                #rotor product          
                Vij = A16 * np.exp( -qAq )

                if  abs(Vij)/(self._gRef.gaussians[i].volume + self._gDb.gaussians[j].volume - abs(Vij) ) < EPS: continue
                    
                atomOverlap += Vij
     
                v2 = 2.0 * Vij
                xlambda -= v2 * qAq
                
                overGrad -= v2 * Aq
                
                # overHessian += 2*Vij(2*Aijq'qAij-Aij); (only upper triangular part)
                overH(v2,Aq, Aij, overHessian)
                
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
            overHessian[il2] = overHessian.T[il2]
            '''
            overHessian[1][0] = overHessian[0][1]
            overHessian[2][0] = overHessian[0][2]
            overHessian[2][1] = overHessian[1][2]
            overHessian[3][0] = overHessian[0][3]
            overHessian[3][1] = overHessian[1][3]
            overHessian[3][2] = overHessian[2][3]
            '''
            #update gradient to make h
            overGrad -= xlambda * rotor
            
            #update gradient based on inverse hessian
            
            overGrad = Grad(overHessian,overGrad)
            
            # update rotor based on gradient information
            rotor -= overGrad
            
            #normalise rotor such that it has unit norm
            nr = np.sqrt(np.matmul(rotor,rotor))
            
            rotor /= nr
            
        return res
    
    def simulatedAnnealing(self, rotor):
        
        EPS = 0.03
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
            for i in range(self._rAtom):
                for j in range(self._dAtom):
                    
                    mapIndex = (i * self._dGauss) + j
                    
                    if self._map16[mapIndex] == 0:

                        Aij,A16 = self._updateMatrixMap(self._gRef.gaussians[i], self._gDb.gaussians[j])
                        
                        self._matrixMap[mapIndex] = Aij
                        self._map16[mapIndex] = A16
                    else:
                        Aij = self._matrixMap[mapIndex] 
                        A16 = self._map16[mapIndex] 
                        
                     #Calculation of q′Aq, rotor product
                    Aq, qAq = func_qAq(Aij, rotor)
                   
                    #Volume overlap rewritten
                    Vij = A16 * np.exp( -qAq )                  
 
                    
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
                if self._map16[mapIndex] == 0: #if np.isnan(self._matrixMap[mapIndex,0,0]):
                    
                    Aij,A16 = self._updateMatrixMap(a = self._gRef.gaussians[i], b = self._gDb.gaussians[j])
                    
                    self._matrixMap[mapIndex] = Aij
                    self._map16[mapIndex] = A16
                    
                else:
                    Aij = self._matrixMap[mapIndex]
                    A16 = self._map16[mapIndex]
                   
                        

                #Calculation of q′Aq, rotor product
                Aq, qAq = func_qAq(Aij, rotor)
                   
                #Volume overlap rewritten
                Vij = A16 * np.exp( -qAq )

                
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
            nr = np.sqrt(sum(rotor**2 ))
            rotor /= nr
            
        return res
    
    def setMaxIterations(self, i):
        
        self._maxIter = i
        return
    '''                           
    def _updateMatrixMap_old(self, a= AtomGaussian(), b = AtomGaussian()):
        
        #A is a matrix thatsolely depends on the position of the two centres of Gaussians a and b
        A = np.empty(17)
        
        dx, dy, dz = (a.centre - b.centre)
        sx, sy, sz = (a.centre + b.centre)
        dx2 = dx*dx
        dy2 = dy*dy
        dz2 = dz*dz
        sx2 = sx*sx
        sy2 = sy*sy
        sz2 = sz*sz
        
    
        weight = a.alpha * b.alpha /(a.alpha + b.alpha)
        
        A[0] = dx2 + dy2 + dz2
        A[1] = dy*sz - dz*sy
        A[8] = A[2] = dz*sx - dx*sz
        A[12] = A[3] = dx*sy - dy*sx
        A[4] = A[1]
        A[5] = dx2 + sy2 + sz2
        A[9] = A[6] = dx*dy - sx*sy
        A[13] = A[7] = dx*dz - sx*sz
       
        A[10] = sx2 + dy2 + sz2
        A[14] = A[11] = dy*dz - sy*sz
               
        A[15] = sx2 + sy2 + dz2
        
        A = A*weight
        
        if ((a.n + b.n) %2) == 0:
            A[16] = a.weight * b.weight * (np.pi/(a.alpha + b.alpha))**1.5      
        else:
            A[16] = - a.weight * b.weight * (np.pi/(a.alpha + b.alpha))**1.5
        return A    
                    
    '''

    def _updateMatrixMap(self, a= AtomGaussian(), b = AtomGaussian()):
            
            #A is a matrix thatsolely depends on the position of the two centres of Gaussians a and b
            A = np.ndarray([4,4])  
            
            dx, dy, dz = (a.centre - b.centre)
            sx, sy, sz = (a.centre + b.centre)
            dx2 = dx*dx
            dy2 = dy*dy
            dz2 = dz*dz
            sx2 = sx*sx
            sy2 = sy*sy
            sz2 = sz*sz
            
        
            weight = a.alpha * b.alpha /(a.alpha + b.alpha)
            
            A[0,0] = dx2 + dy2 + dz2
            A[1,0] = A[0,1] = dy*sz - dz*sy
            A[2,0] = A[0,2] = dz*sx - dx*sz
            A[3,0] = A[0,3] = dx*sy - dy*sx
         
            A[1,1] = dx2 + sy2 + sz2
            A[2,1] = A[1,2] = dx*dy - sx*sy
            A[3,1] = A[1,3] = dx*dz - sx*sz
           
            A[2,2] = sx2 + dy2 + sz2
            A[3,2] = A[2,3] = dy*dz - sy*sz             
            A[3,3] = sx2 + sy2 + dz2
            
            A *=weight
            
            scaling_C = 0
            if ((a.n + b.n) %2) == 0:
                scaling_C = a.weight * b.weight * (np.pi/(a.alpha + b.alpha))**1.5      
            else:
                scaling_C = - a.weight * b.weight * (np.pi/(a.alpha + b.alpha))**1.5
            return A ,scaling_C                  
                    
                    
def f(i,j):
    return i,j             
                
                    
                    
            
            
            