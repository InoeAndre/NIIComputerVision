# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 17:50:19 2017

@author: Inoe ANDRE
"""

import numpy as np
import math

PI = math.pi


class Stitch():
    
    def __init__(self, number_bodyPart):
        self.nb_bp = number_bodyPart
        self.StitchedVertices = 0
        self.StitchedFaces = 0

        
        
    def NaiveStitch(self, PartVtx,PartNmls,PartFaces,PoseBP):
        '''
        This function will just add the vertices and faces of each body parts 
        together after transforming them in the global coordinates system
        '''
        #Initialize values from the list of
        ConcatVtx = self.StitchedVertices
        ConcatFaces = self.StitchedFaces
        ConcatNmls = self.StitchedNormales
        # tranform the vertices in the global coordinates system
        PartVertices = self.TransformVtx(PartVtx,PoseBP,1)
        PartNormales = self.TransformNmls(PartNmls,PoseBP,1)
        PartFacets = PartFaces + np.max(ConcatFaces)+1
        self.StitchedVertices = np.concatenate((ConcatVtx,PartVertices))
        self.StitchedNormales = np.concatenate((ConcatNmls,PartNormales))
        self.StitchedFaces = np.concatenate((ConcatFaces,PartFacets))
        

        
    def TransformVtx(self, Vtx,Pose, s):
        """
        Transform the vertices in a system to another system.
        Here it will be mostly used to transform from local system to global coordiantes system
        """
        stack_pt = np.ones(np.size(Vtx,0), dtype = np.float32)
        pt = np.stack( (Vtx[ ::s,0],Vtx[ ::s,1],Vtx[ ::s,2],stack_pt),axis =1 )
        Vtx = np.dot(pt,Pose.T)
        return Vtx[:,0:3]
        
    def TransformNmls(self, Nmls,Pose, s):
        """
        Transform the normales in a system to another system.
        Here it will be mostly used to transform from local system to global coordiantes system
        """
        nmle = np.zeros((Nmls.shape[0], Nmls.shape[1]), dtype = np.float32)
        nmle[ ::s,:] = np.dot(Nmls[ ::s,:],Pose[0:3,0:3].T)
        return nmle[:,0:3]
        
    