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

        
        
    def NaiveStitch(self, PartVtx,PartFaces,PoseBP):
        '''
        This function will just add the vertices and faces of each body parts 
        together after transforming them in the global coordinates system
        '''
        #Initialize values from the list of
        ConcatVtx = self.StitchedVertices
        ConcatFaces = self.StitchedFaces
        # adapt faces so that the global list still gives the corrects indexs
        #PartFaces = self.TransfoFaces(PartFaces)
        # tranform the vertices in the global coordinates system
        PartVertices = self.TransformVert(PartVtx,PoseBP,1)
        PartFacets = PartFaces + np.max(self.StitchedFaces)+1
        self.StitchedVertices = np.concatenate((ConcatVtx,PartVertices))
        self.StitchedFaces = np.concatenate((ConcatFaces,PartFacets))
        

        
    def TransformVert(self, Vtx,Pose, s):
        stack_pt = np.ones(np.size(Vtx,0), dtype = np.float32)
        pt = np.stack( (Vtx[ ::s,0],Vtx[ ::s,1],Vtx[ ::s,2],stack_pt),axis =1 )
        Vtx = np.dot(pt,Pose.T)
        return Vtx[:,0:3]
        

    '''
    This function adapt the index in Faces to be able to have coherent concatenation
    '''
    def TransfoFaces(self,PartFaces):
        VtxArray_x = np.zeros(self.Size)
        VtxArray_y = np.zeros(self.Size)
        VtxArray_z = np.zeros(self.Size)
        VtxWeights = np.zeros(self.Size)
        VtxIdx = np.zeros(self.Size)
        FacesIdx = np.zeros((self.nb_faces[0], 3, 3), dtype = np.uint16)
        self.Normales = np.zeros((3*self.nb_faces[0], 3), dtype = np.float32)
        # Go through all faces
        pt = np.array([0., 0., 0., 1.])
        self.nbVtx = 0
        for i in range(self.nb_faces[0]):
            for k in range(3):
                pt[0] = self.Vertices[self.Faces[i,k],0]
                pt[1] = self.Vertices[self.Faces[i,k],1]
                pt[2] = self.Vertices[self.Faces[i,k],2]
                indx = (int(round(pt[0]*self.res[1]+self.res[0])), 
                        int(round(pt[1]*self.res[3]+self.res[2])), 
                        int(round(pt[2]*self.res[5]+self.res[4])))
                if (VtxWeights[indx] == 0):
                    self.nbVtx += 1
                VtxArray_x[indx] = (VtxWeights[indx]*VtxArray_x[indx] + pt[0])/(VtxWeights[indx]+1)
                VtxArray_y[indx] = (VtxWeights[indx]*VtxArray_y[indx] + pt[1])/(VtxWeights[indx]+1)
                VtxArray_z[indx] = (VtxWeights[indx]*VtxArray_z[indx] + pt[2])/(VtxWeights[indx]+1)
                VtxWeights[indx] = VtxWeights[indx]+1
                FacesIdx[i,k,0] = indx[0]
                FacesIdx[i,k,1] = indx[1]
                FacesIdx[i,k,2] = indx[2]
                       
        print "nb vertices: ", self.nbVtx
        self.nb_vertices = np.array([self.nbVtx], dtype = np.int32)
        self.Vertices = np.zeros((self.nbVtx, 3), dtype = np.float32)
        self.Normales = np.zeros((self.nbVtx, 3), dtype = np.float32)
        index_count = 0
        for i in range(self.Size[0]):
            #print i
            for j in range(self.Size[1]):
                for k in range(self.Size[2]):
                    if (VtxWeights[i,j,k] > 0):
                        VtxIdx[i,j,k] = index_count
                        self.Vertices[index_count, 0] = VtxArray_x[i,j,k]
                        self.Vertices[index_count, 1] = VtxArray_y[i,j,k]
                        self.Vertices[index_count, 2] = VtxArray_z[i,j,k]
                        index_count += 1 
        
        for i in range(self.nb_faces[0]):
            v = np.zeros((3, 3), dtype = np.float32)
            for k in range(3):
                self.Faces[i,k] = VtxIdx[(FacesIdx[i,k,0],FacesIdx[i,k,1],FacesIdx[i,k,2])]
                v[k,:] = self.Vertices[self.Faces[i,k],:]
            
            v1 = v[1,:] - v[0,:]
            v2 = v[2,:] - v[0,:]
            nmle = [v1[1]*v2[2] - v1[2]*v2[1],
                    -v1[0]*v2[2] + v1[2]*v2[0],
                    v1[0]*v2[1] - v1[1]*v2[0]]
            
            for k in range(3):
                self.Normales[self.Faces[i,k], :] = self.Normales[self.Faces[i,k],:] + nmle
        
        
        for i in range(self.nb_vertices[0]):
            mag = math.sqrt(self.Normales[i, 0]**2 + self.Normales[i, 1]**2 + self.Normales[i, 2]**2)
            if mag == 0.0:
                self.Normales[i, :] = 0.0
            else :
                self.Normales[i, :] = self.Normales[i, :]/mag