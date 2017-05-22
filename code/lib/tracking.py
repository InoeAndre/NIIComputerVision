#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:47:40 2017

@author: diegothomas
"""

import imp
import numpy as np
from numpy import linalg as LA
from math import sin, cos, acos
import scipy as sp
import pandas 

RGBD = imp.load_source('RGBD', './lib/RGBD.py')

def in_mat_zero2one(mat):
    """This fonction replace in the matrix all the 0 to 1"""
    mat_tmp = (mat != 0.0)
    res = mat * mat_tmp + ~mat_tmp
    return res

def Exponential(qsi):
    theta = LA.norm(qsi[3:6])
    res = np.identity(4)
    
    if (theta != 0.):
        res[0,0] = 1.0 + sin(theta)/theta*0.0 + (1.0 - cos(theta)) / (theta*theta) * (-qsi[5]*qsi[5] - qsi[4]*qsi[4])
        res[1,0] = 0.0 + sin(theta)/theta*qsi[5] + (1.0 - cos(theta))/(theta*theta) * (qsi[3]*qsi[4])
        res[2,0] = 0.0 - sin(theta)/theta*qsi[4] + (1.0 - cos(theta))/(theta*theta) * (qsi[3]*qsi[5])
        
        res[0,1] = 0.0 - sin(theta)/theta*qsi[5] + (1.0 - cos(theta))/(theta*theta) * (qsi[3]*qsi[4])
        res[1,1] = 1.0 + sin(theta) / theta*0.0 + (1.0 - cos(theta))/(theta*theta) * (-qsi[5]*qsi[5] - qsi[3]*qsi[3])
        res[2,1] = 0.0 + sin(theta)/theta*qsi[3] + (1.0 - cos(theta))/(theta*theta) * (qsi[4]*qsi[5])
        
        res[0,2] = 0.0 + sin(theta) / theta*qsi[4] + (1.0 - cos(theta))/(theta*theta) * (qsi[3]*qsi[5])
        res[1,2] = 0.0 - sin(theta)/theta*qsi[3] + (1.0 - cos(theta))/(theta*theta) * (qsi[4]*qsi[5])
        res[2,2] = 1.0 + sin(theta)/theta*0.0 + (1.0 - cos(theta))/(theta*theta) * (-qsi[4]*qsi[4] - qsi[3]*qsi[3])
        
        skew = np.zeros((3,3), np.float32)
        skew[0,1] = -qsi[5]
        skew[0,2] = qsi[4]
        skew[1,0] = qsi[5]
        skew[1,2] = -qsi[3]
        skew[2,0] = -qsi[4]
        skew[2,1] = qsi[3]
        
        V = np.identity(3) + ((1.0-cos(theta))/(theta*theta))*skew + ((theta - sin(theta))/(theta*theta))*np.dot(skew,skew)
        
        res[0,3] = V[0,0]*qsi[0] + V[0,1]*qsi[1] + V[0,2]*qsi[2]
        res[1,3] = V[1,0]*qsi[0] + V[1,1]*qsi[1] + V[1,2]*qsi[2]
        res[2,3] = V[2,0]*qsi[0] + V[2,1]*qsi[1] + V[2,2]*qsi[2]
    else:
        res[0,3] = qsi[0]
        res[1,3] = qsi[1]
        res[2,3] = qsi[2]
        
    return res

def Logarithm(Mat):
    trace = Mat[0,0]+Mat[1,1]+Mat[2,2]
    theta = acos((trace-1.0)/2.0)
    
    qsi = np.array([0.,0.,0.,0.,0.,0.])
    if (theta == 0.):
        qsi[3] = qsi[4] = qsi[5] = 0.0
        qsi[0] = Mat[0,3]
        qsi[1] = Mat[1,3]
        qsi[2] = Mat[2,3]
        return qsi
    
    R = Mat[0:3,0:3]
    lnR = (theta/(2.0*sin(theta))) * (R-np.transpose(R))
    
    qsi[3] = (lnR[2,1] - lnR[1,2])/2.0
    qsi[4] = (lnR[0,2] - lnR[2,0])/2.0
    qsi[5] = (lnR[1,0] - lnR[0,1])/2.0
    
    theta = LA.norm(qsi[3:6])

    skew = np.zeros((3,3), np.float32)
    skew[0,1] = -qsi[5]
    skew[0,2] = qsi[4]
    skew[1,0] = qsi[5]
    skew[1,2] = -qsi[3]
    skew[2,0] = -qsi[4]
    skew[2,1] = qsi[3]
    
    V = np.identity(3) + ((1.0 - cos(theta))/(theta*theta))*skew + ((theta-sin(theta))/(theta*theta))*np.dot(skew,skew)
    V_inv = LA.inv(V)
    
    qsi[0] = V_inv[0,0]*Mat[0,3] + V_inv[0,1]*Mat[1,3] + V_inv[0,2]*Mat[2,3]
    qsi[1] = V_inv[1,0]*Mat[0,3] + V_inv[1,1]*Mat[1,3] + V_inv[1,2]*Mat[2,3]
    qsi[2] = V_inv[2,0]*Mat[0,3] + V_inv[2,1]*Mat[1,3] + V_inv[2,2]*Mat[2,3]
    
    return qsi
    

class Tracker():

    # Constructor
    def __init__(self, thresh_dist, thresh_norm, lvl, max_iter, thresh_conv):
        self.thresh_dist = thresh_dist
        self.thresh_norm = thresh_norm
        self.lvl = lvl
        self.max_iter = max_iter
        self.thresh_conv = thresh_conv
        
        
    """
    Function that estimate the relative rigid transformation between two input RGB-D images
    """
    def RegisterRGBD(self, Image1, Image2):
        
        res = np.identity(4)
        
        line_index = 0
        column_index = 0
        pix = np.array([0., 0., 1.])
        
        pt = np.array([0., 0., 0., 1.])
        nmle = np.array([0., 0., 0.])
        for l in range(1,self.lvl+1):
            for it in range(self.max_iter[l-1]):
                nbMatches = 0
                row = np.array([0.,0.,0.,0.,0.,0.,0.])
                Mat = np.zeros(27, np.float32)
                b = np.zeros(6, np.float32)
                A = np.zeros((6,6), np.float32)
                
                # For each pixel find correspondinng point by projection
                for i in range(Image1.Size[0]/l): # line index (i.e. vertical y axis)
                    for j in range(Image1.Size[1]/l):
                        # Transform current 3D position and normal with current transformation
                        pt[0:3] = Image1.Vtx[i*l,j*l][:]
                        if (LA.norm(pt[0:3]) < 0.1):
                            continue
                        pt = np.dot(res, pt)
                        nmle[0:3] = Image1.Nmls[i*l,j*l][0:3]
                        if (LA.norm(nmle) == 0.):
                            continue
                        nmle = np.dot(res[0:3,0:3], nmle)
                        
                        # Project onto Image2
                        pix[0] = pt[0]/pt[2]
                        pix[1] = pt[1]/pt[2]
                        pix = np.dot(Image2.intrinsic, pix)
                        column_index = int(round(pix[0]))
                        line_index = int(round(pix[1]))
                        
                        if (column_index < 0 or column_index > Image2.Size[1]-1 or line_index < 0 or line_index > Image2.Size[0]-1):
                            continue
                        
                        # Compute distance betwn matches and btwn normals
                        match_vtx = Image2.Vtx[line_index, column_index]
                        distance = LA.norm(pt[0:3] - match_vtx)
                        print "[line,column] : [%d , %d] " %(line_index, column_index)
                        print "match_vtx"
                        print match_vtx
                        print pt[0:3]
                        if (distance > self.thresh_dist):
                            continue
                        
                        match_nmle = Image2.Nmls[line_index, column_index]
                        distance = LA.norm(nmle - match_nmle)
                        print "match_nmle"
                        print match_nmle
                        print nmle                      
                        if (distance > self.thresh_norm):
                            continue
                            
                        w = 1.0
                        # Complete Jacobian matrix
                        row[0] = w*nmle[0]
                        row[1] = w*nmle[1]
                        row[2] = w*nmle[2]
                        row[3] = w*(-match_vtx[2]*nmle[1] + match_vtx[1]*nmle[2])
                        row[4] = w*(match_vtx[2]*nmle[0] - match_vtx[0]*nmle[2])
                        row[5] = w*(-match_vtx[1]*nmle[0] + match_vtx[0]*nmle[1])
                        row[6] = w*(nmle[0]*(match_vtx[0] - pt[0]) + nmle[1]*(match_vtx[1] - pt[1]) + nmle[2]*(match_vtx[2] - pt[2]))
                                    
                        nbMatches+=1
                            
                        shift = 0
                        for k in range(6):
                            for k2 in range(k,7):
                                Mat[shift] = Mat[shift] + row[k]*row[k2]
                                shift+=1
               
                print ("nbMatches: ", nbMatches)             
                shift = 0
                for k in range(6):
                    for k2 in range(k,7):
                        val = Mat[shift]
                        shift +=1
                        if (k2 == 6):
                            b[k] = val
                        else:
                            A[k,k2] = A[k2,k] = val
                
                det = LA.det(A)
                if (det < 1.0e-10):
                    print "determinant null"
                    break
        
        
                delta_qsi = -LA.tensorsolve(A, b)
                delta_transfo = LA.inv(Exponential(delta_qsi))
                
                res = np.dot(delta_transfo, res)
                
                print res
        return res
                
    def RegisterRGBD_optimize(self, Image1, Image2):
        
        res = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        
        column_index_ref = np.array([np.array(range(Image1.Size[1])) for _ in range(Image1.Size[0])])
        line_index_ref = np.array([x*np.ones(Image1.Size[1], np.int) for x in range(Image1.Size[0])])
        Indexes_ref = column_index_ref + Image1.Size[1]*line_index_ref
        
        for l in range(1,self.lvl+1):
            for it in range(self.max_iter[l-1]):
                #nbMatches = 0
                #row = np.array([0.,0.,0.,0.,0.,0.,0.])
                #Mat = np.zeros(27, np.float32)
                b = np.zeros(6, np.float32)
                A = np.zeros((6,6), np.float32)
                
                # For each pixel find correspondinng point by projection
                Buffer = np.zeros((Image1.Size[0]*Image1.Size[1], 6), dtype = np.float32)
                Buffer_B = np.zeros((Image1.Size[0]*Image1.Size[1], 1), dtype = np.float32)
                stack_pix = np.ones((Image1.Size[0], Image1.Size[1]), dtype = np.float32)
                stack_pt = np.ones((np.size(Image1.Vtx[ ::l, ::l,:],0), np.size(Image1.Vtx[ ::l, ::l,:],1)), dtype = np.float32)
                pix = np.zeros((Image1.Size[0], Image1.Size[1],2), dtype = np.float32)
                pix = np.dstack((pix,stack_pix))
                pt = np.dstack((Image1.Vtx[ ::l, ::l, :],stack_pt))
                pt = np.dot(res,pt.transpose(0,2,1)).transpose(1,2,0)
#==============================================================================
#                 print "pt"
#                 print pt                
#==============================================================================
                nmle = np.zeros((Image1.Size[0], Image1.Size[1],Image1.Size[2]), dtype = np.float32)
                nmle[ ::l, ::l,:] = np.dot(res[0:3,0:3],Image1.Nmls[ ::l, ::l,:].transpose(0,2,1)).transpose(1,2,0)
#==============================================================================
#                 print "nmle"
#                 print nmle
#==============================================================================
                #if (pt[2] != 0.0):
                lpt = np.dsplit(pt,4)
                lpt[2] = in_mat_zero2one(lpt[2])
                
                # if in 1D pix[0] = pt[0]/pt[2]
                pix[ ::l, ::l,0] = (lpt[0]/lpt[2]).reshape(np.size(Image1.Vtx[ ::l, ::l,:],0), np.size(Image1.Vtx[ ::l, ::l,:],1))
                # if in 1D pix[1] = pt[1]/pt[2]
                pix[ ::l, ::l,1] = (lpt[1]/lpt[2]).reshape(np.size(Image1.Vtx[ ::l, ::l,:],0), np.size(Image1.Vtx[ ::l, ::l,:],1))
                pix = np.dot(Image1.intrinsic,pix[0:Image1.Size[0],0:Image1.Size[1]].transpose(0,2,1)).transpose(1,2,0)
                column_index = (np.round(pix[:,:,0])).astype(int)
                line_index = (np.round(pix[:,:,1])).astype(int)
#==============================================================================
#                 print "pix"
#                 print pix
#==============================================================================
                print "line_index"
                print line_index 
                print "column_index"
                print column_index                 
                # create matrix that have 0 when the conditions are not verified and 1 otherwise
                cdt_column = (column_index > -1) * (column_index < Image2.Size[1])
                cdt_line = (line_index > -1) * (line_index < Image2.Size[0])
                line_index = line_index*cdt_line
                column_index = column_index*cdt_column
                #Indexes = line_index + Image2.Size[0]*column_index
                Indexes = column_index + Image2.Size[1]*line_index
                
                diff_Vtx = Image2.Vtx[line_index[:][:], column_index[:][:]] - pt[:,:,0:3]
                diff_Vtx = diff_Vtx*diff_Vtx
                norm_diff_Vtx = diff_Vtx.sum(axis=2)
                
                diff_Nmle = Image2.Nmls[line_index[:][:], column_index[:][:]] - nmle
                diff_Nmle = diff_Nmle*diff_Nmle
                norm_diff_Nmle = diff_Nmle.sum(axis=2)
                
                Norme_Nmle = nmle*nmle
                norm_Norme_Nmle = Norme_Nmle.sum(axis=2)
                
                mask = cdt_line*cdt_column * (pt[:,:,2] > 0.0) * (norm_Norme_Nmle > 0.0) * (norm_diff_Vtx < self.thresh_dist) * (norm_diff_Nmle < self.thresh_norm)
                print sum(sum(mask))
                
                w = 1.0
                Buffer[Indexes_ref[:][:]] = np.dstack((w*mask[:,:]*nmle[ :, :,0], \
                      w*mask[:,:]*nmle[ :, :,1], \
                      w*mask[:,:]*nmle[ :, :,2], \
                      w*mask[:,:]*(-Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,2]*nmle[:,:,1] + Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,1]*nmle[:,:,2]), \
                      w*mask[:,:]*(Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,2]*nmle[:,:,0] - Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,0]*nmle[:,:,2]), \
                      w*mask[:,:]*(-Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,1]*nmle[:,:,0] + Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,0]*nmle[:,:,1]) ))
                
                Buffer_B[Indexes_ref[:][:]] = np.dstack(w*mask[:,:]*(nmle[:,:,0]*(Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,0] - pt[:,:,0]) + nmle[:,:,1]*(Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,1] - pt[:,:,1]) + nmle[:,:,2]*(Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,2] - pt[:,:,2])) ).transpose()
                        
                A = np.dot(Buffer.transpose(), Buffer)
                b = np.dot(Buffer.transpose(), Buffer_B).reshape(6)
                
                det = LA.det(A)
                if (det < 1.0e-10):
                    print "determinant null"
                    break
           
                delta_qsi = -LA.tensorsolve(A, b)
                delta_transfo = LA.inv(Exponential(delta_qsi))
                
                res = np.dot(delta_transfo, res)
                
                print res
        return res
    
            
    def RegisterRGBDMesh(self, NewImage, MeshVtx, MeshNmls,Pose):            
        res = np.identity(4)
        
        MeshVtx = np.dot(MeshVtx,Pose[0:3,0:3].T)
        MeshNmls = np.dot(MeshNmls,Pose[0:3,0:3].T)
        
        print
        
        line_index = 0
        column_index = 0
        pix = np.array([0., 0., 1.])
        
        pt = np.array([0., 0., 0., 1.])
        nmle = np.array([0., 0., 0.])
        for l in range(1,self.lvl+1):
            for it in range(self.max_iter[l-1]):
                nbMatches = 0
                row = np.array([0.,0.,0.,0.,0.,0.,0.])
                Mat = np.zeros(27, np.float32)
                b = np.zeros(6, np.float32)
                A = np.zeros((6,6), np.float32)
                
                # For each pixel find correspondinng point by projection
                for i in range(MeshVtx.shape[0]): # line index (i.e. vertical y axis)
                    # Transform current 3D position and normal with current transformation
                    pt[0:3] = MeshVtx[i][:]
                    if (LA.norm(pt[0:3]) < 0.1):
                        continue
                    pt = np.dot(res, pt)
                    nmle[0:3] = MeshNmls[i][0:3]
                    if (LA.norm(nmle) == 0.):
                        continue
                    nmle = np.dot(res[0:3,0:3], nmle)
                    
                    # Project onto Image2
                    pix[0] = pt[0]/pt[2]
                    pix[1] = pt[1]/pt[2]
                    pix = np.dot(NewImage.intrinsic, pix)
                    column_index = int(round(pix[0]))
                    line_index = int(round(pix[1]))
                    
                    if (column_index % l != 0) or (line_index % l != 0):
                        continue
                    
                    if (column_index < 0 or column_index > NewImage.Size[1]-1 or line_index < 0 or line_index > NewImage.Size[0]-1):
                        continue
                    
                    # Compute distance betwn matches and btwn normals
                    match_vtx = pt[0:3]
                    pt[0:3] = NewImage.Vtx[line_index, column_index]
                    #print "[line,column] : [%d , %d] " %(line_index, column_index)
                    #print "match_vtx"
                    #print match_vtx
                    #print pt[0:3]
                    distance = LA.norm(pt[0:3] - match_vtx)
                    if (distance > self.thresh_dist):
                        #print "no Vtx correspondance"
                        #print distance
                        continue
                    match_nmle = nmle
                    nmle = NewImage.Nmls[line_index, column_index]
                    #print "match_nmle"
                    #print match_nmle
                    #print nmle
                    distance = LA.norm(nmle - match_nmle)
                    if (distance > self.thresh_norm):
                        #print "no Nmls correspondance"
                        #print distance
                        continue
                        
                    w = 1.0
                    # Complete Jacobian matrix
                    row[0] = w*nmle[0]
                    row[1] = w*nmle[1]
                    row[2] = w*nmle[2]
                    row[3] = w*(-match_vtx[2]*nmle[1] + match_vtx[1]*nmle[2])
                    row[4] = w*(match_vtx[2]*nmle[0] - match_vtx[0]*nmle[2])
                    row[5] = w*(-match_vtx[1]*nmle[0] + match_vtx[0]*nmle[1])
                    row[6] = w*(nmle[0]*(match_vtx[0] - pt[0]) + nmle[1]*(match_vtx[1] - pt[1]) + nmle[2]*(match_vtx[2] - pt[2]))
                                
                    nbMatches+=1
                        
                    shift = 0
                    for k in range(6):
                        for k2 in range(k,7):
                            Mat[shift] = Mat[shift] + row[k]*row[k2]
                            shift+=1
               
                print ("nbMatches: ", nbMatches)             
                shift = 0
                for k in range(6):
                    for k2 in range(k,7):
                        val = Mat[shift]
                        shift +=1
                        if (k2 == 6):
                            b[k] = val
                        else:
                            A[k,k2] = A[k2,k] = val
                
                det = LA.det(A)
                if (det < 1.0e-10):
                    print "determinant null"
                    break
        
        
                delta_qsi = -LA.tensorsolve(A, b)
                delta_transfo = LA.inv(Exponential(delta_qsi))
                
                res = np.dot(delta_transfo, res)
                
                print res
        return res
    
    
    
    
    def RegisterRGBDMesh_optimize(self, NewImage, MeshVtx, MeshNmls,Pose):
        res = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        
        column_index_ref = np.array([np.array(range(NewImage.Size[1])) for _ in range(NewImage.Size[0])])
        line_index_ref = np.array([x*np.ones(NewImage.Size[1], np.int) for x in range(NewImage.Size[0])])
        Indexes_ref = column_index_ref + NewImage.Size[1]*line_index_ref  
        
#==============================================================================
#         MeshVtx = pandas.DataFrame(MeshVtx).drop_duplicates().values
#         MeshNmls = pandas.DataFrame(MeshNmls).drop_duplicates().values
#         
#         print "MeshVtx.shape"
#         print MeshVtx.shape
#         print "MeshNmls.shape"
#         print MeshNmls.shape
#==============================================================================

        MeshVtx = np.dot(MeshVtx,Pose[0:3,0:3].T)
        MeshNmls = np.dot(MeshNmls,Pose[0:3,0:3].T)  
        for l in range(1,self.lvl+1):
            for it in range(self.max_iter[l-1]):
                #nbMatches = 0
                #row = np.array([0.,0.,0.,0.,0.,0.,0.])
                #Mat = np.zeros(27, np.float32)
                b = np.zeros(6, np.float32)
                A = np.zeros((6,6), np.float32)
                
                # For each pixel find correspondinng point by projection
                Buffer = np.zeros((NewImage.Size[0]*NewImage.Size[1], 6), dtype = np.float32)
                Buffer_B = np.zeros((NewImage.Size[0]*NewImage.Size[1], 1), dtype = np.float32)
                stack_pix = np.ones((NewImage.Size[0], NewImage.Size[1]), dtype = np.float32)
                stack_pt = np.ones((np.size(NewImage.Vtx[ ::l, ::l,:],0), np.size(NewImage.Vtx[ ::l, ::l,:],1)), dtype = np.float32)
                pix = np.zeros((NewImage.Size[0], NewImage.Size[1],2), dtype = np.float32)
                pix = np.dstack((pix,stack_pix))
                
                #print "newImage Vtx"
                #print NewImage.Vtx[ ::l, ::l, :]
                
                pt = np.dstack((NewImage.Vtx[ ::l, ::l, :],stack_pt))
                pt = np.dot(res,pt.transpose(0,2,1)).transpose(1,2,0)
                #print "newImage pt Vtx"
                #print pt
                nmle = np.zeros((NewImage.Size[0], NewImage.Size[1],NewImage.Size[2]), dtype = np.float32)
                nmle[ ::l, ::l,:] = np.dot(res[0:3,0:3],NewImage.Nmls[ ::l, ::l,:].transpose(0,2,1)).transpose(1,2,0)
                

                lpt = np.dsplit(pt,4)
                lpt[2] = in_mat_zero2one(lpt[2])
                
                # compute the indexes in the new images to reshape the lists
                pix[ ::l, ::l,0] = (lpt[0]/lpt[2]).reshape(np.size(NewImage.Vtx[ ::l, ::l,:],0), np.size(NewImage.Vtx[ ::l, ::l,:],1))
                pix[ ::l, ::l,1] = (lpt[1]/lpt[2]).reshape(np.size(NewImage.Vtx[ ::l, ::l,:],0), np.size(NewImage.Vtx[ ::l, ::l,:],1))
                pix = np.dot(NewImage.intrinsic,pix[0:NewImage.Size[0],0:NewImage.Size[1]].transpose(0,2,1)).transpose(1,2,0)
                column_index = (np.round(pix[:,:,0])).astype(int)
                line_index = (np.round(pix[:,:,1])).astype(int)
                
                ###############
                # Meshes
                ###############
                
                
                Indexes = np.stack( (line_index,column_index),axis =2)
                cdtIdx = (Indexes > -1) * (Indexes < MeshVtx.shape[0])
                print "Indexes"
                print Indexes 
                print Indexes.shape
                # create matrix that have 0 when the conditions are not verified and 1 otherwise for theenewImage
                cdt_column = (column_index > -1) * (column_index < NewImage.Size[1])
                cdt_line = (line_index > -1) * (line_index < NewImage.Size[0])
                line_index = line_index*cdt_line
                column_index = column_index*cdt_column  
                

                Indexes *= cdtIdx
                print "Indexes"
                print Indexes
                print Indexes.shape
                diff_Vtx = np.zeros((NewImage.Size[0],NewImage.Size[1], 3), dtype = np.float)
                MeshVtxResh = MeshVtx[Indexes[:,:,0]*NewImage.Size[1]+Indexes[:,:,1]].reshape(NewImage.Size[0],NewImage.Size[1], 3)
                MeshNmlsResh = MeshNmls[Indexes[:,:,0]*NewImage.Size[1]+Indexes[:,:,1]].reshape(NewImage.Size[0],NewImage.Size[1], 3)
                print "MeshVtxResh"
                print MeshVtxResh
                print "MeshNmlsResh"
                print MeshNmlsResh               
                
                # compute the indexes from the list to have the correspondence
                MeshStack_pix = np.ones( (MeshVtxResh.shape[0],MeshVtxResh.shape[1]), dtype = np.float32)
                MeshStack_pt = np.ones((MeshVtxResh.shape[0],MeshVtxResh.shape[1]), dtype = np.float32)
                MeshPix = np.zeros((MeshVtxResh.shape[0],MeshVtxResh.shape[1], 2), dtype = np.float32)
                MeshPix = np.stack((MeshPix[:,:,0],MeshPix[:,:,1],MeshStack_pix), axis = 2)
                MeshPt = np.stack((MeshVtxResh[ ::l, ::l, 0],MeshVtxResh[ ::l, ::l, 1],MeshVtxResh[ ::l, ::l, 2],MeshStack_pt),axis = 2)
                
                MeshLpt = np.split(MeshPt,4,axis=1)
                MeshLpt[2] = in_mat_zero2one(MeshLpt[2])
                
                # if in 1D pix[0] = pt[0]/pt[2]
                MeshPix[  ::l, ::l,0] = (MeshLpt[0]/MeshLpt[2]).reshape(np.size(MeshVtxResh[  ::l, ::l,:],0),np.size(MeshVtxResh[  ::l, ::l,:],1))
                # if in 1D pix[1] = pt[1]/pt[2]
                MeshPix[  ::l, ::l,1] = (MeshLpt[1]/MeshLpt[2]).reshape(np.size(MeshVtxResh[  ::l, ::l,:],0),np.size(MeshVtxResh[  ::l, ::l,:],1))
                #MeshPix = np.dot(NewImage.intrinsic,MeshPix[0:MeshVtxResh.shape[0],0:MeshVtxResh.shape[1]].transpose(0,2,1)).transpose(1,2,0)  

                
                column_indexMesh = (np.round(MeshPix[:,:,0])).astype(int)
                line_indexMesh = (np.round(MeshPix[:,:,1])).astype(int)                  
                # create matrix that have 0 when the conditions are not verified and 1 otherwise for the newImage
                cdt_column_indexMesh = (column_indexMesh > -1) * (column_indexMesh < NewImage.Size[1])
                cdt_line_indexMesh = (line_indexMesh > -1) * (line_indexMesh < NewImage.Size[0])
                line_indexMesh = line_indexMesh*cdt_line_indexMesh
                column_indexMesh = column_indexMesh*cdt_column_indexMesh                 
                
                
                print "pt.shape"
                print pt
                print "line_indexMesh[:][:]"
                print line_indexMesh[:][:]
                print "column_indexMesh[:][:]"
                print column_indexMesh[:][:]
                #print pt[line_indexMesh[:][:],column_indexMesh[:][:]].shape
                diff_Vtx[:,:,0:3] = pt[line_indexMesh[:][:],column_indexMesh[:][:]][:,:,0:3] - MeshVtxResh[:,:,0:3]  
                #diff_Vtx = diff_Vtx.reshape(NewImage.Size[0], NewImage.Size[1], 3)
                diff_Vtx = diff_Vtx*diff_Vtx
                norm_diff_Vtx = np.sqrt(diff_Vtx.sum(axis=2))
                

                diff_Nmle = np.zeros((NewImage.Size[0],NewImage.Size[1], 3), dtype = np.float)
                diff_Nmle[:,:,0:3] = MeshNmlsResh[:,:,0:3] - nmle[line_indexMesh[:][:],column_indexMesh[:][:]]
                #diff_Nmle = diff_Nmle.reshape(NewImage.Size[0], NewImage.Size[1], 3)
                diff_Nmle = diff_Nmle*diff_Nmle
                norm_diff_Nmle = np.sqrt(diff_Nmle.sum(axis=2))
                
                Norme_Nmle = nmle*nmle
                norm_Norme_Nmle = Norme_Nmle.sum(axis=2)
                
                mask = cdt_line*cdt_column * (pt[line_indexMesh[:][:],column_indexMesh[:][:]][:,:,2] > 0.0) * (norm_Norme_Nmle > 0.0) * (norm_diff_Vtx < self.thresh_dist) * (norm_diff_Nmle < self.thresh_norm)
                print sum(sum(mask))
                
                
                w = 1.0
                Buffer[Indexes_ref[:][:]] = np.dstack((w*mask[:,:]*nmle[ :, :,0], \
                      w*mask[:,:]*nmle[line_indexMesh[:][:],column_indexMesh[:][:]][:,:,1], \
                      w*mask[:,:]*nmle[ line_indexMesh[:][:],column_indexMesh[:][:]][:,:,2], \
                      w*mask[:,:]*(-MeshVtxResh[:,:,2]*nmle[line_indexMesh[:][:],column_indexMesh[:][:]][:,:,1] + MeshVtxResh[:,:,1]*nmle[line_indexMesh[:][:],column_indexMesh[:][:]][:,:,2]), \
                      w*mask[:,:]*(MeshVtxResh[:,:,2]*nmle[line_indexMesh[:][:],column_indexMesh[:][:]][:,:,0] - MeshVtxResh[:,:,1]*nmle[line_indexMesh[:][:],column_indexMesh[:][:]][:,:,2]), \
                      w*mask[:,:]*(-MeshVtxResh[:,:,2]*nmle[line_indexMesh[:][:],column_indexMesh[:][:]][:,:,0] + MeshVtxResh[:,:,1]*nmle[line_indexMesh[:][:],column_indexMesh[:][:]][:,:,1]) ))
                
                Buffer_B[Indexes_ref[:][:]] = np.dstack(w*mask[:,:]*(nmle[line_indexMesh[:][:],column_indexMesh[:][:]][:,:,0]*(MeshVtxResh[:,:,0] - pt[line_indexMesh[:][:],column_indexMesh[:][:]][:,:,0])\
                                                                    + nmle[line_indexMesh[:][:],column_indexMesh[:][:]][:,:,1]*(MeshVtxResh[:,:,1]- pt[line_indexMesh[:][:],column_indexMesh[:][:]][:,:,1])\
                                                                    + nmle[line_indexMesh[:][:],column_indexMesh[:][:]][:,:,2]*(MeshVtxResh[:,:,2] - pt[line_indexMesh[:][:],column_indexMesh[:][:]][:,:,2])) ).transpose()
                        
                A = np.dot(Buffer.transpose(), Buffer)
                b = np.dot(Buffer.transpose(), Buffer_B).reshape(6)
                
                det = LA.det(A)
                if (det < 1.0e-10):
                    print "determinant null"
                    break
           
                delta_qsi = -LA.tensorsolve(A, b)
                delta_transfo = LA.inv(Exponential(delta_qsi))
                
                res = np.dot(delta_transfo, res)
                
                print res
        return res
     
#==============================================================================
#                 #define indexes
#                 stack_pixMesh = np.ones( (np.size(MeshVtx[ ::l,:],0)) , dtype = np.float32)
#                 stack_ptMesh = np.ones( (np.size(MeshVtx[ ::l,:],0)) , dtype = np.float32)
#                 pixMesh = np.zeros( (np.size(MeshVtx[ ::l,:],0),2) , dtype = np.float32)
#                 pixMesh = np.stack((pixMesh[:,0],pixMesh[:,1],stack_pixMesh),axis = 1)
#                 ptMesh = np.stack( (MeshVtx[ ::l,0],MeshVtx[ ::l,1],MeshVtx[ ::l,2],stack_ptMesh),axis =1 )
#                 ptMesh = np.dot(ptMesh,Pose.T)
# 
#                 nmlsMesh = np.zeros((MeshNmls.shape[0], MeshNmls.shape[1]), dtype = np.float32)
#                 nmlsMesh[ ::l,:] = np.dot(MeshNmls[ ::l,:],Pose[0:3,0:3].T)  
# 
# 
#                 # projection in 2D space
#                 lptMesh = np.split(ptMesh,4,axis=1)
#                 lptMesh[2] = in_mat_zero2one(lptMesh[2])
#                 pixMesh[ ::l,0] = (lptMesh[0]/lptMesh[2]).reshape(np.size(MeshVtx[ ::l,:],0))
#                 pixMesh[ ::l,1] = (lptMesh[1]/lptMesh[2]).reshape(np.size(MeshVtx[ ::l,:],0))
#                 pixMesh = np.dot(pixMesh,NewImage.intrinsic.T)
#         
#                 column_indexMesh = (np.round(pixMesh[ ::l,0])).astype(int)
#                 line_indexMesh = (np.round(pixMesh[ ::l,1])).astype(int)
#                 # create matrix that have 0 when the conditions are not verified and 1 otherwise
#                 cdt_columnMesh = (column_indexMesh > -1) * (column_indexMesh < Image1.Size[1])
#                 cdt_lineMesh = (line_indexMesh > -1) * (line_indexMesh < Image1.Size[0])
#                 line_indexMesh = line_indexMesh*cdt_lineMesh
#                 column_indexMesh = column_indexMesh*cdt_columnMesh
# 
#                 
#                 diff_Vtx = np.zeros((NewImage.Size[0], NewImage.Size[1], 3), dtype = np.float)
#                 diff_Vtx[:, :] = ptMesh[ Indexes[:],0:3] - pt[:, :,0:3]
#                 diff_Vtx = diff_Vtx*diff_Vtx
#                 norm_diff_Vtx = diff_Vtx.sum(axis=2)
#                 print "Vtx Mesh"
#                 print ptMesh[ Indexes[:],0:3]
#                 print pt[:, :,0:3]
#                 print norm_diff_Vtx
#                 
#                 
#                 diff_Nmle = np.zeros((NewImage.Size[0], NewImage.Size[1], 3), dtype = np.float)
#                 diff_Nmle = nmlsMesh[ Indexes[:],:] - nmle
#                 diff_Nmle = diff_Nmle*diff_Nmle
#                 norm_diff_Nmle = diff_Nmle.sum(axis=2)
#                 print "Nmls Mesh"
#                 print nmlsMesh[ Indexes[:],:]
#                 print nmle
#                 print norm_diff_Nmle
#                 
#                 Norme_Nmle = nmle*nmle
#                 norm_Norme_Nmle = Norme_Nmle.sum(axis=2)
#                 
#                 
#                 mask = np.zeros((NewImage.Size[0], NewImage.Size[1]), dtype = np.float)
#                 mask = cdt_line*cdt_column * (pt[:,:,2] > 0.0) * (norm_Norme_Nmle > 0.0) * (norm_diff_Vtx < self.thresh_dist) * (norm_diff_Nmle < self.thresh_norm)
#                 print sum(sum(mask))
#                 
#                 w = 1.0
#                 # Reshape the long vector created from the operation in a 424*512 image
#                 buff3 = np.zeros((NewImage.Size[0], NewImage.Size[1]), dtype = np.float)
#                 buff4 = np.zeros((NewImage.Size[0], NewImage.Size[1]), dtype = np.float)
#                 buff5 = np.zeros((NewImage.Size[0], NewImage.Size[1]), dtype = np.float)
#                 buff3[:,:] = w*mask[:,:]*(-MeshVtx[ Indexes[:],2]*nmle[:, :,1] + MeshVtx[ Indexes[:],1]*nmle[:, :,2])
#                 buff4[:, :] = w*mask[:, :]*(MeshVtx[ Indexes[:],2]*nmle[:, :,0] - MeshVtx[Indexes[:],0]*nmle[:, :,2])
#                 buff5[:, :] = w*mask[:,:]*(-MeshVtx[ Indexes[:],1]*nmle[:, :,0] + MeshVtx[Indexes[:],0]*nmle[:, :,1])
#                 
#                 Buffer[Indexes_ref[:][:]] = np.dstack((w*mask[:,:]*nmle[ :, :,0], \
#                       w*mask[:,:]*nmle[ :, :,1], \
#                       w*mask[:,:]*nmle[ :, :,2], \
#                       buff3, \
#                       buff4, \
#                       buff5 ))
#                 
#                 buff = np.zeros((NewImage.Size[0], NewImage.Size[1]), dtype = np.float)
#                 buff[:, :] = w*mask[:, :]*(nmle[:, :,0]*(MeshVtx[ Indexes[:],0] - pt[:, :,0]) + nmle[:, :,1]*(MeshVtx[ Indexes[:],1] - pt[:, :,1]) + nmle[:, :,2]*(MeshVtx[ Indexes[:],2] - pt[:, :,2]))
#                 Buffer_B[Indexes_ref[:][:]] = np.dstack(buff ).transpose()
#                         
#                 A = np.dot(Buffer.transpose(), Buffer)
#                 b = np.dot(Buffer.transpose(), Buffer_B).reshape(6)
#                 
#                 det = LA.det(A)
#                 if (det < 1.0e-10):
#                     print "determinant null"
#                     break
#            
#                 delta_qsi = -LA.tensorsolve(A, b)
#                 delta_transfo = LA.inv(Exponential(delta_qsi))
#                 
#                 res = np.dot(delta_transfo, res)
#                 
#                 print res
#         return res                
#==============================================================================
        

#==============================================================================
#         res = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
#         
#         Size = MeshVtx.shape
#         MeshVtx = np.dot(MeshVtx,Pose[0:3,0:3].T)
#         MeshNmls = np.dot(MeshNmls,Pose[0:3,0:3].T)    
#         
# 
#         Indexes_ref = np.arange(Size[0])
#         
#         for l in range(1,self.lvl+1):
#             for it in range(self.max_iter[l-1]):
#                 #nbMatches = 0
#                 #row = np.array([0.,0.,0.,0.,0.,0.,0.])
#                 #Mat = np.zeros(27, np.float32)
#                 b = np.zeros(6, np.float32)
#                 A = np.zeros((6,6), np.float32)
#                 
#                 # For each pixel find correspondinng point by projection
#                 Buffer = np.zeros((Size[0]*Size[1], 6), dtype = np.float32)
#                 Buffer_B = np.zeros((Size[0]*Size[1], 1), dtype = np.float32)
#                 stack_pix = np.ones(Size[0], dtype = np.float32)
#                 stack_pt = np.ones(np.size(MeshVtx[ ::l,:],0), dtype = np.float32)
#                 pix = np.zeros((Size[0], 2), dtype = np.float32)
#                 pix = np.stack((pix[:,0],pix[:,1],stack_pix), axis = 1)
#                 pt = np.stack((MeshVtx[ ::l, 0],MeshVtx[ ::l, 1],MeshVtx[ ::l, 2],stack_pt),axis = 1)
#                 pt = np.dot(res,pt.T).T
#                 nmle = np.zeros((Size[0], Size[1]), dtype = np.float32)
#                 #print "res before nmle"
#                 #print res
#                 nmle[ ::l,:] = np.dot(res[0:3,0:3],MeshNmls[ ::l,:].T).T
#                 #print nmle
#                 
#                 #if (pt[2] != 0.0):
#                 lpt = np.split(pt,4,axis=1)
#                 lpt[2] = in_mat_zero2one(lpt[2])
#                 
#                 # if in 1D pix[0] = pt[0]/pt[2]
#                 pix[ ::l,0] = (lpt[0]/lpt[2]).reshape(np.size(MeshVtx[ ::l,:],0))
#                 # if in 1D pix[1] = pt[1]/pt[2]
#                 pix[ ::l,1] = (lpt[1]/lpt[2]).reshape(np.size(MeshVtx[ ::l,:],0))
#                 pix = np.dot(NewImage.intrinsic,pix[0:Size[0],0:Size[1]].T).T
#                 #print "pix"
#                 #print pix
#                 column_index = (np.round(pix[:,0])).astype(int)
#                 line_index = (np.round(pix[:,1])).astype(int)
#                 
#                 
#                 # create matrix that have 0 when the conditions are not verified and 1 otherwise
#                 cdt_column = (column_index > -1) * (column_index < NewImage.Size[1])*(column_index % l == 0)
#                 cdt_line = (line_index > -1) * (line_index < NewImage.Size[0])*(line_index % l == 0)
#                 line_index = line_index*cdt_line
#                 column_index = column_index*cdt_column
# 
# #==============================================================================
# #                 #kill the duplicate
# #                 indStack = np.stack((line_index,column_index),axis = 1)
# #                 # delete all the not needed duplicate
# #                 indStack = pandas.DataFrame(indStack).drop_duplicates().values 
# #                 pt = pandas.DataFrame(pt).drop_duplicates().values 
# #                 nmle = pandas.DataFrame(nmle).drop_duplicates().values 
# #                 line_index = indStack[:,0]
# #                 column_index = indStack[:,1]
# #==============================================================================
#                 
#                 diff_Vtx =  NewImage.Vtx[line_index[:], column_index[:]] - pt[:,0:3] 
#                 tmp = pt[:,0:3]
#                 pt[:,0:3] = NewImage.Vtx[line_index[:], column_index[:]] 
#                 NewImage.Vtx[line_index[:], column_index[:]]  = tmp
#                 diff_Vtx = diff_Vtx*diff_Vtx
#                 norm_diff_Vtx = diff_Vtx.sum(axis=1)
#                 
#                 diff_Nmle = NewImage.Nmls[line_index[:], column_index[:]] - nmle 
#                 tmp = nmle
#                 nmle = NewImage.Nmls[line_index[:], column_index[:]] 
#                 NewImage.Nmls[line_index[:], column_index[:]]  = nmle
#                 diff_Nmle = diff_Nmle*diff_Nmle
#                 norm_diff_Nmle = diff_Nmle.sum(axis=1)
#                 
#                 Norme_Nmle = nmle*nmle
#                 norm_Norme_Nmle = Norme_Nmle.sum(axis=1)
#                 
#                 mask = cdt_line*cdt_column * (pt[:,2] > 0.0) * (norm_Norme_Nmle > 0.0) * (norm_diff_Vtx < self.thresh_dist) * (norm_diff_Nmle < self.thresh_norm)
#                 print sum(mask)
# #==============================================================================
# #                 print mask.shape
# #                 print NewImage.Vtx[line_index[:], column_index[:]][:,2].shape
# #                 print Buffer.shape
# #                 print Indexes_ref.shape
# #                 print Buffer[Indexes_ref[:]].shape
# #==============================================================================
#                 
#                 w = 1.0
#                 Buffer[Indexes_ref[:]] = np.stack((w*mask[:]*nmle[ :,0], \
#                       w*mask[:]*nmle[ :, 1], \
#                       w*mask[:]*nmle[ :, 2], \
#                       w*mask[:]*(-NewImage.Vtx[line_index[:], column_index[:]][:,2]*nmle[:,1] + NewImage.Vtx[line_index[:], column_index[:]][:,1]*nmle[:,2]), \
#                       w*mask[:]*(NewImage.Vtx[line_index[:], column_index[:]][:,2]*nmle[:,0] - NewImage.Vtx[line_index[:], column_index[:]][:,0]*nmle[:,2]), \
#                       w*mask[:]*(-NewImage.Vtx[line_index[:], column_index[:]][:,1]*nmle[:,0] + NewImage.Vtx[line_index[:], column_index[:]][:,0]*nmle[:,1]) ) , axis = 1)
#                 
#                 Buffer_B[Indexes_ref[:]] = ((w*mask[:]*(nmle[:,0]*(NewImage.Vtx[line_index[:], column_index[:]][:,0] - pt[:,0])\
#                                                       + nmle[:,1]*(NewImage.Vtx[line_index[:], column_index[:]][:,1] - pt[:,1])\
#                                                       + nmle[:,2]*(NewImage.Vtx[line_index[:], column_index[:]][:,2] - pt[:,2])) ).transpose()).reshape(Buffer_B[Indexes_ref[:]].shape)
#                         
#                 A = np.dot(Buffer.transpose(), Buffer)
#                 b = np.dot(Buffer.transpose(), Buffer_B).reshape(6)
#                 
#                 det = LA.det(A)
#                 if (det < 1.0e-10):
#                     print "determinant null"
#                     break
#            
#                 delta_qsi = -LA.tensorsolve(A, b)
#                 delta_transfo = LA.inv(Exponential(delta_qsi))
#                 
#                 res = np.dot(delta_transfo, res)
#                 
#                 print res
#         return res        
#==============================================================================




#==============================================================================
#         res = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
#         
#         column_index_ref = np.array([np.array(range(NewImage.Size[1])) for _ in range(NewImage.Size[0])])
#         line_index_ref = np.array([x*np.ones(NewImage.Size[1], np.int) for x in range(NewImage.Size[0])])
#         Indexes_ref = column_index_ref + NewImage.Size[1]*line_index_ref
#         
#         for l in range(1,self.lvl+1):
#             for it in range(self.max_iter[l-1]):
#                 #nbMatches = 0
#                 #row = np.array([0.,0.,0.,0.,0.,0.,0.])
#                 #Mat = np.zeros(27, np.float32)
#                 b = np.zeros(6, np.float32)
#                 A = np.zeros((6,6), np.float32)
#                 
#                 # For each pixel find correspondinng point by projection
#                 Buffer = np.zeros((NewImage.Size[0]*NewImage.Size[1], 6), dtype = np.float32)
#                 Buffer_B = np.zeros((NewImage.Size[0]*NewImage.Size[1], 1), dtype = np.float32)
#                 
#                 # compute the indexes from the list
#                 MeshStack_pix = np.ones(MeshVtx.shape[0], dtype = np.float32)
#                 MeshStack_pt = np.ones(np.size(MeshVtx[ ::l,:],0), dtype = np.float32)
#                 MeshPix = np.zeros((MeshVtx.shape[0], 2), dtype = np.float32)
#                 MeshPix = np.stack((MeshPix[:,0],MeshPix[:,1],MeshStack_pix), axis = 1)
#                 MeshPt = np.stack((MeshVtx[ ::l, 0],MeshVtx[ ::l, 1],MeshVtx[ ::l, 2],MeshStack_pt),axis = 1)
#                 MeshPt = np.dot(res,MeshPt.T).T
#                 MeshNormle = np.zeros((MeshNmls.shape[0], MeshNmls.shape[1]), dtype = np.float32)
#                 #print "res before nmle"
#                 #print res
#                 MeshNormle[ ::l,:] = np.dot(res[0:3,0:3],MeshNmls[ ::l,:].T).T
#                 #print nmle
#                 
#                 #if (pt[2] != 0.0):
#                 MeshLpt = np.split(MeshPt,4,axis=1)
#                 MeshLpt[2] = in_mat_zero2one(MeshLpt[2])
#                 
#                 # if in 1D pix[0] = pt[0]/pt[2]
#                 MeshPix[ ::l,0] = (MeshLpt[0]/MeshLpt[2]).reshape(np.size(MeshVtx[ ::l,:],0))
#                 # if in 1D pix[1] = pt[1]/pt[2]
#                 MeshPix[ ::l,1] = (MeshLpt[1]/MeshLpt[2]).reshape(np.size(MeshVtx[ ::l,:],0))
#                 MeshPix = np.dot(NewImage.intrinsic,MeshPix[0:MeshVtx.shape[0],0:MeshVtx.shape[1]].T).T  
# 
#                 
#                 column_index = (np.round(MeshPix[:,0])).astype(int)
#                 line_index = (np.round(MeshPix[:,1])).astype(int)
#               
#                 # create matrix that have 0 when the conditions are not verified and 1 otherwise
#                 cdt_column = (column_index > -1) * (column_index < NewImage.Size[1])
#                 cdt_line = (line_index > -1) * (line_index < NewImage.Size[0])
#                 line_index = line_index*cdt_line
#                 column_index = column_index*cdt_column
#                 #Indexes = line_index + Image2.Size[0]*column_index
#                 #Indexes = column_index + Image2.Size[1]*line_index
#                 
#                 diff_Vtx = NewImage.Vtx[line_index[:], column_index[:]] - MeshPt[:,0:3]
#                 diff_Vtx = diff_Vtx*diff_Vtx
#                 norm_diff_Vtx = diff_Vtx.sum(axis=1)
#                 
#                 diff_Nmle = NewImage.Nmls[line_index[:], column_index[:]] - MeshNormle
#                 diff_Nmle = diff_Nmle*diff_Nmle
#                 norm_diff_Nmle = diff_Nmle.sum(axis=1)
#                 
#                 Norme_Nmle = MeshNormle*MeshNormle
#                 norm_Norme_Nmle = Norme_Nmle.sum(axis=1)
#                 
#                 mask = cdt_line*cdt_column * (MeshPt[:,2] > 0.0) * (norm_Norme_Nmle > 0.0) * (norm_diff_Vtx < self.thresh_dist) * (norm_diff_Nmle < self.thresh_norm)
#                 print sum(sum(mask))
#                 
#                 w = 1.0
#                 Buffer[Indexes_ref[:][:]] = np.dstack((w*mask[:]*MeshNormle[ :,0], \
#                       w*mask[:]*MeshNormle[ :,1], \
#                       w*mask[:]*MeshNormle[ :,2], \
#                       w*mask[:]*(-NewImage.Vtx[line_index[:], column_index[:]][:,:,2]*MeshNormle[:,1] + NewImage.Vtx[line_index[:], column_index[:]][:,:,1]*MeshNormle[:,2]), \
#                       w*mask[:]*(NewImage.Vtx[line_index[:], column_index[:]][:,:,2]*MeshNormle[:,0] - NewImage.Vtx[line_index[:], column_index[:]][:,:,0]*MeshNormle[:,2]), \
#                       w*mask[:]*(-NewImage.Vtx[line_index[:], column_index[:]][:,:,1]*MeshNormle[:,0] + NewImage.Vtx[line_index[:], column_index[:]][:,:,0]*MeshNormle[:,1]) ))
#                 
#                 Buffer_B[Indexes_ref[:][:]] = np.dstack(w*mask[:,:]*(MeshNormle[:,0]*(NewImage.Vtx[line_index[:], column_index[:]][:,:,0] - MeshPt[:,0])\
#                                                                     + MeshNormle[:,1]*(NewImage.Vtx[line_index[:], column_index[:]][:,:,1] - MeshPt[:,1])\
#                                                                     + MeshNormle[:,2]*(NewImage.Vtx[line_index[:], column_index[:]][:,:,2] - MeshPt[:,2])) ).transpose()
#                         
#                 A = np.dot(Buffer.transpose(), Buffer)
#                 b = np.dot(Buffer.transpose(), Buffer_B).reshape(6)
#                 
#                 det = LA.det(A)
#                 if (det < 1.0e-10):
#                     print "determinant null"
#                     break
#            
#                 delta_qsi = -LA.tensorsolve(A, b)
#                 delta_transfo = LA.inv(Exponential(delta_qsi))
#                 
#                 res = np.dot(delta_transfo, res)
#                 
#                 print res
#         return res
#==============================================================================