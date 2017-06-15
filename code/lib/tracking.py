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
import warnings

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
        
        skew = np.zeros((3,3), np.float64)
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
        
        
    def InvPose(self,Pose):
        '''Compute the inverse transform of Pose''' 
        PoseInv = np.zeros(Pose.shape,Pose.dtype)
        PoseInv[0:3,0:3] = LA.inv(Pose[0:3,0:3])
        PoseInv[0:3,3] = -np.dot(PoseInv[0:3,0:3],Pose[0:3,3])
        PoseInv[3,3] = 1.0
        return PoseInv
        
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
              
            

                nmle = np.zeros((Image1.Size[0], Image1.Size[1],Image1.Size[2]), dtype = np.float32)
                nmle[ ::l, ::l,:] = np.dot(res[0:3,0:3],Image1.Nmls[ ::l, ::l,:].transpose(0,2,1)).transpose(1,2,0)
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
                # create matrix that have 0 when the conditions are not verified and 1 otherwise
                cdt_column = (column_index > -1) * (column_index < Image2.Size[1])
                cdt_line = (line_index > -1) * (line_index < Image2.Size[0])
                line_index = line_index*cdt_line
                column_index = column_index*cdt_column
                  
                diff_Vtx = Image2.Vtx[line_index[:][:], column_index[:][:]] - pt[:,:,0:3]
                diff_Vtx = diff_Vtx*diff_Vtx
                norm_diff_Vtx = diff_Vtx.sum(axis=2)
                mask_vtx =  (norm_diff_Vtx < self.thresh_dist)                
                print "mask_vtx"
                print sum(sum(mask_vtx))     
                
                diff_Nmle = Image2.Nmls[line_index[:][:], column_index[:][:]] - nmle        
                diff_Nmle = diff_Nmle*diff_Nmle
                norm_diff_Nmle = diff_Nmle.sum(axis=2)
                mask_nmls =  (norm_diff_Nmle < self.thresh_norm)                 
                print "mask_nmls"
                print sum(sum(mask_nmls))   
                
                Norme_Nmle = nmle*nmle
                norm_Norme_Nmle = Norme_Nmle.sum(axis=2)
                
                mask_pt =  (pt[:,:,2] > 0.0)
                print "mask_pt"
                print sum(sum(mask_pt)  )
                
                print "cdt_column"
                print sum(sum( (cdt_column==0))  )
                
                print "cdt_line"
                print sum(sum( (cdt_line==0)) )
                
                mask = cdt_line*cdt_column * (pt[:,:,2] > 0.0) * (norm_Norme_Nmle > 0.0) * (norm_diff_Vtx < self.thresh_dist) * (norm_diff_Nmle < self.thresh_norm)
                print "final correspondence"
                print sum(sum(mask))
                

                
                w = 1.0
                Buffer[Indexes_ref[:][:]] = np.dstack((w*mask[:,:]*nmle[ :, :,0], \
                      w*mask[:,:]*nmle[ :, :,1], \
                      w*mask[:,:]*nmle[ :, :,2], \
                      w*mask[:,:]*(-Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,2]*nmle[:,:,1] + Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,1]*nmle[:,:,2]), \
                      w*mask[:,:]*(Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,2]*nmle[:,:,0] - Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,0]*nmle[:,:,2]), \
                      w*mask[:,:]*(-Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,1]*nmle[:,:,0] + Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,0]*nmle[:,:,1]) ))
                
                Buffer_B[Indexes_ref[:][:]] = np.dstack(w*mask[:,:]*(nmle[:,:,0]*(Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,0] - pt[:,:,0])\
                                                                    + nmle[:,:,1]*(Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,1] - pt[:,:,1])\
                                                                    + nmle[:,:,2]*(Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,2] - pt[:,:,2])) ).transpose()
                   
                
                A = np.dot(Buffer.transpose(), Buffer)
                b = np.dot(Buffer.transpose(), Buffer_B).reshape(6)
                
                sign,logdet = LA.slogdet(A)
                det = sign * np.exp(logdet)
                if (det == 0.0):
                    print "determinant null"
                    print det
                    warnings.warn("this is a warning message")
                    break
           
                delta_qsi = -LA.tensorsolve(A, b)
                delta_transfo = Exponential(delta_qsi)
                delta_transfo = self.InvPose(delta_transfo)             
                res = np.dot(delta_transfo, res)
                print "delta_transfo"
                print delta_transfo                    
                print "res"
                print res
        return res
    
    def RegisterRGBD_optimize2(self, Image1, Image2,Pose):
        
        res = Pose#np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        resInv = self.InvPose(Pose)#np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        
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
                
                # adapt the current image with 4 coordinates and update the position
                stack_NewPt = np.ones((np.size(Image1.Vtx[ ::l, ::l,:],0), np.size(Image1.Vtx[ ::l, ::l,:],1)), dtype = np.float64)
                NewPt = np.dstack((Image1.Vtx[ ::l, ::l, :],stack_NewPt))
                NewVtx = np.dot(resInv,NewPt.transpose(0,2,1)).transpose(1,2,0)
                NewNmls = np.dot(resInv[0:3,0:3],Image1.Nmls.transpose(0,2,1)).transpose(1,2,0)
                
                # For each pixel find correspondinng point by projection
                Buffer = np.zeros((Image2.Size[0]*Image2.Size[1], 6), dtype = np.float32)
                Buffer_B = np.zeros((Image2.Size[0]*Image2.Size[1], 1), dtype = np.float32)
                stack_pix = np.ones((Image2.Size[0], Image2.Size[1]), dtype = np.float32)
                stack_pt = np.ones((np.size(Image2.Vtx[ ::l, ::l,:],0), np.size(Image2.Vtx[ ::l, ::l,:],1)), dtype = np.float32)
                pix = np.zeros((Image2.Size[0], Image2.Size[1],2), dtype = np.float32)
                pix = np.dstack((pix,stack_pix))
                pt = np.dstack((Image2.Vtx[ ::l, ::l, :],stack_pt))


                nmle = np.zeros((Image2.Size[0], Image2.Size[1],Image2.Size[2]), dtype = np.float32)
                nmle[ ::l, ::l,:] = Image2.Nmls[ ::l, ::l,:]
                #if (pt[2] != 0.0):
                lpt = np.dsplit(pt,4)
                lpt[2] = in_mat_zero2one(lpt[2])

                
                # if in 1D pix[0] = pt[0]/pt[2]
                pix[ ::l, ::l,0] = (lpt[0]/lpt[2]).reshape(np.size(Image2.Vtx[ ::l, ::l,:],0), np.size(Image2.Vtx[ ::l, ::l,:],1))
                # if in 1D pix[1] = pt[1]/pt[2]
                pix[ ::l, ::l,1] = (lpt[1]/lpt[2]).reshape(np.size(Image2.Vtx[ ::l, ::l,:],0), np.size(Image2.Vtx[ ::l, ::l,:],1))
                pix = np.dot(Image1.intrinsic,pix[0:Image2.Size[0],0:Image2.Size[1]].transpose(0,2,1)).transpose(1,2,0)

                column_index = (np.round(pix[:,:,0])).astype(int)
                line_index = (np.round(pix[:,:,1])).astype(int)

                # create matrix that have 0 when the conditions are not verified and 1 otherwise
                cdt_column = (column_index > -1) * (column_index < Image2.Size[1])
                cdt_line = (line_index > -1) * (line_index < Image2.Size[0])
                line_index = line_index*cdt_line
                column_index = column_index*cdt_column
                diff_Vtx = pt[:,:,0:3] - NewVtx[line_index[:][:], column_index[:][:]][:,:,0:3] 
                diff_Vtx = diff_Vtx*diff_Vtx
                norm_diff_Vtx = diff_Vtx.sum(axis=2)
                mask_vtx =  (norm_diff_Vtx < self.thresh_dist)

                print "mask_vtx"
                print sum(sum(mask_vtx))     
                
                diff_Nmle = nmle - NewNmls[line_index[:][:], column_index[:][:]]
                diff_Nmle = diff_Nmle*diff_Nmle
                norm_diff_Nmle = diff_Nmle.sum(axis=2)
                mask_nmls =  (norm_diff_Nmle < self.thresh_norm)
                print "mask_nmls"
                print sum(sum(mask_nmls))   
                
                Norme_Nmle = nmle*nmle
                norm_Norme_Nmle = Norme_Nmle.sum(axis=2)
                
                mask_pt =  (pt[:,:,2] > 0.0)
                print "mask_pt"
                print sum(sum(mask_pt)  )
                
                print "cdt_column"
                print sum(sum( (cdt_column==0))  )
                
                print "cdt_line"
                print sum(sum( (cdt_line==0)) )
                
                mask = cdt_line*cdt_column * (pt[:,:,2] > 0.0) * (norm_Norme_Nmle > 0.0) * (norm_diff_Vtx < self.thresh_dist) * (norm_diff_Nmle < self.thresh_norm)
                print "final correspondence"
                print sum(sum(mask))

                
                w = 1.0
                Buffer[Indexes_ref[:][:]] = np.dstack((w*mask[:,:]*nmle[ :, :,0], \
                      w*mask[:,:]*nmle[ :, :,1], \
                      w*mask[:,:]*nmle[ :, :,2], \
                      w*mask[:,:]*(-NewVtx[line_index[:][:], column_index[:][:]][:,:,2]*nmle[:,:,1] + NewVtx[line_index[:][:], column_index[:][:]][:,:,1]*nmle[:,:,2]), \
                      w*mask[:,:]*(NewVtx[line_index[:][:], column_index[:][:]][:,:,2]*nmle[:,:,0] - NewVtx[line_index[:][:], column_index[:][:]][:,:,0]*nmle[:,:,2]), \
                      w*mask[:,:]*(-NewVtx[line_index[:][:], column_index[:][:]][:,:,1]*nmle[:,:,0] + NewVtx[line_index[:][:], column_index[:][:]][:,:,0]*nmle[:,:,1]) ))
                
                Buffer_B[Indexes_ref[:][:]] = np.dstack(w*mask[:,:]*(nmle[:,:,0]*(NewVtx[line_index[:][:], column_index[:][:]][:,:,0] - pt[:,:,0])\
                                                                    + nmle[:,:,1]*(NewVtx[line_index[:][:], column_index[:][:]][:,:,1] - pt[:,:,1])\
                                                                    + nmle[:,:,2]*(NewVtx[line_index[:][:], column_index[:][:]][:,:,2] - pt[:,:,2])) ).transpose()

                A = np.dot(Buffer.transpose(), Buffer)
                b = np.dot(Buffer.transpose(), Buffer_B).reshape(6)
                
                sign,logdet = LA.slogdet(A)
                det = sign * np.exp(logdet)
                if (det == 0.0):
                    print "determinant null"
                    print det
                    warnings.warn("this is a warning message")
                    break
           
                delta_qsi = -LA.tensorsolve(A, b)
                delta_transfo = Exponential(delta_qsi)
                delta_transfo = self.InvPose(delta_transfo)             
                res = np.dot(delta_transfo, res)
                resInv = self.InvPose(res) 
                print "delta_transfo"
                print delta_transfo                    
                print "res"
                print res
                #break
        return resInv



            
    def RegisterRGBDMesh(self, NewImage, MeshVtx, MeshNmls,Pose):            

        res = Pose
        
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
                    
                    
                    if (column_index < 0 or column_index > NewImage.Size[1]-1 or line_index < 0 or line_index > NewImage.Size[0]-1):
                        continue
                    
                    # Compute distance betwn matches and btwn normals
                    match_vtx = NewImage.Vtx[line_index, column_index]

                    distance = LA.norm(pt[0:3] - match_vtx)
                    if (distance > self.thresh_dist):
                        print "no Vtx correspondance"
                        print distance
                        continue
                    match_nmle = NewImage.Nmls[line_index, column_index]

                    distance = LA.norm(nmle - match_nmle)
                    if (distance > self.thresh_norm):
                        print "no Nmls correspondance"
                        print distance
                        continue
                        
                    w = 1.0
                    # Complete Jacobian matrix
                    row[0] = w*nmle[0]
                    row[1] = w*nmle[1]
                    row[2] = w*nmle[2]
                    row[3] = w*(-match_vtx[2]*nmle[1] + match_vtx[1]*nmle[2])
                    row[4] = w*(match_vtx[2]*nmle[0] - match_vtx[0]*nmle[2])
                    row[5] = w*(-match_vtx[1]*nmle[0] + match_vtx[0]*nmle[1])
                    row[6] = w*( nmle[0]*(match_vtx[0] - pt[0])\
                               + nmle[1]*(match_vtx[1] - pt[1])\
                               + nmle[2]*(match_vtx[2] - pt[2]))
                                
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
        print "MeshVtx.shape"
        print MeshVtx.shape
        
        # Initializing the res with the current Pose so that 
        Size = MeshVtx.shape
        res = Pose
        
        for l in range(1,self.lvl+1):
            for it in range(self.max_iter[l-1]):
                #nbMatches = 0
                #row = np.array([0.,0.,0.,0.,0.,0.,0.])
                #Mat = np.zeros(27, np.float32)
                b = np.zeros(6, np.float32)
                A = np.zeros((6,6), np.float32)
                
                # For each pixel find correspondinng point by projection
                Buffer = np.zeros((Size[0], 6), dtype = np.float32)
                Buffer_B = np.zeros((Size[0], 1), dtype = np.float32)
                stack_pix = np.ones(Size[0], dtype = np.float32)
                stack_pt = np.ones(np.size(MeshVtx[ ::l,:],0), dtype = np.float32)
                pix = np.zeros((Size[0], 2), dtype = np.float32)
                pix = np.stack((pix[:,0],pix[:,1],stack_pix), axis = 1)
                pt = np.stack((MeshVtx[ ::l, 0],MeshVtx[ ::l, 1],MeshVtx[ ::l, 2],stack_pt),axis = 1)

                pt = np.dot(res,pt.T).T
                nmle = np.zeros((Size[0], Size[1]), dtype = np.float32)
                nmle[ ::l,:] = np.dot(res[0:3,0:3],MeshNmls[ ::l,:].T).T

                lpt = np.split(pt,4,axis=1)
                lpt[2] = in_mat_zero2one(lpt[2])
                
                # if in 1D pix[0] = pt[0]/pt[2]
                pix[ ::l,0] = (lpt[0]/lpt[2]).reshape(np.size(MeshVtx[ ::l,:],0))
                # if in 1D pix[1] = pt[1]/pt[2]
                pix[ ::l,1] = (lpt[1]/lpt[2]).reshape(np.size(MeshVtx[ ::l,:],0))
                pix = np.dot(NewImage.intrinsic,pix[0:Size[0],0:Size[1]].T).T
                column_index = (np.round(pix[:,0])).astype(int)
                line_index = (np.round(pix[:,1])).astype(int)
                
                
                # create matrix that have 0 when the conditions are not verified and 1 otherwise
                cdt_column = (column_index > -1) * (column_index < NewImage.Size[1])
                cdt_line = (line_index > -1) * (line_index < NewImage.Size[0])
                line_index = line_index*cdt_line
                column_index = column_index*cdt_column
                
                diff_Vtx =  NewImage.Vtx[line_index[:], column_index[:]] - pt[:,0:3] 
                diff_Vtx = diff_Vtx*diff_Vtx
                norm_diff_Vtx = diff_Vtx.sum(axis=1)
                mask_vtx =  (norm_diff_Vtx < self.thresh_dist)
                print "norm_diff_Vtx : max, min , median"
                #print norm_diff_Vtx[51300]
                print "max : %f; min : %f; median : %f; var :  %f " % (np.max(norm_diff_Vtx),np.min(norm_diff_Vtx) ,np.median(norm_diff_Vtx),np.var(norm_diff_Vtx) )
                #print "mask_vtx"
                #print norm_diff_Vtx*mask_vtx
                #print sum(mask_vtx)  
                
                diff_Nmle = NewImage.Nmls[line_index[:], column_index[:]] - nmle 
                diff_Nmle = diff_Nmle*diff_Nmle
                norm_diff_Nmle = diff_Nmle.sum(axis=1)
                mask_nmls =  (norm_diff_Nmle < self.thresh_norm)
                print "mask_nmls"
                print sum(mask_nmls)     
                
                Norme_Nmle = nmle*nmle
                norm_Norme_Nmle = Norme_Nmle.sum(axis=1)
                
                
                mask_pt =  (pt[:,2] > 0.0)
                print "mask_pt"
                print sum(mask_pt)  
                
                print "cdt_column"
                print sum( (cdt_column==0))  
                
                print "cdt_line"
                print sum( (cdt_line==0))  
                
                mask = cdt_line*cdt_column * (pt[:,2] > 0.0) * (norm_Norme_Nmle > 0.0) * (norm_diff_Vtx < self.thresh_dist) * (norm_diff_Nmle < self.thresh_norm)
                print "final correspondence"
                print sum(mask)

                
                w = 1.0
                Buffer[:] = np.stack((w*mask[:]*nmle[ :,0], \
                      w*mask[:]*nmle[ :, 1], \
                      w*mask[:]*nmle[ :, 2], \
                      w*mask[:]*(NewImage.Vtx[line_index[:], column_index[:]][:,1]*nmle[:,2] - NewImage.Vtx[line_index[:], column_index[:]][:,2]*nmle[:,1]), \
                      w*mask[:]*(NewImage.Vtx[line_index[:], column_index[:]][:,2]*nmle[:,0] - NewImage.Vtx[line_index[:], column_index[:]][:,0]*nmle[:,2]), \
                      w*mask[:]*(NewImage.Vtx[line_index[:], column_index[:]][:,0]*nmle[:,1] - NewImage.Vtx[line_index[:], column_index[:]][:,1]*nmle[:,0]) ) , axis = 1)
                
                Buffer_B[:] = ((w*mask[:]*(nmle[:,0]*(NewImage.Vtx[line_index[:], column_index[:]][:,0] - pt[:,0])\
                                                      + nmle[:,1]*(NewImage.Vtx[line_index[:], column_index[:]][:,1] - pt[:,1])\
                                                      + nmle[:,2]*(NewImage.Vtx[line_index[:], column_index[:]][:,2] - pt[:,2])) ).transpose()).reshape(Buffer_B[:].shape)
  
                A = np.dot(Buffer.transpose(), Buffer)
                b = np.dot(Buffer.transpose(), Buffer_B).reshape(6)

                
                sign,logdet = LA.slogdet(A)
                det = sign * np.exp(logdet)
                if (det == 0.0):
                    print "determinant null"
                    print det
                    warnings.warn("this is a warning message")
                    break
           
                delta_qsi = -LA.tensorsolve(A, b)
                delta_transfo = LA.inv(Exponential(delta_qsi))
                
                res = np.dot(delta_transfo, res)
                print "delta_transfo"
                print delta_transfo
                print "res"
                print res

                
        res = self.InvPose(res)   
        
        return res        




