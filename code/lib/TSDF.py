#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 17:16:49 2017

@author: diegothomas
"""

import imp
import numpy as np
from numpy import linalg as LA
import pyopencl as cl

RGBD = imp.load_source('RGBD', './lib/RGBD.py')
GPU = imp.load_source('GPUManager', './lib/GPUManager.py')

def sign(x):
    if (x < 0):
        return -1.0
    return 1.0

def in_mat_zero2one(mat):
    """This fonction replace in the matrix all the 0 to 1"""
    mat_tmp = (mat != 0.0)
    res = mat * mat_tmp + ~mat_tmp
    return res

mf = cl.mem_flags

class TSDFManager():
    
    # Constructor
    def __init__(self, Size, Image, GPUManager,nbLabel):
        self.Size = Size
        self.TSDF = np.zeros(self.Size, dtype = np.float32)
        self.prevTSDF = np.zeros(self.Size, dtype = np.float32)
        self.Weight = np.ones(self.Size, dtype = np.float32)
        
        self.SizeSeg = []
        self.TSDFSeg = []
        self.prevTSDFSeg = []
        self.WeightSeg = []
        self.TSDFSegGPU = []
        self.prevTSDFSegGPU = []
        self.WeightSegGPU = []
        for i in range(nbLabel):
            self.SizeSeg.append( (self.Size[0],self.Size[1],self.Size[2]) )
            self.TSDFSeg.append(np.zeros(self.SizeSeg[i], dtype = np.float32) )
            self.prevTSDFSeg.append(np.zeros(self.SizeSeg[i], dtype = np.float32) )
            self.WeightSeg.append(np.ones(self.SizeSeg[i], dtype = np.float32) )

        self.c_x = self.Size[0]/2
        self.c_y = self.Size[1]/2
        self.c_z = -0.1
        self.dim_x = self.Size[0]/5.0
        self.dim_y = self.Size[1]/5.0
        self.dim_z = self.Size[2]/5.0
        
        self.GPUManager = GPUManager
        self.Size_Volume = cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, \
                               hostbuf = np.array([self.Size[0], self.Size[1], self.Size[2]], dtype = np.int32))
        self.TSDFGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.TSDF.nbytes)
        self.prevTSDFGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.prevTSDF.nbytes)
        self.WeightGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Weight.nbytes)
        
        for i in range(nbLabel):
            self.TSDFSegGPU.append( cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.TSDFSeg[i].nbytes) )
            self.prevTSDFSegGPU.append( cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.prevTSDFSeg[i].nbytes) )
            self.WeightSegGPU.append( cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.WeightSeg[i].nbytes) )  

        self.Param = cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, \
                               hostbuf = np.array([self.c_x, self.dim_x, self.c_y, self.dim_y, self.c_z, self.dim_z], dtype = np.float32))
        
        #fmt = cl.ImageFormat(cl.channel_order.RGB, cl.channel_type.FLOAT)
        #self.VMapGPU = cl.Image(self.GPUManager.context, mf.READ_ONLY, fmt, shape = (Image.Size[1], Image.Size[0]))
        #cl.enqueue_copy(self.GPUManager.queue, self.VMapGPU, Image.Vtx, origin = (0,0), region = (Image.Size[1], Image.Size[0]))
        self.DepthGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, Image.depth_image.nbytes)
        self.Calib_GPU = cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = Image.intrinsic)
        self.Pose_GPU = cl.Buffer(self.GPUManager.context, mf.READ_ONLY, 64)
        
    
#######
##GPU code
#####

    # Fuse on the GPU
    def FuseRGBD_GPU(self, Image, Pose ):
        Transform = LA.inv(Pose)
        
        cl.enqueue_write_buffer(self.GPUManager.queue, self.Pose_GPU, Transform)
        cl.enqueue_write_buffer(self.GPUManager.queue, self.DepthGPU, Image.depth_image)
        
        self.prevTSDFGPU = self.TSDFGPU
        #add this in the function self.prevTSDFGPU, self.WeightGPU,
        self.GPUManager.programs['FuseTSDF'].FuseTSDF(self.GPUManager.queue, (self.Size[0], self.Size[1]), None, \
                                self.TSDFGPU, self.DepthGPU, self.Param, self.Size_Volume, self.Pose_GPU, self.Calib_GPU, \
                                np.int32(Image.Size[0]), np.int32(Image.Size[1]),self.prevTSDFGPU, self.WeightGPU)

        cl.enqueue_read_buffer(self.GPUManager.queue, self.TSDFGPU, self.TSDF).wait()
        
        return self.TSDF

    # Fuse on the GPU
    def FuseRGBDSeg_GPU(self, Image, Pose, idx ):
        Transform = LA.inv(Pose)
        
        cl.enqueue_write_buffer(self.GPUManager.queue, self.Pose_GPU, Transform)
        cl.enqueue_write_buffer(self.GPUManager.queue, self.DepthGPU, Image.depth_image)
        #isSeg = 0
        #add this in the function self.prevTSDFGPU, self.WeightGPU,
        self.GPUManager.programs['FuseTSDF'].FuseTSDF(self.GPUManager.queue, (self.Size[0], self.Size[1]), None, \
                                self.TSDFSegGPU[idx], self.DepthGPU, self.Param, self.Size_Volume, self.Pose_GPU, self.Calib_GPU, \
                                np.int32(Image.Size[0]), np.int32(Image.Size[1]),self.prevTSDFSegGPU[idx], self.WeightSegGPU[idx])
        
        cl.enqueue_read_buffer(self.GPUManager.queue, self.TSDFSegGPU[idx], self.TSDFSeg[idx]).wait()
        
        return self.TSDFSeg[idx]
    
        
    # Reay tracing on the GPU
    def RayTracing_GPU(self, Image, Pose):
        result = np.zeros((Image.Size[0], Image.Size[1]), np.float32)
        cl.enqueue_write_buffer(self.GPUManager.queue, self.DepthGPU, result)
        cl.enqueue_write_buffer(self.GPUManager.queue, self.Pose_GPU, Pose)
        
        self.GPUManager.programs['RayTracing'].RayTracing(self.GPUManager.queue, (Image.Size[0], Image.Size[1]), None, \
                                self.TSDFGPU, self.DepthGPU, self.Param, self.Size_Volume, self.Pose_GPU, self.Calib_GPU, \
                                np.int32(Image.Size[0]), np.int32(Image.Size[1]))
        
        cl.enqueue_read_buffer(self.GPUManager.queue, self.DepthGPU, result).wait()
        
        return result
    
#####
#End GPU code
#####
    
    # Fuse a new RGBD image into the TSDF volume
    def FuseRGBD(self, Image, Pose, s = 1):
        Transform = LA.inv(Pose)
        
        nu = 0.1
        line_index = 0
        column_index = 0
        pix = np.array([0., 0., 1.])
        pt = np.array([0., 0., 0., 1.])
        for x in range(self.Size[0]/s): # line index (i.e. vertical y axis)
            pt[0] = (x-self.c_x)/self.dim_x
            print x
            for y in range(self.Size[1]/s):
                pt[1] = (y-self.c_y)/self.dim_y
                for z in range(self.Size[2]/s):
                    # Project each voxel into  the Image
                    pt[2] = (z-self.c_z)/self.dim_z
                    pt = np.dot(Transform, pt)
                    
                    # Project onto Image
                    pix[0] = pt[0]/pt[2]
                    pix[1] = pt[1]/pt[2]
                    pix = np.dot(Image.intrinsic, pix)
                    column_index = int(round(pix[0]))
                    line_index = int(round(pix[1]))
                    
                    if (column_index < 0 or column_index > Image.Size[1]-1 or line_index < 0 or line_index > Image.Size[0]-1):
                        continue
                        
                    depth = Image.Vtx[line_index][column_index][2]
                    
                    # Listing 2
                    dist = pt[2] - depth
                    if (dist > 0):
                        self.TSDF[x][y][z] = min(1.0, dist/nu)
                    else:
                        self.TSDF[x][y][z] = max(-1.0, dist/nu)
                        
    # Fuse a new RGBD image into the TSDF volume
    def FuseRGBD_optimized(self, Image, Pose, s = 1):
        Transform = LA.inv(Pose)
        
        nu = 0.1
        
        column_index_ref = np.array([np.array(range(self.Size[1])) for _ in range(self.Size[0])]) # x coordinates
        column_index_ref = (column_index_ref - self.c_x)/self.dim_x
        
        line_index_ref = np.array([x*np.ones(self.Size[1], np.int) for x in range(self.Size[0])]) # y coordinates
        line_index_ref = (line_index_ref - self.c_y)/self.dim_y
        
        voxels2D = np.dstack((line_index_ref, column_index_ref))
        
        normVtxInput = Image.Vtx[:,:,0:3]*Image.Vtx[:,:,0:3]
        distVtxInput = np.sqrt(normVtxInput.sum(axis=2))
                
        for z in range(self.Size[2]/s): 
            curr_z = (z-self.c_z)/self.dim_z
            stack_z = curr_z*np.ones((self.Size[0], self.Size[1],1), dtype = np.float32)
            
            stack_pix = np.ones((self.Size[0], self.Size[1]), dtype = np.float32)
            stack_pt = np.ones((self.Size[0], self.Size[1],1), dtype = np.float32)
            pix = np.zeros((self.Size[0], self.Size[1],2), dtype = np.float32) # recorded projected location of all voxels in the current slice
            pix = np.dstack((pix, stack_pix))
            pt = np.dstack((voxels2D, stack_z))
            pt = np.dstack((pt, stack_pt))  # record transformed 3D positions of all voxels
            pt = np.dot(Transform,pt.transpose(0,2,1)).transpose(1,2,0)                
                    
            #if (pt[2] != 0.0):
            lpt = np.dsplit(pt,4)
            lpt[2] = in_mat_zero2one(lpt[2])
            
            # if in 1D pix[0] = pt[0]/pt[2]
            pix[ ::s, ::s,0] = (lpt[0]/lpt[2]).reshape((self.Size[0], self.Size[1]))
            # if in 1D pix[1] = pt[1]/pt[2]
            pix[ ::s, ::s,1] = (lpt[1]/lpt[2]).reshape((self.Size[0], self.Size[1]))
            pix = np.dot(Image.intrinsic, pix[0:self.Size[0],0:self.Size[1]].transpose(0,2,1)).transpose(1,2,0)
            column_index = (np.round(pix[:,:,0])).astype(int)
            line_index = (np.round(pix[:,:,1])).astype(int)
            
            # create matrix that have 0 when the conditions are not verified and 1 otherwise
            cdt_column = (column_index > -1) * (column_index < Image.Size[1])
            cdt_line = (line_index > -1) * (line_index < Image.Size[0])
            line_index = line_index*cdt_line
            column_index = column_index*cdt_column
            
            empty_mat = (Image.Vtx[:, :,2] != 0.0)
            #normPt = pt[:,:,0:3]*pt[:,:,0:3]
            #distPt = np.sqrt(normPt.sum(axis=2))
            #diff_Vtx = distPt[:,:] - distVtxInput[line_index[:][:], column_index[:][:]]
            diff_Vtx = pt[:,:, 2] - Image.Vtx[line_index[:][:], column_index[:][:], 2]
            diff_Vtx = diff_Vtx[:,:]*empty_mat[line_index[:][:], column_index[:][:]] - ~empty_mat[line_index[:][:], column_index[:][:]]
            
            self.TSDF[:,:,z] = diff_Vtx/nu
            
    
    def RayTracing(self, Image, Pose):
        result = np.zeros((Image.Size[0], Image.Size[1]), np.float32)
        
        for i in range(Image.Size[0]):
            for j in range(Image.Size[1]):
                # Shoot a ray for the current pixel
                x = (j - Image.intrinsic[0,2])/Image.intrinsic[0,0]
                y = (i - Image.intrinsic[1,2])/Image.intrinsic[1,1]
                tmp = np.array([x, y, 1.0, 1.0])
                tmp = np.dot(Pose, tmp)
                ray = tmp[0:3]
                ray = ray / LA.norm(ray)
                
                nu = 0.1
                pt = 0.5*ray
                voxel = np.round(np.array([pt[0]*self.dim_x + self.c_x, pt[1]*self.dim_y + self.c_y, pt[2]*self.dim_z + self.c_z])).astype(int)
                if (voxel[0] < 0 or voxel[0] > self.Size[0]-1 or voxel[1] < 0 or voxel[1] > self.Size[1]-1 or voxel[2] < 0 or voxel[2] > self.Size[2]-1):
                    break
                prev_TSDF = self.TSDF[voxel[0],voxel[1],voxel[2]]
                pt = pt + nu*ray
                
                while (LA.norm(pt) < 5.0 and prev_TSDF < 0.0):
                    voxel = np.round(np.array([pt[0]*self.dim_x + self.c_x, pt[1]*self.dim_y + self.c_y, pt[2]*self.dim_z + self.c_z])).astype(int)
                    if (voxel[0] < 0 or voxel[0] > self.Size[0]-1 or voxel[1] < 0 or voxel[1] > self.Size[1]-1 or voxel[2] < 0 or voxel[2] > self.Size[2]-1):
                        break
                    new_TSDF = self.TSDF[voxel[0],voxel[1],voxel[2]]
                        
                    if (sign(prev_TSDF*new_TSDF) == -1.0 and prev_TSDF > -1.0):
                        result[i,j] = ((1.0-np.abs(prev_TSDF))*LA.norm(pt - nu*ray) + (1.0-new_TSDF)*LA.norm(pt)) / (2.0 - (np.abs(prev_TSDF) + new_TSDF))
                        break
                    
                    if (new_TSDF > -1.0 and new_TSDF < 0.0):
                        nu = 0.01
                        
                    prev_TSDF = new_TSDF
                    pt = pt + nu*ray
        
        return result
        

                    