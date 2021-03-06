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
from array import array

RGBD = imp.load_source('RGBD', './lib/RGBD.py')
GPU = imp.load_source('GPUManager', './lib/GPUManager.py')
KernelsOpenCL = imp.load_source('KernelsOpenCL', 'lib/KernelsOpenCL.py')
General = imp.load_source('General', './lib/General.py')


def sign(x):
    if (x < 0):
        return -1.0
    return 1.0


mf = cl.mem_flags

class TSDFManager():
    """
    Manager Truncated Signed Distance Function.
    """

    def __init__(self, Size, Image, GPUManager):
        """
        Constructor
        :param Size: dimension of each axis of the volume
        :param Image: RGBD image to compare
        :param GPUManager: GPU environment for GPU computation
        """
        #dimensions
        self.Size = Size
        self.c_x = Size[0]/2
        self.c_y = Size[1]/2
        self.c_z = Size[2]/2
        # resolution
        VoxSize = 0.005
        self.dim_x = 1.0/VoxSize
        self.dim_y = 1.0/VoxSize
        self.dim_z = 1.0/VoxSize
        # put dimensions and resolution in one vector
        self.res = np.array([self.c_x, self.dim_x, self.c_y, self.dim_y, self.c_z, self.dim_z], dtype = np.float32)
        
        self.GPUManager = GPUManager
        self.Size_Volume = cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, \
                               hostbuf = np.array([self.Size[0], self.Size[1], self.Size[2]], dtype = np.int32))

        self.TSDF = np.zeros(self.Size, dtype=np.int16)
        self.TSDFGPU =cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.TSDF.nbytes)
        self.Weight = np.zeros(self.Size, dtype=np.int16)
        self.WeightGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Weight.nbytes)

        self.Param = cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, \
                               hostbuf = self.res)
        self.Pose = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        self.DepthGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, Image.depth_image.nbytes)
        self.Calib_GPU = cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = Image.intrinsic)
        self.Pose_GPU = cl.Buffer(self.GPUManager.context, mf.READ_ONLY, self.Pose.nbytes)



#######
##GPU code
#####

    # Fuse on the GPU
    def FuseRGBD_GPU(self, Image, Pose):
        """
        Update the TSDF volume with Image
        :param Image: RGBD image to update to its surfaces
        :param Pose: transform from the first camera pose to the last camera pose
        :return: none
        """
        # initialize buffers
        cl.enqueue_write_buffer(self.GPUManager.queue, self.Pose_GPU, Pose)
        cl.enqueue_write_buffer(self.GPUManager.queue, self.DepthGPU, Image.depth_image)

        # fuse data of the RGBD imnage with the TSDF volume 3D model
        self.GPUManager.programs['FuseTSDF'].FuseTSDF(self.GPUManager.queue, (self.Size[0], self.Size[1]), None, \
                                self.TSDFGPU, self.DepthGPU, self.Param, self.Size_Volume, self.Pose_GPU, self.Calib_GPU, \
                                np.int32(Image.Size[0]), np.int32(Image.Size[1]),self.WeightGPU)

        # update CPU array. Read the buffer to write in the CPU array.
        cl.enqueue_read_buffer(self.GPUManager.queue, self.TSDFGPU, self.TSDF).wait()
        '''
        # TEST if TSDF contains NaN
        TSDFNaN = np.count_nonzero(np.isnan(self.TSDF))
        print "TSDFNaN : %d" %(TSDFNaN)
        '''
        cl.enqueue_read_buffer(self.GPUManager.queue, self.WeightGPU, self.Weight).wait()  


    
#####
#End GPU code
#####
    
    # Fuse a new RGBD image into the TSDF volume
    def FuseRGBD(self, Image, Pose, s = 1):
        """
        Fuse data of 3D TSDF model with RGBD image CPU
        :param Image: RGBD image to update to its surfaces
        :param Pose: transform from the first camera pose to the last camera pose
        :param s:  subsampling factor
        :return: none
        NOT USED FUNCTIONS
        """
        print self.c_x
        print self.dim_x
        print self.c_y
        print self.dim_y
        print self.c_z
        print self.dim_z

        # init
        Transform = Pose
        convVal = 32767.0
        nu = 0.05
        line_index = 0
        column_index = 0
        nb_vert = 0
        nb_miss = 0
        pix = np.array([0., 0., 1.])
        pt = np.array([0., 0., 0., 1.])
        for x in range(self.Size[0]/s): # line index (i.e. vertical y axis)
            # put the middle of the volume in the middle of the image for axis x
            pt[0] = (x-self.c_x)/self.dim_x
            print x
            #print pt[0]
            for y in range(self.Size[1]/s):
                # put the middle of the volume in the middle of the image for axis y
                pt[1] = (y-self.c_y)/self.dim_y
                # Transofrm in current camera pose
                x_T =  Transform[0,0]*pt[0] + Transform[0,1]*pt[1] + Transform[0,3]
                y_T =  Transform[1,0]*pt[0] + Transform[1,1]*pt[1] + Transform[1,3]
                z_T =  Transform[2,0]*pt[0] + Transform[2,1]*pt[1] + Transform[2,3] 
                #print Transform
                #print pt
                for z in range(self.Size[2]/s):
                    nb_vert +=1
                    # Project each voxel into  the Image
                    pt[2] = (z-self.c_z)/self.dim_z
                    pt_Tx = x_T + Transform[0,2]*pt[2] 
                    pt_Ty = y_T + Transform[1,2]*pt[2]
                    pt_Tz = z_T + Transform[2,2]*pt[2]

                    # Project onto Image
                    pix[0] = pt_Tx/pt_Tz
                    pix[1] = pt_Ty/pt_Tz
                    pix = np.dot(Image.intrinsic, pix)
                    #print pix
                    column_index = int(round(pix[0]))
                    line_index = int(round(pix[1]))

                    # check if the pix is in the frame
                    if (column_index < 0 or column_index > Image.Size[1]-1 or line_index < 0 or line_index > Image.Size[0]-1):
                        #print "column_index : %d , line_index : %d" %(column_index,line_index)
                        nb_miss +=1
                        if self.Weight[x][y][z]==0 : 
                            self.TSDF[x][y][z] = int(convVal)
                            continue

                    # get corresponding depth
                    depth = Image.depth_image[line_index][column_index]

                    # Check depth value
                    if depth == 0 :
                        nb_miss +=1
                        if self.Weight[x][y][z]==0 :
                            self.TSDF[x][y][z] = int(convVal)
                            continue
                   
                    # compute distance between voxel and surface
                    dist = -(pt_Tz - depth)/nu
                    dist = min(1.0, max(-1.0, dist))
                    
                    if (dist > 1.0):
                        self.TSDF[x][y][z] = 1.0
                        print "x %d, y %d, z %d" %(x,y,z)
                    else:
                        self.TSDF[x][y][z] = max(-1.0, dist)

                    #running average
                    prev_tsdf = float(self.TSDF[x][y][z])/convVal
                    prev_weight = float(self.Weight[x][y][z])
                    
                    self.TSDF[x][y][z] =  int(round(((prev_tsdf*prev_weight+dist)/(prev_weight+1.0))*convVal))
                    if (dist < 1.0 and dist > -1.0):
                        self.Weight[x][y][z] = min(1000, self.Weight[x][y][z]+1)
        print "nb vert : %d, nb miss : %d" % (nb_vert,nb_miss)
        cl.enqueue_copy(self.GPUManager.queue, self.TSDFGPU, self.TSDF).wait()

                
    # Fuse a new RGBD image into the TSDF volume
    def FuseRGBD_optimized(self, Image, Pose, s = 1):
        """
        Fuse data of 3D TSDF model with RGBD image CPU optimize
        :param Image: RGBD image to update to its surfaces
        :param Pose: transform from the first camera pose to the last camera pose
        :param s:  subsampling factor
        :return: none
        NOT USED FUNCTIONS
        """
        Transform = Pose#LA.inv(Pose)
        
        nu = 0.05
        
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
            lpt[2] = General.in_mat_zero2one(lpt[2])
            
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
        # running average is missing in the function
    
