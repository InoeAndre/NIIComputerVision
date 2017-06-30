#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 09:35:14 2017

@author: diegothomas
"""

import imp
import numpy as np
from numpy import linalg as LA
import pyopencl as cl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel
from os import path

APP_ROOT = path.dirname( path.abspath( __file__ ) )
RGBD = imp.load_source('RGBD', APP_ROOT + '/RGBD.py')
GPU = imp.load_source('GPUManager', APP_ROOT + '/GPUManager.py')
KernelsOpenCL = imp.load_source('TSDFKernel', APP_ROOT + '/TSDFKernel.py')

class TSDFManager():
    
    # Constructor
    def __init__(self, Size, Image, GPUManager):
        self.Size = Size
        self.c_x = self.Size[0]/2
        self.c_y = self.Size[1]/2
        self.c_z = -0.1
        self.dim_x = self.Size[0]/5.0
        self.dim_y = self.Size[1]/5.0
        self.dim_z = self.Size[2]/5.0
        self.res = np.array([self.c_x, self.dim_x, self.c_y, self.dim_y, self.c_z, self.dim_z], dtype = np.float32)
        self.nu = 0.1
        
        self.GPUManager = GPUManager
        
        self.TSDF_Update_GPU = ElementwiseKernel(self.GPUManager.context, 
                                               """short *TSDF, short *Weight, float *depth, float *Pose, float *Param, 
                                               float *Intrinsic, float nu, int dim_x, int dim_y, int dim_z, 
                                               int nbLines, int nbColumns""",
                                               KernelsOpenCL.Kernel_TSDF,
                                               "TSDF_Update_GPU")
        
        
        self.TSDF_d = cl.array.zeros(self.GPUManager.queue, self.Size, np.int16)
        self.Weight_d = cl.array.zeros(self.GPUManager.queue, self.Size, np.int16)
        self.Param_d = cl.array.to_device(self.GPUManager.queue, self.res)
        self.Pose_d = cl.array.zeros(self.GPUManager.queue, (4,4), np.float32)
        intrinsic_curr = np.array([Image.intrinsic[0,0], Image.intrinsic[1,1], Image.intrinsic[0,2], Image.intrinsic[1,2]])
        self.intrinsic_d = cl.array.to_device(self.GPUManager.queue, intrinsic_curr)
        
    
    # Fuse on the GPU
    def FuseRGBD_GPU(self, Image, Pose):
#        Transform = np.identity(4)
#        Transform[0:3,0:3] = LA.inv(Pose[0:3,0:3])
#        Transform[0:3, 3] = -np.dot(Transform[0:3,0:3],Pose[0:3,3])
        self.Pose_d.set(Pose.astype(np.float32))
        
        self.TSDF_Update_GPU(self.TSDF_d, self.Weight_d, Image.depth_raw_d, self.Pose_d, self.Param_d, self.intrinsic_d, self.nu, self.Size[0], self.Size[1], self.Size[2], Image.Size[0], Image.Size[1])
        
        self.GPUManager.queue.finish()
        
#        self.TSDF = self.TSDF_d.get()
#        import cv2
#        for i in range(512):
#            slice_curr = 255.0*(self.TSDF[:,:,i]+30000.0)/60000.0
#            cv2.imshow("slice", slice_curr.astype(np.uint8))
#            cv2.waitKey(1)
        
                        
                        
                    