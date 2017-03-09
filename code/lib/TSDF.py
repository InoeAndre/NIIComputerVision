#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 17:16:49 2017

@author: diegothomas
"""

import imp
import numpy as np
from numpy import linalg as LA
RGBD = imp.load_source('RGBD', './lib/RGBD.py')

class TSDFManager():
    
    # Constructor
    def __init__(self, Size):
        self.Size = Size
        self.Voxels = np.array(self.Size, np.float32)
        self.c_x = self.Size[0]/2
        self.c_y = self.Size[1]/2
        self.c_z = 0
        self.dim_x = self.Size[0]/3.0
        self.dim_y = self.Size[1]/3.0
        self.dim_z = self.Size[2]/3.0
        
    
    # Fuse a new RGBD image into the TSDF volume
    def FuseRGBD(self, Image, Pose, s = 1):
        Transform = Pose.inverse()
        
        line_index = 0
        column_index = 0
        pix = np.array([0., 0., 1.])
        pt = np.array([0., 0., 0., 1.])
        for x in range(self.Size[0]/s): # line index (i.e. vertical y axis)
            pt[0] = (x-self.c_x)/self.dim_x
            for y in range(self.Size[1]/s):
                pt[1] = (y-self.c_y)/self.dim_y
                for z in range(self.Size[2]/s):
                    # Project each voxel into  the Image
                    pt[2] = (z-self.c_z)/self.dim_z
                    pt = np.dot(Transform, pt)
                    
                    # Project onto Image2
                    pix[0] = pt[0]/pt[2]
                    pix[1] = pt[1]/pt[2]
                    pix = np.dot(Image.intrinsic, pix)
                    column_index = int(round(pix[0]))
                    line_index = int(round(pix[1]))