#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 13:38:34 2017

@author: diegothomas
"""


import imp
import numpy as np
import pyopencl as cl

GPU = imp.load_source('GPUManager', './lib/GPUManager.py')
KernelsOpenCL = imp.load_source('MarchingCubes_KernelOpenCL', './lib/MarchingCubes_KernelOpenCL.py')

mf = cl.mem_flags

class My_MarchingCube():
    
    def __init__(self, Size, Res, Iso, GPUManager):
        self.Size = Size
        self.res = Res
        self.iso = Iso
        #self.Faces = np.zeros((self.Size[0]*self.Size[1]*self.Size[2], 3), dtype = np.int32)
        #self.Vertices = np.zeros((3*self.Size[0]*self.Size[1]*self.Size[2], 3), dtype = np.float32)
        self.nb_faces = np.array([0], dtype = np.int32)
        
        self.GPUManager = GPUManager
        
        self.GPUManager.programs['MarchingCubesIndexing'] = cl.Program(self.GPUManager.context, KernelsOpenCL.Kernel_MarchingCubeIndexing).build()
        self.GPUManager.programs['MarchingCubes'] = cl.Program(self.GPUManager.context, KernelsOpenCL.Kernel_MarchingCube).build()
        
        
        self.Size_Volume = cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, \
                               hostbuf = np.array([self.Size[0], self.Size[1], self.Size[2]], dtype = np.int32))
        
        tmp = np.zeros(self.Size, dtype = np.int32)
        self.OffsetGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, tmp.nbytes)
        self.IndexGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, tmp.nbytes)
        self.FaceCounterGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = self.nb_faces)
        
        self.ParamGPU = cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, \
                               hostbuf = self.res)
        
        #self.FacesGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Faces.nbytes)
        #self.VerticesGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Vertices.nbytes)
        
        
    def runGPU(self, VolGPU):
        
        self.GPUManager.programs['MarchingCubesIndexing'].MarchingCubesIndexing(self.GPUManager.queue, (self.Size[0]-1, self.Size[1]-1), None, \
                                VolGPU, self.OffsetGPU, self.IndexGPU, self.Size_Volume, np.int32(self.iso), self.FaceCounterGPU)
        
        cl.enqueue_read_buffer(self.GPUManager.queue, self.FaceCounterGPU, self.nb_faces).wait()
        print "nb Faces: ", self.nb_faces[0]
        
        self.Faces = np.zeros((self.nb_faces[0], 3), dtype = np.int32)
        self.Vertices = np.zeros((3*self.nb_faces[0], 3), dtype = np.float32)
        
        self.FacesGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Faces.nbytes)
        self.VerticesGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Vertices.nbytes)
        
        self.GPUManager.programs['MarchingCubes'].MarchingCubes(self.GPUManager.queue, (self.Size[0]-1, self.Size[1]-1), None, \
                                self.OffsetGPU, self.IndexGPU, self.VerticesGPU, self.FacesGPU, self.ParamGPU, self.Size_Volume)
        
        cl.enqueue_read_buffer(self.GPUManager.queue, self.VerticesGPU, self.Vertices).wait()
        cl.enqueue_read_buffer(self.GPUManager.queue, self.FacesGPU, self.Faces).wait()


    '''
        Function to record the created mesh into a .ply file
    '''
    def SaveToPly(self, name):
        f = open(name, 'wb')    
        
        # Write headers
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment ply file created by Diego Thomas\n")
        f.write("element vertex %d \n" %(3*self.nb_faces[0]))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("element face %d \n" %(self.nb_faces[0]))
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        # Write vertices
        for i in range(3*self.nb_faces[0]):
            f.write("%f %f %f \n" %(self.Vertices[i,0], self.Vertices[i,1], self.Vertices[i,2]))
            
        # Write the faces
        for i in range(self.nb_faces[0]):
            f.write("3 %d %d %d \n" %(self.Faces[i,0], self.Faces[i,1], self.Faces[i,2])) 
                     
        f.close()
                    
                    
                    
                    
                    
                    
                    
                    