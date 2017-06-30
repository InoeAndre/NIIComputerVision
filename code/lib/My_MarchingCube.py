#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 09:43:24 2017

@author: diegothomas
"""


import imp
import numpy as np
import pyopencl as cl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel
#import pyopencl.bitonic_sort
from os import path

GPU = imp.load_source('GPUManager', './lib/GPUManager.py')
KernelsOpenCL = imp.load_source('MarchingCubesKernel', './lib/MarchingCubes_KernelOpenCL.py')

mf = cl.mem_flags

class My_MarchingCube():
    
    def __init__(self, Size, Res, Iso, GPUManager):
        self.Size = Size
        self.res = Res
        self.iso = Iso
        self.nb_faces = np.array([0], dtype = np.int32)
        self.nb_vertices = np.array([0], dtype = np.int32)
        
        self.GPUManager = GPUManager
        
        #self.sorter = cl.bitonic_sort.BitonicSort(self.GPUManager.context)
        
        self.MC_Indexing = ElementwiseKernel(self.GPUManager.context, 
                                               """short *TSDF, short *IndexVal, int *Offset, int *faces_counter,
                                               float iso, int dim_x, int dim_y, int dim_z""",
                                               KernelsOpenCL.Kernel_MarchingCubeIndexing,
                                               "MC_Indexing", 
                                               preamble = KernelsOpenCL.Preambule_MCIndexing)
        
        self.MC = ElementwiseKernel(self.GPUManager.context, 
                                                """short *TSDF, int *Offset, short *IndexVal, 
                                                float *Vertices, int *Faces, float *Param, 
                                                int dim_x, int dim_y, int dim_z""",
                                               KernelsOpenCL.Kernel_MarchingCube,
                                               "MC", 
                                               preamble = KernelsOpenCL.Preambule_MC)
        
        self.OrganizeFaces = ElementwiseKernel(self.GPUManager.context, 
                                                """int *Faces, float *Vertices, float *Unique_Vertices, 
                                                int nb_vertices""",
                                               KernelsOpenCL.Kernel_OrganizeFaces,
                                               "OrganizeFaces",)
        
        self.ComputeNormales = ElementwiseKernel(self.GPUManager.context, 
                                                """int *Faces, float *Vertices, int *Normales""",
                                               KernelsOpenCL.Kernel_ComputeNormales,
                                               "ComputeNormales",)
        
        self.Normalise = ElementwiseKernel(self.GPUManager.context, 
                                                """float *Normales""",
                                               KernelsOpenCL.Kernel_Normalise,
                                               "Normalise",)
        
        
        self.Offset_d = cl.array.zeros(self.GPUManager.queue, self.Size, np.int32)
        self.Index_d = cl.array.zeros(self.GPUManager.queue, self.Size, np.int16)
        self.FaceCounter_d = cl.array.zeros(self.GPUManager.queue, 1, np.int32)
        self.Param_d = cl.array.to_device(self.GPUManager.queue, self.res)
        
    def runGPU(self, VolGPU):
        
        self.nb_faces[0] = 0
        self.FaceCounter_d.set(self.nb_faces)
        
        self.MC_Indexing(VolGPU, self.Index_d, self.Offset_d, self.FaceCounter_d, self.iso, self.Size[0], self.Size[1], self.Size[2])
        self.nb_faces = self.FaceCounter_d.get()
        self.nb_vertices[0] = 3*self.nb_faces[0]
        
        print "nb Faces: ", self.nb_faces[0]
        
        self.FacesGPU = cl.array.zeros(self.GPUManager.queue, (self.nb_faces[0], 3), np.int32)
        self.VerticesGPU = cl.array.zeros(self.GPUManager.queue, (self.nb_vertices[0], 3), np.float32)
        
        self.MC(VolGPU, self.Offset_d, self.Index_d, self.VerticesGPU, self.FacesGPU, self.Param_d, self.Size[0], self.Size[1], self.Size[2])
        
        self.GPUManager.queue.finish()
        
        
    '''
        Merge duplicated vertices
    '''
    def MergeVtx(self):
        
#        new_dim = 2
#        while (new_dim < self.nb_vertices[0]):
#            new_dim = new_dim*2
#        
#        diff_dim = new_dim - self.nb_vertices[0]
#        
#        buff = 10000.0*np.ones((diff_dim,3), dtype=np.float32)
#        buff_cl = cl.array.to_device(self.GPUManager.queue, buff)
#        self.VerticesGPU = cl.array.concatenate((self.VerticesGPU, buff_cl))
#        
#        idx_cl = cl.array.arange(self.GPUManager.queue, self.VerticesGPU.shape[0], dtype = np.int)
#        tmp_cl = self.VerticesGPU.copy()
#        self.sorter(tmp_cl[:,0], idx_cl, queue = self.GPUManager.queue)
#        #self.VerticesGPU[:,1] = self.VerticesGPU[idx_cl,1]
#        #self.VerticesGPU[:,2] = self.VerticesGPU[idx_cl,2]
#        self.VerticesGPU = self.VerticesGPU[idx_cl,:]
#        
#        self.VerticesGPU = self.VerticesGPU[0:self.nb_vertices[0], :]
#        
#        print self.VerticesGPU[0:10,:]

        #if (self.nb_faces[0] > 600000):
        #    return
        
        self.Vertices = self.VerticesGPU.get()
        b = np.ascontiguousarray(self.Vertices).view(np.dtype((np.void, self.Vertices.dtype.itemsize * self.Vertices.shape[1])))
        _, idx = np.unique(b, return_index=True)
        unique_Vertices = self.Vertices[idx]
        unique_Vertices = unique_Vertices[unique_Vertices[:,0].argsort()]
        self.nb_vertices[0] = unique_Vertices.shape[0]
        
        #self.Normales = np.zeros((self.nb_vertices[0], 3), dtype = np.float32)
        
#        for i in range(unique_Vertices.shape[0]):
#            print unique_Vertices[i,:]
        
        self.Unique_VerticesGPU = cl.array.to_device(self.GPUManager.queue, unique_Vertices) 
        
        self.OrganizeFaces(self.FacesGPU, self.VerticesGPU, self.Unique_VerticesGPU, self.nb_vertices[0])
        
        # Compute normals
        self.NormalesGPU = cl.array.zeros(self.GPUManager.queue, (self.nb_vertices[0], 3), np.int32) 
        
        self.ComputeNormales(self.FacesGPU, self.Unique_VerticesGPU, self.NormalesGPU)
        
        self.NormalesGPU = self.NormalesGPU.astype(np.float32)/1000000000000.0
        
        self.Normalise(self.NormalesGPU)
        
        self.GPUManager.queue.finish()
        
    '''
        Function to record the created mesh into a .ply file
    '''
    def SaveToPly(self, name):
        
        self.Faces = self.FacesGPU.get()
        self.Vertices = self.Unique_VerticesGPU.get()
        self.Normales = self.NormalesGPU.get()
        
        f = open(name, 'wb')    
        
        # Write headers
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment ply file created by Diego Thomas\n")
        f.write("element vertex %d \n" %(self.nb_vertices[0]))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("element face %d \n" %(self.nb_faces[0]))
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
#        for i in range(self.nb_vertices[0]):
#            f.write("%f %f %f\n" %(self.Vertices[i,0], self.Vertices[i,1], self.Vertices[i,2]))

        # Write vertices
        for i in range(self.nb_vertices[0]):
            f.write("%f %f %f %f %f %f\n" %(self.Vertices[i,0], self.Vertices[i,1], self.Vertices[i,2],
                                            self.Normales[i,0], self.Normales[i,1], self.Normales[i,2]))
            
        # Write the faces
        for i in range(self.nb_faces[0]):
            f.write("3 %d %d %d \n" %(self.Faces[i,0], self.Faces[i,1], self.Faces[i,2])) 
                     
        f.close()
                    
                    
    '''
        Function to draw the mesh using tkinter
    '''
    def DrawMesh(self, Pose, intrinsic, Size, canvas):
        
        self.Faces = self.FacesGPU.get()
        self.Vertices = self.Unique_VerticesGPU.get()
        
        #Draw all faces
        pix = np.array([0., 0., 1.])
        pt = np.array([0., 0., 0., 1.])
        for i in range(self.nb_faces[0]):
            inviewingvolume = False
            poly = []
            for k in range(3):
                pt[0] = self.Vertices[self.Faces[i,k],0]
                pt[1] = self.Vertices[self.Faces[i,k],1]
                pt[2] = self.Vertices[self.Faces[i,k],2]
                pt = np.dot(Pose, pt)
                pix[0] = pt[0]/pt[2]
                pix[1] = pt[1]/pt[2]
                pix = np.dot(intrinsic, pix)
                column_index = int(round(pix[0]))
                line_index = int(round(pix[1]))
                poly.append((column_index, line_index))
                    
                if (column_index > -1 and column_index < Size[1] and line_index > -1 and line_index < Size[0]):
                    inviewingvolume = True
                    
            if inviewingvolume:
                canvas.create_polygon(*poly, fill='white')
                    
                
    '''
        Function to draw the vertices of the mesh using tkinter
    '''
    def DrawPoints(self, Pose, intrinsic, Size, s=1):
        self.Faces = self.FacesGPU.get()
        self.Vertices = self.Unique_VerticesGPU.get()
        self.Normales = self.NormalesGPU.get()
        
        result = np.zeros((Size[0], Size[1], 3), dtype = np.uint8)
        
        pt = np.ones(((np.size(self.Vertices, 0)-1)/s+1, np.size(self.Vertices, 1)+1))
        pt[:,:-1] = self.Vertices[::s, :]
        pt = np.dot(Pose,pt.transpose()).transpose()
        nmle = np.zeros(((np.size(self.Vertices, 0)-1)/s+1, 3), dtype = np.float32)
        nmle[ :, :] = np.dot(Pose[0:3,0:3],self.Normales[ ::s, :].transpose()).transpose()
        
        pix = np.ones(((np.size(self.Vertices, 0)-1)/s+1, np.size(self.Vertices, 1)))
        pix[:,0] = pt[:,0]/pt[:,2]
        pix[:,1] = pt[:,1]/pt[:,2]
        pix = np.dot(intrinsic,pix.transpose()).transpose()
        
        column_index = (np.round(pix[:,0])).astype(int)
        line_index = (np.round(pix[:,1])).astype(int)
        # create matrix that have 0 when the conditions are not verified and 1 otherwise
        cdt_column = (column_index > -1) * (column_index < Size[1]) * pt[:,2] > 0.0
        cdt_line = (line_index > -1) * (line_index < Size[0]) * pt[:,2] > 0.0
        line_index = line_index*cdt_line
        column_index = column_index*cdt_column
        #result[line_index[:], column_index[:]]= 255*np.ones((np.size(pix, 0), 3), dtype = np.uint8)
        result[line_index[:], column_index[:]]= np.dstack( ( (nmle[ :,0]+1.0)*(255./2.), \
                                                              ((nmle[ :,1]+1.0)*(255./2.))*cdt_line, \
                                                              ((nmle[ :,2]+1.0)*(255./2.))*cdt_column ) ).astype(int)
        return result
                    
                    
                    
                    
                    