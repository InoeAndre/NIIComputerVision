#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 13:38:34 2017

@author: diegothomas
"""


import imp
import numpy as np
import pyopencl as cl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel
import math

GPU = imp.load_source('GPUManager', './lib/GPUManager.py')
KernelsOpenCL = imp.load_source('MarchingCubes_KernelOpenCL', './lib/MarchingCubes_KernelOpenCL.py')

mf = cl.mem_flags

class My_MarchingCube():
    
    def __init__(self, Size, Res, Iso, GPUManager):
        self.Size = Size
        self.res = Res
        self.iso = Iso
        self.nb_faces = np.array([0], dtype = np.int32)
        self.nb_vertices = np.array([0], dtype = np.int32)
        
        self.GPUManager = GPUManager
        
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
        
        self.Offset_d = cl.array.zeros(self.GPUManager.queue, self.Size, np.int32)
        self.Index_d = cl.array.zeros(self.GPUManager.queue, self.Size, np.int16)
        self.FaceCounter_d = cl.array.zeros(self.GPUManager.queue, 1, np.int32)
        self.Param_d = cl.array.to_device(self.GPUManager.queue, self.res)
        
        
        self.GPUManager.programs['MarchingCubesIndexing'] = cl.Program(self.GPUManager.context, KernelsOpenCL.Kernel_G_MarchingCubeIndexing).build()
        self.GPUManager.programs['MarchingCubes'] = cl.Program(self.GPUManager.context, KernelsOpenCL.Kernel_G_MarchingCube).build()
        self.GPUManager.programs['MergeVtx'] = cl.Program(self.GPUManager.context, KernelsOpenCL.Kernel_MergeVtx).build()
        self.GPUManager.programs['SimplifyMesh'] = cl.Program(self.GPUManager.context, KernelsOpenCL.Kernel_SimplifyMesh).build()
        self.GPUManager.programs['InitArray'] = cl.Program(self.GPUManager.context, KernelsOpenCL.Kernel_InitArray).build()
        
        
        self.Size_Volume = cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, \
                               hostbuf = np.array([self.Size[0], self.Size[1], self.Size[2]], dtype = np.int32))
        
        self.OffsetGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Size[0]*self.Size[1]*self.Size[2]*4)
        self.IndexGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Size[0]*self.Size[1]*self.Size[2]*4)
        self.FaceCounterGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = self.nb_faces)
        #self.VertexCounterGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = self.nb_vertices)
        
        self.ParamGPU = cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = self.res)
        
        self.Array_x_GPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Size[0]*self.Size[1]*self.Size[2]*2)
        self.Array_y_GPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Size[0]*self.Size[1]*self.Size[2]*2)
        self.Array_z_GPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Size[0]*self.Size[1]*self.Size[2]*2)
        self.Normales_x_GPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Size[0]*self.Size[1]*self.Size[2]*2)
        self.Normales_y_GPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Size[0]*self.Size[1]*self.Size[2]*2)
        self.Normales_z_GPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Size[0]*self.Size[1]*self.Size[2]*2)
        self.Weights_GPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Size[0]*self.Size[1]*self.Size[2]*2)
        self.VtxInd_GPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Size[0]*self.Size[1]*self.Size[2]*2)
        
        
    def runGPU(self, VolGPU):
        self.nb_faces[0] = 0
        self.FaceCounter_d.set(self.nb_faces)
        
        self.MC_Indexing(VolGPU, self.Index_d, self.Offset_d, self.FaceCounter_d, self.iso, self.Size[0], self.Size[1], self.Size[2])
        self.nb_faces = self.FaceCounter_d.get()
        self.nb_vertices[0] = 3*self.nb_faces[0]
        
        print "nb Faces: ", self.nb_faces[0]
        
        self.FacesGPU = cl.array.zeros(self.GPUManager.queue, (self.nb_faces[0], 3), np.int32) #cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.nb_faces[0]*3*4) #int32 = 4 bytes
        self.VerticesGPU = cl.array.zeros(self.GPUManager.queue, (self.nb_vertices[0], 3), np.float32) #cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.nb_faces[0]*9*4)
        
        self.MC(VolGPU, self.Offset_d, self.Index_d, self.VerticesGPU, self.FacesGPU, self.Param_d, self.Size[0], self.Size[1], self.Size[2])
       
#        self.Vertices = self.VerticesGPU.get()
#        self.Normales = np.zeros((self.nb_vertices[0], 3), dtype = np.float32)
#        self.Faces = self.FacesGPU.get()
        
#        self.nb_faces[0] = 0
#        
#        self.GPUManager.programs['MarchingCubesIndexing'].MarchingCubesIndexing(self.GPUManager.queue, (self.Size[0]-1, self.Size[1]-1), None, \
#                                VolGPU.data, self.OffsetGPU, self.IndexGPU, self.Size_Volume, np.int32(self.iso), self.FaceCounterGPU)
#        
#        cl.enqueue_read_buffer(self.GPUManager.queue, self.FaceCounterGPU, self.nb_faces).wait()
#        print "nb Faces: ", self.nb_faces[0]
#        
#        self.FacesGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.nb_faces[0]*3*4) #int32 = 4 bytes
#        self.VerticesGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.nb_faces[0]*9*4)
#        
#        self.GPUManager.programs['MarchingCubes'].MarchingCubes(self.GPUManager.queue, (self.Size[0]-1, self.Size[1]-1), None, \
#                                VolGPU, self.OffsetGPU, self.IndexGPU, self.VerticesGPU, self.FacesGPU, self.ParamGPU, self.Size_Volume)

    '''
        Merge duplicated vertices
    '''
    def MergeVtxGPU(self):
        
        self.nb_vertices[0] = 0
        
        dim_x = self.nb_faces[0]
        dim_y = 1
        stride = 1
        if (self.nb_faces[0] > stride):
            dim_x = self.nb_faces[0]/stride +1
            dim_y = stride
            

        self.GPUManager.programs['InitArray'].InitArray(self.GPUManager.queue, (dim_x, dim_y), None, \
                                self.Array_x_GPU, self.Array_y_GPU, self.Array_z_GPU, self.Weights_GPU, 
                                self.Normales_x_GPU, self.Normales_y_GPU, self.Normales_z_GPU,
                                self.VerticesGPU, self.ParamGPU, self.Size_Volume, np.int32(self.nb_faces[0]))
        

        self.GPUManager.programs['MergeVtx'].MergeVtx(self.GPUManager.queue, (dim_x, dim_y), None, \
                                self.Array_x_GPU, self.Array_y_GPU, self.Array_z_GPU, self.Weights_GPU, 
                                self.Normales_x_GPU, self.Normales_y_GPU, self.Normales_z_GPU, self.VtxInd_GPU,
                                self.VerticesGPU, self.FacesGPU, self.ParamGPU, self.Size_Volume, self.VertexCounterGPU, np.int32(self.nb_faces[0]))
        
        cl.enqueue_read_buffer(self.GPUManager.queue, self.VertexCounterGPU, self.nb_vertices).wait()
        print "nb Vertices: ", self.nb_vertices[0]
        
        self.Vertices = np.zeros((self.nb_vertices[0], 3), dtype = np.float32)
        self.VerticesGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Vertices.nbytes)
        
        self.Normales = np.zeros((self.nb_vertices[0], 3), dtype = np.float32)
        self.NormalesGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Normales.nbytes)
        
        self.GPUManager.programs['SimplifyMesh'].SimplifyMesh(self.GPUManager.queue, (dim_x, dim_y), None, \
                                self.Array_x_GPU, self.Array_y_GPU, self.Array_z_GPU, self.Weights_GPU, 
                                self.Normales_x_GPU, self.Normales_y_GPU, self.Normales_z_GPU, self.VtxInd_GPU,
                                self.VerticesGPU, self.NormalesGPU, self.FacesGPU, self.Size_Volume, np.int32(self.nb_faces[0]))
        
        cl.enqueue_read_buffer(self.GPUManager.queue, self.VerticesGPU, self.Vertices)
        cl.enqueue_read_buffer(self.GPUManager.queue, self.NormalesGPU, self.Normales)
        
        self.Faces = np.zeros((self.nb_faces[0], 3), dtype = np.int32)
        cl.enqueue_read_buffer(self.GPUManager.queue, self.FacesGPU, self.Faces).wait()
        
        
    
    
    def MergeVtx(self):
        VtxArray_x = np.zeros(self.Size)
        VtxArray_y = np.zeros(self.Size)
        VtxArray_z = np.zeros(self.Size)
        VtxWeights = np.zeros(self.Size)
        VtxIdx = np.zeros(self.Size)
        FacesIdx = np.zeros((self.nb_faces[0], 3, 3), dtype = np.uint16)
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
            print i
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
            self.Normales[i, :] = self.Normales[i, :]/mag
        
        
    '''
        Function to record the created mesh into a .ply file
    '''
    def SaveToPly(self, name):
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
                    
                    
                    
                    
                    
