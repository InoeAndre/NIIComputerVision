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

def in_mat_zero2one(mat):
    """This fonction replace in the matrix all the 0 to 1"""
    mat_tmp = (mat != 0.0)
    res = mat * mat_tmp + ~mat_tmp
    return res

def division_by_norm(mat,norm):
    """This fonction divide a n by m=3 matrix, point by point, by the norm made through the p dimension>
    It ignores division that makes infinite values or overflow to replace it by the former mat values or by 0"""
    for i in range(3):
        with np.errstate(divide='ignore', invalid='ignore'):
            mat[:,i] = np.true_divide(mat[:,i],norm)
            mat[:,i][mat[:,i] == np.inf] = 0
            mat[:,i] = np.nan_to_num(mat[:,i])
    return mat
                
def normalized_cross_prod_optimize(a,b, axs =1):
    #res = np.zeros(a.Size, dtype = "float")
    norm_mat_a = np.sqrt(np.sum(a*a,axis=axs))
    norm_mat_b = np.sqrt(np.sum(b*b,axis=axs))
    #changing every 0 to 1 in the matrix so that the division does not generate nan or infinite values
    norm_mat_a = in_mat_zero2one(norm_mat_a)
    norm_mat_b = in_mat_zero2one(norm_mat_b)
    # compute a/ norm_mat_a
    a = division_by_norm(a,norm_mat_a)
    b = division_by_norm(b,norm_mat_b)
    #compute cross product with matrix
    res = np.cross(a,b)
    #compute the norm of res using the same method for a and b 
    norm_mat_res = np.sqrt(np.sum(res*res,axis=axs))
    norm_mat_res = in_mat_zero2one(norm_mat_res)
    #norm division
    res = division_by_norm(res,norm_mat_res)
    return res



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
        
        self.nb_faces[0] = 0
        cl.enqueue_write_buffer(self.GPUManager.queue, self.FaceCounterGPU, self.nb_faces)
        self.GPUManager.programs['MarchingCubesIndexing'].MarchingCubesIndexing(self.GPUManager.queue, (self.Size[0]-1, self.Size[1]-1), None, \
                                VolGPU, self.OffsetGPU, self.IndexGPU, self.Size_Volume, np.int32(self.iso), self.FaceCounterGPU)
        
        
        cl.enqueue_read_buffer(self.GPUManager.queue, self.FaceCounterGPU, self.nb_faces).wait()
        print "nb Faces: ", self.nb_faces[0]
        
        self.Faces = np.zeros((self.nb_faces[0], 3), dtype = np.int32)
        self.Vertices = np.zeros((3*self.nb_faces[0], 3), dtype = np.float32)
        self.Normals = np.zeros((3*self.nb_faces[0], 3), dtype = np.float32)
        
        self.FacesGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Faces.nbytes)
        self.VerticesGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Vertices.nbytes)
        #self.NormalsGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Normals.nbytes)
        
        self.GPUManager.programs['MarchingCubes'].MarchingCubes(self.GPUManager.queue, (self.Size[0]-1, self.Size[1]-1), None, \
                                VolGPU, self.OffsetGPU, self.IndexGPU, self.VerticesGPU, self.FacesGPU, self.ParamGPU, self.Size_Volume)
        
        cl.enqueue_read_buffer(self.GPUManager.queue, self.VerticesGPU, self.Vertices).wait()
        cl.enqueue_read_buffer(self.GPUManager.queue, self.FacesGPU, self.Faces).wait()
        #cl.enqueue_read_buffer(self.GPUManager.queue, self.NormalsGPU, self.Normals).wait()
        self.ComputeMCNmls()


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
    def DrawPoints(self, Pose, intrinsic, Size,background,s=1):
        #result = np.zeros((Size[0], Size[1], 3), dtype = np.uint8)
        result = background.astype(np.uint8)
        
        pt = np.ones(((np.size(self.Vertices, 0)-1)/s+1, np.size(self.Vertices, 1)+1))
        pt[:,:-1] = self.Vertices[::s, :]
        pt = np.dot(Pose,pt.transpose()).transpose()
        #nmle = np.zeros((self.Size[0], self.Size[1],self.Size[2]), dtype = np.float32)
        #nmle[ ::s, ::s,:] = np.dot(Pose[0:3,0:3],self.Nmls[ ::s, ::s,:].transpose(0,2,1)).transpose(1,2,0)
        
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
        result[line_index[:], column_index[:]]= 255*np.ones((np.size(pix, 0), 3), dtype = np.uint8)
        return result
                    
                    
                    
    '''
        Function to compute the normals of the mesh 
    '''
    def ComputeMCNmls(self):
        
        # instanciation
        nb_faces = self.nb_faces[0]
        vectsFaces = np.zeros((2,nb_faces, 3), dtype = np.float)
        facesNmls = np.zeros((nb_faces, 3), dtype = np.float)
        
        # edges of triangles
        vectsFaces[0,:,:] = self.Vertices[self.Faces[:,1]]-self.Vertices[self.Faces[:,0]]   
        vectsFaces[1,:,:] = self.Vertices[self.Faces[:,2]]-self.Vertices[self.Faces[:,0]]

        # compute each face's normal
        #facesNmls = np.cross(vectsFaces[0,:,:],vectsFaces[1,:,:])
        facesNmls[:,0] = vectsFaces[0,:,1]*vectsFaces[1,:,2]- vectsFaces[0,:,2]*vectsFaces[1,:,1]
        facesNmls[:,1] = vectsFaces[0,:,2]*vectsFaces[1,:,0]- vectsFaces[0,:,0]*vectsFaces[1,:,2]
        facesNmls[:,2] = vectsFaces[0,:,0]*vectsFaces[1,:,1]- vectsFaces[0,:,1]*vectsFaces[1,:,0]
        self.Normals[self.Faces[:,0]] += facesNmls
        self.Normals[self.Faces[:,1]] += facesNmls
        self.Normals[self.Faces[:,2]] += facesNmls

        
        # compute the norm of nmlsSum
        norm_nmlsSum = np.sqrt(np.sum(self.Normals*self.Normals,axis=1))
        norm_nmlsSum = in_mat_zero2one(norm_nmlsSum)
        # normalize the mean of the norm
        self.Normals = division_by_norm(self.Normals,norm_nmlsSum)

        '''
        Non optimized version for understanding
        # instanciation
        nb_faces = self.nb_faces[0]
        vectsFaces = np.zeros((2, 3), dtype = np.int32)
        
        for f in range(nb_faces):
            # vectors of triangles
            vectsFaces[0,:] = self.Vertices[self.Faces[f,1]]-self.Vertices[self.Faces[f,0]]
            vectsFaces[1,:] = self.Vertices[self.Faces[f,2]]-self.Vertices[self.Faces[f,0]]
    
            # compute each face's normal
            facesNmls = np.cross(vectsFaces[1,:],vectsFaces[0,:])
            for s in range(3):
                # sum of normals
                self.Normals[self.Faces[f,s]] += facesNmls
               
        for v in range(self.Normals.shape[0]):
            # compute the norm of normals
            norm_nmlsSum = np.sqrt(np.sum(self.Normals[v]*self.Normals[v]))
            if norm_nmlsSum == 0:
                continue;
            # normalize the normals
            self.Normals[v] = self.Normals[v]/norm_nmlsSum
        '''


        
        
        
        