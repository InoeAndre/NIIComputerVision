# File created by Diego Thomas the 21-11-2016
# Second Author Inoe AMDRE

# File to handle program main loop
import sys
import cv2
from math import *
import numpy as np
from numpy import linalg as LA
from numpy.matlib import rand,zeros,ones,empty,eye
import Tkinter as tk
import tkMessageBox
from tkFileDialog import askdirectory
from PIL import Image, ImageTk
import imp
import scipy.io
import time
import random
import pyopencl as cl
import pandas



RGBD = imp.load_source('RGBD', './lib/RGBD.py')
RGBDimg = imp.load_source('RGBDimg', './lib/RGBDimg.py')
TrackManager = imp.load_source('TrackManager', './lib/tracking.py')
TSDFtk = imp.load_source('TSDFtk', './lib/TSDF.py')
GPU = imp.load_source('GPUManager', './lib/GPUManager.py')
My_MC = imp.load_source('My_MarchingCube', './lib/My_MarchingCube.py')

def in_mat_zero2one(mat):
    """This fonction replace in the matrix all the 0 to 1"""
    mat_tmp = (mat != 0.0)
    res = mat * mat_tmp + ~mat_tmp
    return res

class Application(tk.Frame):
    ## Function to handle keyboard inputs
    def key(self, event):
        Transfo = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        
        if (event.keysym == 'Escape'):
            self.root.destroy()
        if (event.keysym == 'd'):
            Transfo[0,3] = -0.1
        if (event.keysym == 'a'):
            Transfo[0,3] = 0.1
        if (event.keysym == 'w'):
            Transfo[1,3] = 0.1
        if (event.keysym == 's'):
            Transfo[1,3] = -0.1
        if (event.keysym == 'e'):
            Transfo[2,3] = 0.1
        if (event.keysym == 'q'):
            Transfo[2,3] = -0.1
        if (event.keysym == 'c'):
            self.color_tag = (self.color_tag+1) %2

        if (event.keysym != 'Escape'):
            self.Pose = np.dot(self.Pose, Transfo)
            rendering =np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
            rendering = self.RGBD2.Draw_optimize(rendering,self.Pose, self.w.get(), self.color_tag)
            rendering = self.RGBD.DrawMesh(rendering, self.MC.Vertices,self.MC.Normals,self.Pose, 1, self.color_tag)
            img = Image.fromarray(rendering, 'RGB')
            self.imgTk=ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgTk)
            #self.DrawCenters2D(self.Pose)
            #self.DrawSys2D(self.Pose)
            #self.DrawOBBox2D(self.Pose)


    ## Function to handle mouse press event
    def mouse_press(self, event):
        self.x_init = event.x
        self.y_init = event.y
    
    ## Function to handle mouse release events
    def mouse_release(self, event):
        x = event.x
        y = event.y
    
    
    ## Function to handle mouse motion events
    def mouse_motion(self, event):
        if (event.y < 480):
            delta_x = event.x - self.x_init
            delta_y = event.y - self.y_init
            
            angley = 0.
            if (delta_x > 0.):
                angley = -0.01
            elif (delta_x < 0.):
                angley = 0.01 #pi * 2. * delta_x / float(self.Size[0])
            RotY = np.array([[cos(angley), 0., sin(angley), 0.], \
                             [0., 1., 0., 0.], \
                             [-sin(angley), 0., cos(angley), 0.], \
                             [0., 0., 0., 1.]])
            self.Pose = np.dot(self.Pose, RotY)
            
            anglex = 0.
            if (delta_y > 0.):
                anglex = 0.01
            elif (delta_y < 0.):
                anglex = -0.01 # pi * 2. * delta_y / float(self.Size[0])
            RotX = np.array([[1., 0., 0., 0.], \
                            [0., cos(anglex), -sin(anglex), 0.], \
                            [0., sin(anglex), cos(anglex), 0.], \
                            [0., 0., 0., 1.]])

            self.Pose = np.dot(self.Pose, RotX)
            rendering =np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
            rendering = self.RGBD2.Draw_optimize(rendering,self.Pose, self.w.get(), self.color_tag)
            rendering = self.RGBD.DrawMesh(rendering, self.MC.Vertices,self.MC.Normals,self.Pose, 1, self.color_tag)
            img = Image.fromarray(rendering, 'RGB')
            self.imgTk=ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgTk)
            #self.DrawCenters2D(self.Pose)
            #self.DrawSys2D(self.Pose)
            #self.DrawOBBox2D(self.Pose)
       
        self.x_init = event.x
        self.y_init = event.y

    def DrawPoint2D(self,point,radius,color):
        if point[0]>0 and point[1]>0:
            x1, y1 = (point[0] - radius), (point[1] - radius)
            x2, y2 = (point[0] + radius), (point[1] + radius)
        else:
            x1, y1 = (point[0]), (point[1])
            x2, y2 = (point[0]), (point[1]) 
        self.canvas.create_oval(x1, y1, x2, y2, fill=color)


    def DrawColors2D(self,RGBD,img,Pose):
        '''this function draw the color of each segmented part of the body'''
        newImg = img.copy()
        Txy = RGBD.transCrop
        label = RGBD.labels
        for k in range(1,RGBD.bdyPart.shape[0]+1):
            color = RGBD.bdyColor[k-1]
            for i in range(Txy[1],Txy[3]):
                for j in range(Txy[0],Txy[2]):
                    if label[i][j]==k :
                        newImg[i,j] = color
                    else :
                        newImg[i,j] = newImg[i,j] 
        return newImg               

                      
    def DrawSkeleton2D(self,Pose):
        '''this function draw the Skeleton of a human and make connections between each part'''
        pos = self.pos2d[0][self.Index]
        for i in range(np.size(self.connection,0)): 
            pt1 = (pos[self.connection[i,0]-1,0],pos[self.connection[i,0]-1,1])
            pt2 = (pos[self.connection[i,1]-1,0],pos[self.connection[i,1]-1,1])
            radius = 1
            color = "blue"        
            self.DrawPoint2D(pt1,radius,color)
            self.DrawPoint2D(pt2,radius,color)      
            self.canvas.create_line(pt1[0],pt1[1],pt2[0],pt2[1],fill="red")

    def DrawCenters2D(self,Pose,s=1):
        '''this function draw the center of each oriented coordinates system for each body part''' 
        self.ctr2D = self.RGBD.GetProjPts2D_optimize(self.RGBD.ctr3D,Pose)        
        for i in range( len(self.RGBD.ctr3D)):
            c = self.ctr2D[i]
            self.DrawPoint2D(c,2,"yellow")

    def DrawSys2D(self,Pose):
        '''this function draw the sys of oriented coordinates system for each body part''' 
        self.RGBD.GetNewSys(Pose,self.ctr2D,10)
        for i in range(len(self.ctr2D)):
            c = self.ctr2D[i]
            pt0 = self.RGBD.drawNewSys[i][0]
            pt1 = self.RGBD.drawNewSys[i][1]
            pt2 = self.RGBD.drawNewSys[i][2]    
            self.canvas.create_line(pt0[0],pt0[1],c[0],c[1],fill="gray",width = 2)
            self.canvas.create_line(pt1[0],pt1[1],c[0],c[1],fill="gray",width = 2)
            self.canvas.create_line(pt2[0],pt2[1],c[0],c[1],fill="gray",width = 2)

    def DrawOBBox2D(self,Pose):
        '''Draw in the canvas the oriented bounding boxes for each body part''' 
        self.OBBcoords2D = []
        for i in range(len(self.RGBD.coords)):
            self.OBBcoords2D.append(self.RGBD.GetProjPts2D_optimize(self.RGBD.coords[i],Pose))
            pt = self.OBBcoords2D[i]
            #print pt
            for j in range(3):
                self.canvas.create_line(pt[j][0],pt[j][1],pt[j+1][0],pt[j+1][1],fill="red",width =2)
                self.canvas.create_line(pt[j+4][0],pt[j+4][1],pt[j+5][0],pt[j+5][1],fill="red",width = 2)
                self.canvas.create_line(pt[j][0],pt[j][1],pt[j+4][0],pt[j+4][1],fill="red",width = 2)
            self.canvas.create_line(pt[3][0],pt[3][1],pt[0][0],pt[0][1],fill="red",width = 2)
            self.canvas.create_line(pt[7][0],pt[7][1],pt[4][0],pt[4][1],fill="red",width = 2)
            self.canvas.create_line(pt[3][0],pt[3][1],pt[7][0],pt[7][1],fill="red",width = 2)
            for j in range(8):
                self.DrawPoint2D(pt[j],2,"black")

    def DrawMesh2D(self,Pose,vertex,triangle):
        '''Draw in the canvas the triangles of the Mesh in 2D''' 
        python_green = "#476042"
        for i in range(triangle.shape[0]):
            pt0 = vertex[triangle[i][0]]
            pt1 = vertex[triangle[i][1]]
            pt2 = vertex[triangle[i][2]]
            self.canvas.create_polygon(pt0[0],pt0[1],pt1[0],pt1[1],pt2[0],pt2[1],outline = python_green, fill='yellow', width=1)


    def CheckVerts2D(self,verts):
        '''Change the indexes values that are outside the frame''' 
        #make sure there are not false values
        cdt_line = (verts[:,1] > -1) * (verts[:,1] < self.Size[0])
        cdt_column = (verts[:,0] > -1) * (verts[:,0] < self.Size[1])
        verts[:,0] = verts[:,0]*cdt_column
        verts[:,1] = verts[:,1]*cdt_line
        return verts     
       
    def InvPose(self,Pose):
        '''Compute the inverse transform of Pose''' 
        PoseInv = np.zeros(Pose.shape,Pose.dtype)
        PoseInv[0:3,0:3] = LA.inv(Pose[0:3,0:3])
        PoseInv[0:3,3] = -np.dot(PoseInv[0:3,0:3],Pose[0:3,3])
        PoseInv[3,3] = 1.0
        return PoseInv
      

    ## Constructor function
    def __init__(self, path,  GPUManager, master=None):
        self.root = master
        self.path = path
        self.GPUManager = GPUManager
        self.draw_bump = False
        self.draw_spline = False

        tk.Frame.__init__(self, master)
        self.pack()
        

        '''
        TUM's data
        '''
        
        '''
        #general parameters
        path = "/Users/nii-user/inoe/data/TechnischeUniversitatMunchen"
        calib = "/fr1_rgb_calibration/ost.txt"
        #calib2 = "/fr2_rgb_calibration/ost.txt"
        calib_file = open(path + calib, 'r')
        calib_data = calib_file.readlines()       
        # fr1_rgb_calibration intrinsic
        self.intrinsic = np.array([[calib_data[14].split()[0], calib_data[14].split()[1], calib_data[14].split()[2]], \
                                    [calib_data[15].split()[0], calib_data[15].split()[1], calib_data[15].split()[2]], \
                                    [calib_data[16].split()[0], calib_data[16].split()[1], calib_data[16].split()[2]]], dtype = np.float32)
        self.color_tag = 1
        self.Size = [480,640]

        print self.intrinsic


        self.fact = 5000.0
        
        self.Pose = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        self.PoseTrack = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        Id4 = np.identity(4)   
        

        dataset = "/rgbd_dataset_freiburg1_xyz/"
        name = "depth.txt"
        with open(path+dataset+name, "r") as f:
            data = f.readlines()
            #print data[3].split()[1]
            
            # previous image
            first = 3
            img = cv2.imread(path + dataset + data[first].split()[1],cv2.CV_LOAD_IMAGE_UNCHANGED)
            PrevDepth = np.asarray( img[:,:], np.float32)
            print "np.max(PrevDepth)"
            print np.max(PrevDepth)
            print "np.min(PrevDepth)"
            print np.min(PrevDepth)
            print PrevDepth.shape 
            
            
            # depth conversion
            self.RGBD = RGBDimg.RGBD(PrevDepth, self.intrinsic, self.fact,self.Size)
            self.RGBD.BilateralFilter(-1, 0.02, 3) 
            self.RGBD.Vmap_optimize()   
            self.RGBD.NMap_optimize() 
            
            self.FirstRGBD = RGBDimg.RGBD(PrevDepth, self.intrinsic, self.fact,self.Size)
            self.FirstRGBD.BilateralFilter(-1, 0.02, 3) 
            self.FirstRGBD.Vmap_optimize()   
            self.FirstRGBD.NMap_optimize() 
            
            # For global Fusion, init TSDF, Marching and tracking
            mf = cl.mem_flags
            self.TSDF = np.zeros((512,512,512), dtype = np.int16)
            self.TSDFGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.TSDF.nbytes)
            self.Weight = np.zeros((512,512,512), dtype = np.int16)
            self.WeightGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Weight.nbytes)
            
            TSDFManager = TSDFtk.TSDFManager((512,512,512), self.RGBD, self.GPUManager,self.TSDFGPU,self.WeightGPU) 
            self.MC = My_MC.My_MarchingCube(TSDFManager.Size, TSDFManager.res, 0.0, self.GPUManager)
            Tracker = TrackManager.Tracker(0.001, 0.5, 1, [10], 0.001) 
    
            # TSDF Fusion
            TSDFManager.FuseRGBD_GPU(self.RGBD, self.PoseTrack)  
            # Marching cubes
            self.MC.runGPU(TSDFManager.TSDFGPU)    
            #self.MC.SaveToPly("result0.ply")
            end = 60
            for index in range(first,end):
                # time counter
                start_time2 = time.time() 
                #current image, new Image
                img2 = cv2.imread(path + dataset + data[index+1].split()[1],cv2.CV_LOAD_IMAGE_UNCHANGED)
                NewDepth = np.asarray( img2[:,:], np.float32 )
                print NewDepth.shape 
                self.NewRGBD = RGBDimg.RGBD(NewDepth, self.intrinsic, self.fact,self.Size)
                #self.NewRGBD.BilateralFilter(-1, 0.02, 3)
                self.NewRGBD.Vmap_optimize()
                self.NewRGBD.NMap_optimize()
                
                
                T_Pose = Tracker.RegisterRGBDMesh_optimize(self.NewRGBD, self.MC.Vertices,self.MC.Normales, self.PoseTrack)#self.MC.

                for k in range(4):
                    for l in range(4):
                        self.PoseTrack[k,l] = T_Pose[k,l]             
                print 'self.PoseTrack'
                print self.PoseTrack
                
                #if index !=end-1:
                #TSDF Fusion
                TSDFManager.FuseRGBD_GPU(self.NewRGBD, self.PoseTrack)   
                # Mesh rendering
                self.MC.runGPU(TSDFManager.TSDFGPU)           
                
                elapsed_time = time.time() - start_time2
                print "Image number %d done : %f s" % (index,elapsed_time)

            #self.MC.runGPU(TSDFManager.TSDFGPU)
            start_time3 = time.time()
            self.MC.SaveToPly("result.ply")
            elapsed_time = time.time() - start_time3
            print "SaveToPly: %f" % (elapsed_time)
            
    
            # projection in 2d space to draw it
            rendering =np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
            # projection of the current image/ Overlay
            rendering = self.NewRGBD.Draw_optimize(rendering,Id4, 1, self.color_tag)
            # Projection directly with the output of the marching cubes  
            rendering = self.RGBD.DrawMesh(rendering, self.MC.Vertices,self.MC.Normales,self.PoseTrack, 1, self.color_tag)
    
            # Show figure and images
                
            # 3D reconstruction of the whole image
            self.canvas = tk.Canvas(self, bg="black", height=self.Size[0], width=self.Size[1])
            self.canvas.pack()        
            img = Image.fromarray(rendering, 'RGB')
            self.imgTk=ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgTk)        

        f.close()
        '''



        '''
        NGuyen's data
        This require to put in TSDf: 
            self.dim_x = self.Size[0]/5.0
            self.dim_y = self.Size[1]/5.0
            self.dim_z = self.Size[2]/5.0
        '''
        
        
        self.color_tag = 1
        calib_file = open(self.path + '/Calib.txt', 'r')
        calib_data = calib_file.readlines()
        self.Size = [int(calib_data[0]), int(calib_data[1])]
        self.intrinsic = np.array([[float(calib_data[2]), float(calib_data[3]), float(calib_data[4])], \
                                   [float(calib_data[5]), float(calib_data[6]), float(calib_data[7])], \
                                   [float(calib_data[8]), float(calib_data[9]), float(calib_data[10])]], dtype = np.float32)
    
        print self.intrinsic

        self.fact = 1000.0

        mat = scipy.io.loadmat(self.path + '/String4b.mat')
        self.lImages = mat['DepthImg']
        self.pos2d = mat['Pos2D']
        #self.bdyIdx = mat['BodyIndex']
        self.bdyIdx = 0

        connectionMat = scipy.io.loadmat(self.path + '/SkeletonConnectionMap.mat')
        self.connection = connectionMat['SkeletonConnectionMap']
        self.Pose = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        self.T_Pose = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        Id4 = np.identity(4)
        
        
        # Current Depth Image (i.e: i)
        start_time = time.time()
        self.RGBD = RGBD.RGBD(self.path + '/Depth.tiff', self.path + '/RGB.tiff', self.intrinsic, self.fact)
        self.RGBD.LoadMat(self.lImages,self.pos2d,self.connection,self.bdyIdx )   
        self.Index = 0
        self.RGBD.ReadFromMat(self.Index) 
        self.RGBD.BilateralFilter(-1, 0.02, 3) 
        #self.RGBD.Crop2Body() 
        #self.RGBD.BodySegmentation() 
        #self.RGBD.BodyLabelling()         
        #self.RGBD.depth_image *= (self.RGBD.labels >0) 
        self.RGBD.Vmap_optimize()   
        self.RGBD.NMap_optimize()  
        #self.RGBD.myPCA()
        elapsed_time = time.time() - start_time
        print "depth conversion: %f s" % (elapsed_time)
        
        self.RGBD2 = RGBD.RGBD(self.path + '/Depth.tiff', self.path + '/RGB.tiff', self.intrinsic, self.fact) 
        self.RGBD2.LoadMat(self.lImages,self.pos2d,self.connection,self.bdyIdx ) 
#==============================================================================
#         self.RGBD2.ReadFromMat(self.Index) 
#         self.RGBD2.BilateralFilter(-1, 0.02, 3) 
#         #self.RGBD2.Crop2Body() 
#         #self.RGBD2.BodySegmentation() 
#         #self.RGBD2.BodyLabelling()         
#         #self.RGBD2.depth_image *= (self.RGBD2.labels >0) 
#         self.RGBD2.Vmap_optimize()   
#         self.RGBD2.NMap_optimize()          
#==============================================================================
        
        # For global Fusion
        mf = cl.mem_flags
        self.TSDF = np.zeros((512,512,512), dtype = np.int16)
        self.TSDFGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.TSDF.nbytes)
        self.Weight = np.zeros((512,512,512), dtype = np.int16)
        self.WeightGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Weight.nbytes)
        
        TSDFManager = TSDFtk.TSDFManager((512,512,512), self.RGBD, self.GPUManager,self.TSDFGPU,self.WeightGPU) 
        self.MC = My_MC.My_MarchingCube(TSDFManager.Size, TSDFManager.res, 0.0, self.GPUManager)
        Tracker = TrackManager.Tracker(0.01, 0.5, 1, [10], 0.001)
        
        #self.RGBD.VtxToPly("pointCloud.ply",Vertices,Normales)
        # TSDF Fusion
        TSDFManager.FuseRGBD_GPU(self.RGBD, self.Pose)  
        self.MC.runGPU(TSDFManager.TSDFGPU)    
        end =20
        for i in range(self.Index+1,end):
            start_time2 = time.time() 
            #depthMap conversion of the new image
            self.RGBD2.ReadFromMat(i) 
            self.RGBD2.BilateralFilter(-1, 0.02, 3) 
            #self.RGBD2.Crop2Body() 
            #self.RGBD2.BodySegmentation() 
            #self.RGBD2.BodyLabelling()         
            #self.RGBD2.depth_image *= (self.RGBD2.labels >0) 
            self.RGBD2.Vmap_optimize()   
            self.RGBD2.NMap_optimize()  
            #self.RGBD2.myPCA()
            
            # New pose estimation
            NewPose = Tracker.RegisterRGBDMesh_optimize(self.RGBD2,self.MC.Vertices,self.MC.Normales, self.T_Pose)
            for k in range(4):
                for l in range(4):
                    self.T_Pose[k,l] = NewPose[k,l]
            print 'self.T_Pose'
            print self.T_Pose          
            
            #TSDF Fusion
            TSDFManager.FuseRGBD_GPU(self.RGBD2, self.T_Pose)   
            # Mesh rendering
            self.MC.runGPU(TSDFManager.TSDFGPU)           

            elapsed_time = time.time() - start_time2
            print "Image number %d done : %f s" % (i,elapsed_time)
            

            
        start_time3 = time.time()
        self.MC.SaveToPly("result.ply")
        elapsed_time = time.time() - start_time3
        print "SaveToPly: %f" % (elapsed_time)

        # projection in 2d space to draw it
        rendering =np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
        # projection of the current image/ Overlay
        rendering = self.RGBD2.Draw_optimize(rendering,Id4, 1, self.color_tag)
        # Projection directly with the output of the marching cubes  
        rendering = self.RGBD.DrawMesh(rendering, self.MC.Vertices,self.MC.Normales,self.T_Pose, 1, self.color_tag)
        

        # Show figure and images
            
        # 3D reconstruction of the whole image
        self.canvas = tk.Canvas(self, bg="black", height=self.Size[0], width=self.Size[1])
        self.canvas.pack()        
        #rendering = self.DrawColors2D(self.RGBD2,rendering,self.Pose)
        img = Image.fromarray(rendering, 'RGB')
        self.imgTk=ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgTk)
        #self.DrawSkeleton2D(self.Pose)
        #self.DrawCenters2D(self.Pose)
        #self.DrawSys2D(self.Pose)
        #self.DrawOBBox2D(self.Pose)
        #self.DrawMesh2D(self.Pose,self.verts,self.faces)
        

        #enable keyboard and mouse monitoring
        self.root.bind("<Key>", self.key)
        self.root.bind("<Button-1>", self.mouse_press)
        self.root.bind("<ButtonRelease-1>", self.mouse_release)
        self.root.bind("<B1-Motion>", self.mouse_motion)

        self.w = tk.Scale(master, from_=1, to=10, orient=tk.HORIZONTAL)
        self.w.pack()


            
#==============================================================================
#         from mayavi import mlab 
#         mlab.triangular_mesh([vert[0] for vert in self.MC.Vertices],\
#                              [vert[1] for vert in self.MC.Vertices],\
#                              [vert[2] for vert in self.MC.Vertices],self.MC.Faces) 
#         mlab.show()
#==============================================================================




