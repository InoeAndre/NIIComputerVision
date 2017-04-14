# File created by Diego Thomas the 21-11-2016

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

RGBD = imp.load_source('RGBD', './lib/RGBD.py')
TrackManager = imp.load_source('TrackManager', './lib/tracking.py')
TSDFtk = imp.load_source('TSDFtk', './lib/TSDF.py')

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
            rendering = self.RGBD.Draw_optimize(self.Pose, self.w.get(), self.color_tag)
            img = Image.fromarray(rendering, 'RGB')
            self.imgTk=ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgTk)


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
            
            rendering = self.RGBD.Draw_optimize(self.Pose, self.w.get(), self.color_tag)
            img = Image.fromarray(rendering, 'RGB')
            self.imgTk=ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgTk)

       
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


    def DrawColors2D(self,img,Pose):
        '''this function draw the color of each segmented part of the body'''
        newImg = img.copy()
        Txy = self.RGBD.transCrop
        label = self.RGBD.labels
        for k in range(1,self.RGBD.bdyPart.shape[0]+1):
            color = self.RGBD.bdyColor[k-1]
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
        self.ctr2D = self.RGBD.GetProjPts2D(self.RGBD.ctr3D,Pose)        
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
            self.OBBcoords2D.append(self.RGBD.GetProjPts2D(self.RGBD.coords[i],Pose))
            pt = self.OBBcoords2D[i]
            print pt
            for j in range(3):
                self.canvas.create_line(pt[j][0],pt[j][1],pt[j+1][0],pt[j+1][1],fill="red",width =2)
                self.canvas.create_line(pt[j+4][0],pt[j+4][1],pt[j+5][0],pt[j+5][1],fill="red",width = 2)
                self.canvas.create_line(pt[j][0],pt[j][1],pt[j+4][0],pt[j+4][1],fill="red",width = 2)
            self.canvas.create_line(pt[3][0],pt[3][1],pt[0][0],pt[0][1],fill="red",width = 2)
            self.canvas.create_line(pt[7][0],pt[7][1],pt[4][0],pt[4][1],fill="red",width = 2)
            self.canvas.create_line(pt[3][0],pt[3][1],pt[7][0],pt[7][1],fill="red",width = 2)
            for j in range(8):
                self.DrawPoint2D(pt[j],2,"black")
            
    ## Constructor function
    def __init__(self, path, master=None):
        self.root = master
        self.path = path
        self.draw_bump = False
        self.draw_spline = False

        tk.Frame.__init__(self, master)
        self.pack()

        self.color_tag = 1
        calib_file = open(self.path + '/Calib.txt', 'r')
        calib_data = calib_file.readlines()
        self.Size = [int(calib_data[0]), int(calib_data[1])]
        self.intrinsic = np.array([[float(calib_data[2]), float(calib_data[3]), float(calib_data[4])], \
                                   [float(calib_data[5]), float(calib_data[6]), float(calib_data[7])], \
                                   [float(calib_data[8]), float(calib_data[9]), float(calib_data[10])]])
    
        print self.intrinsic
    

        mat = scipy.io.loadmat(self.path + '/String4b.mat')
        self.lImages = mat['DepthImg']
        self.pos2d = mat['Pos2D']
        self.bdyIdx = mat['BodyIndex']

        connectionMat = scipy.io.loadmat(self.path + '/SkeletonConnectionMap.mat')
        self.connection = connectionMat['SkeletonConnectionMap']

        

        self.RGBD = RGBD.RGBD(self.path + '/Depth.tiff', self.path + '/RGB.tiff', self.intrinsic, 1000.0)
        #self.RGBD.ReadFromDisk()
        self.RGBD.LoadMat(self.lImages,self.pos2d,self.connection,self.bdyIdx )
        self.Index = 20
        self.RGBD.ReadFromMat(self.Index)
        self.RGBD.BilateralFilter(-1, 0.02, 3)
        self.RGBD.Crop2Body()
        segm = self.RGBD.BodySegmentation()
        self.RGBD.BodyLabelling()
        start_time = time.time()
        self.RGBD.Vmap_optimize()  
        elapsed_time = time.time() - start_time
        print "Vmap: %f" % (elapsed_time)
        self.RGBD.NMap_optimize()
        elapsed_time2 = time.time() - start_time - elapsed_time
        print "Nmap: %f" % (elapsed_time2)
        self.Pose = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        start_time2 = time.time()
        rendering = self.RGBD.Draw_optimize(self.Pose, 1, self.color_tag)
        self.RGBD.myPCA()
        elapsed_time3 = time.time() - start_time2
        print "bounding boxes process time: %f" % (elapsed_time3)
        
        # Show figure and images
            
        # 3D reconstruction of the whole image
        self.canvas = tk.Canvas(self, bg="white", height=self.Size[0], width=self.Size[1])
        self.canvas.pack()
        rendering = self.DrawColors2D(rendering,self.Pose)
        img = Image.fromarray(rendering, 'RGB')
        self.imgTk=ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgTk)
        #self.DrawSkeleton2D(self.Pose)
        self.DrawCenters2D(self.Pose)
        self.DrawSys2D(self.Pose)
        self.DrawOBBox2D(self.Pose)


        '''
        Test Register
        '''
        '''
        ImageTest = RGBD.RGBD(self.path + '/Depth.tiff', self.path + '/RGB.tiff', self.intrinsic, 10000.0)
        ImageTest.LoadMat(self.lImages,self.pos2d,self.connection)
        ImageTest.ReadFromMat()
        ImageTest.BilateralFilter(-1, 0.02, 3)
        ImageTest.Vmap_optimize()
        ImageTest.NMap_optimize()
        test_v = np.array([0.01, 0.02,0.015, 0.01, 0.02, 0.03]) #[random.random()/10 for _ in range(6)])
        A = TrackManager.Exponential(test_v)
        R = LA.inv(A[0:3,0:3])
        tra = -np.dot(R,A[0:3,3])
        print A
        print R
        print tra
        ImageTest.Transform(A)
        
        Tracker = TrackManager.Tracker(0.01, 0.04, 1, [10], 0.001)
        Tracker.RegisterRGBD_optimize(ImageTest, self.RGBD)
        
        #Tracker = TrackManager.Tracker(0.1, 0.2, 1, [10], 0.001)
        #Tracker.RegisterRGBD(ImageTest, self.RGBD)
        '''
        '''
        End test
        '''
        
        '''
        Test TSDF
        '''
        
#==============================================================================
#         TSDFManager = TSDFtk.TSDFManager((512,512,512))
#         start_time = time.time()
#         TSDFManager.FuseRGBD_optimized(self.RGBD, self.Pose)
#         elapsed_time = time.time() - start_time
#         print "FuseRGBD_optimized: %f" % (elapsed_time)
#         self.RGBD.depth_image = TSDFManager.RayTracing(self.RGBD, self.Pose)
#         self.RGBD.BilateralFilter(-1, 0.02, 3)
#         self.RGBD.Vmap_optimize()
#         self.RGBD.NMap_optimize()
#         renderingTr = self.RGBD.Draw_optimize(self.Pose, 1, self.color_tag)
#         
#         '''
#         End Test
#         '''
#         
#         imgTr = Image.fromarray(renderingTr, 'RGB')
#         self.imgTkTr=ImageTk.PhotoImage(imgTr)
#         self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgTkTr)
#==============================================================================
        
        #enable keyboard and mouse monitoring
        self.root.bind("<Key>", self.key)
        self.root.bind("<Button-1>", self.mouse_press)
        self.root.bind("<ButtonRelease-1>", self.mouse_release)
        self.root.bind("<B1-Motion>", self.mouse_motion)

        self.w = tk.Scale(master, from_=1, to=10, orient=tk.HORIZONTAL)
        self.w.pack()
