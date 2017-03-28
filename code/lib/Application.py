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

#==============================================================================
#             self.RGBD.DrawBB(self.Pose, self.w.get(), self.color_tag)
#             renderingBB =self.RGBD.drawBB
#             imgBB = Image.fromarray(renderingBB, 'RGB')
#             self.imgTkBB=ImageTk.PhotoImage(imgBB)
#             self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgTkBB)
#==============================================================================

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
            
#==============================================================================
#             self.RGBD.DrawBB(self.Pose, self.w.get(), self.color_tag)
#             renderingBB =self.RGBD.drawBB
#             imgBB = Image.fromarray(renderingBB, 'RGB')
#             self.imgTkBB=ImageTk.PhotoImage(imgBB)
#             self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgTkBB)
#==============================================================================
            
        
        self.x_init = event.x
        self.y_init = event.y
    
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
        self.RGBD.ReadFromMat()
        self.RGBD.BilateralFilter(-1, 0.02, 3)

        self.RGBD.BodyBBox()
        segm = self.RGBD.BodySegmentation()
        self.RGBD.CoordChange2D()
        self.RGBD.DrawSkeleton()
        start_time = time.time()
        self.RGBD.VmapBB()    
        self.RGBD.Vmap_optimize()  
        elapsed_time = time.time() - start_time
        print "VmapBB: %f" % (elapsed_time)
        self.RGBD.NMapBB()
        self.RGBD.NMap_optimize()
        elapsed_time2 = time.time() - start_time - elapsed_time
        print "NmapBB: %f" % (elapsed_time2)
        self.Pose = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        start_time2 = time.time()
        rendering = self.RGBD.Draw_optimize(self.Pose, 1, self.color_tag)
        self.RGBD.DrawBB(self.Pose, 1, self.color_tag)
        renderingBB =self.RGBD.drawBB
        elapsed_time3 = time.time() - start_time2
        print "DrawBB: %f" % (elapsed_time3)
        
        # Show figure and images
        self.imgTkBB = []
        for i in range(self.RGBD.bdyPart.shape[0]):
            Size = self.RGBD.PartBox[i].shape
            self.canvas = tk.Canvas(self, bg="white", height=Size[0], width=Size[1])
            self.canvas.pack()
            imgBB = Image.fromarray(renderingBB[i], 'RGB')
            self.imgTkBB.append(ImageTk.PhotoImage(imgBB))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgTkBB[i])

#==============================================================================
#         self.canvas = tk.Canvas(self, bg="white", height=self.Size[0], width=self.Size[1])
#         self.canvas.pack()
#         img = Image.fromarray(rendering, 'RGB')
#         self.imgTk=ImageTk.PhotoImage(img)
#         self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgTk)
#==============================================================================

   
#==============================================================================
#         self.canvas = tk.Canvas(self, bg="white", height=self.RGBD.BBBox.shape[0], width=self.RGBD.BBBox.shape[1])
#         self.canvas.pack()
#         imgSeg = Image.fromarray(segm, 'RGB')
#         self.imgTk2=ImageTk.PhotoImage(imgSeg)
#         self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgTk2)
#==============================================================================
        


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
