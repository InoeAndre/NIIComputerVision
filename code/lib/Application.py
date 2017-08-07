# File created by Diego Thomas the 21-11-2016
# Second Author Inoe ANDRE

# File to handle program main loop
import sys
import cv2
from math import cos,sin
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
import pyopencl as cl




RGBD = imp.load_source('RGBD', './lib/RGBD.py')
RGBDimg = imp.load_source('RGBDimg', './lib/RGBDimg.py')
TrackManager = imp.load_source('TrackManager', './lib/tracking.py')
TSDFtk = imp.load_source('TSDFtk', './lib/TSDF.py')
GPU = imp.load_source('GPUManager', './lib/GPUManager.py')
My_MC = imp.load_source('My_MarchingCube', './lib/My_MarchingCube.py')
Stitcher = imp.load_source('Stitcher', './lib/Stitching.py')
BdyPrt = imp.load_source('BodyParts', './lib/BodyParts.py')
General = imp.load_source('General', './lib/General.py')


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
            rendering = self.RGBD.Draw_optimize(rendering,self.Pose, self.w.get(), self.color_tag)
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
        for i in range(1, len(self.RGBD.ctr3D)):
            c = self.ctr2D[i]
            self.DrawPoint2D(c,2,"yellow")

    def DrawSys2D(self,Pose):
        '''this function draw the sys of oriented coordinates system for each body part''' 
        self.RGBD.GetNewSys(Pose,self.ctr2D,10)
        for i in range(1,len(self.ctr2D)):
            c = self.ctr2D[i]
            #print 'c'
            #print c
            pt0 = self.RGBD.drawNewSys[i-1][0]
            pt1 = self.RGBD.drawNewSys[i-1][1]
            pt2 = self.RGBD.drawNewSys[i-1][2]    
            self.canvas.create_line(pt0[0],pt0[1],c[0],c[1],fill="gray",width = 2)
            self.canvas.create_line(pt1[0],pt1[1],c[0],c[1],fill="gray",width = 2)
            self.canvas.create_line(pt2[0],pt2[1],c[0],c[1],fill="gray",width = 2)

    def DrawOBBox2D(self,Pose):
        '''Draw in the canvas the oriented bounding boxes for each body part''' 
        self.OBBcoords2D = []  
        self.OBBcoords2D.append([0.,0.,0.])
        # for each body part
        for i in range(1,len(self.RGBD[0].coordsGbl)):
            self.OBBcoords2D.append(self.RGBD[0].GetProjPts2D_optimize(self.RGBD[0].coordsGbl[i],Pose))
            pt = self.OBBcoords2D[i]
            #print 'self.OBBcoords2D[]'
            #print pt.shape
            # create lines of the boxes
            for j in range(3):
                self.canvas.create_line(pt[j][0],pt[j][1],pt[j+1][0],pt[j+1][1],fill="red",width =2)
                self.canvas.create_line(pt[j+4][0],pt[j+4][1],pt[j+5][0],pt[j+5][1],fill="red",width = 2)
                self.canvas.create_line(pt[j][0],pt[j][1],pt[j+4][0],pt[j+4][1],fill="red",width = 2)
            self.canvas.create_line(pt[3][0],pt[3][1],pt[0][0],pt[0][1],fill="red",width = 2)
            self.canvas.create_line(pt[7][0],pt[7][1],pt[4][0],pt[4][1],fill="red",width = 2)
            self.canvas.create_line(pt[3][0],pt[3][1],pt[7][0],pt[7][1],fill="red",width = 2)
            #draw points of the bounding boxes
            for j in range(8):
                self.DrawPoint2D(pt[j],2,"blue")
    
    
    ## Constructor function
    def __init__(self, path,  GPUManager, master=None):
        self.root = master
        self.path = path
        self.GPUManager = GPUManager
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
                                   [float(calib_data[8]), float(calib_data[9]), float(calib_data[10])]], dtype = np.float32)
    
        print self.intrinsic

        fact = 1000.0

        #load data
        mat = scipy.io.loadmat(path + '/String4b.mat')
        lImages = mat['DepthImg']
        self.pos2d = mat['Pos2D']
        bdyIdx = mat['BodyIndex']

        self.connectionMat = scipy.io.loadmat(path + '/SkeletonConnectionMap.mat')
        self.connection = self.connectionMat['SkeletonConnectionMap']
        self.Pose = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        T_Pose = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        PoseBP = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        Id4 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        
        
        self.Index = 0
        nunImg = 2

        # Former Depth Image (i.e: i)
        self.RGBD = []
        for bp in range(15):
            self.RGBD.append(RGBD.RGBD(path + '/Depth.tiff', path + '/RGB.tiff', self.intrinsic, fact))
            self.RGBD[bp].LoadMat(lImages,self.pos2d,self.connection,bdyIdx )
            self.RGBD[bp].ReadFromMat(self.Index)
            self.RGBD[bp].BilateralFilter(-1, 0.02, 3)
            if bp == 0:
                # segmenting the body
                self.RGBD[bp].RGBDSegmentation()
                self.RGBD[bp].depth_image *= (self.RGBD[bp].labels >0)
            # elif bp == 9:
            #     self.RGBD[bp].depth_image *= (self.RGBD[bp].labels == bp) + (self.RGBD[bp].labels == bp+1)
            # elif bp == 10:
            #     self.RGBD[bp].depth_image *= (self.RGBD[bp].labels == bp -1 ) + (self.RGBD[bp].labels == bp)
            else:
                self.RGBD[bp].depth_image *= (self.RGBD[0].labels == bp)
            self.RGBD[bp].Vmap_optimize()   
            self.RGBD[bp].NMap_optimize()

        # create the transform matrices that transform from local to global coordinate
        self.RGBD[0].myPCA()

        '''
        The first image is process differently from the other since it does not have any previous value.
        '''
        # Init for Local to global Transform
        Tg = []
        Tg.append(Id4)
        for bp in range(1,self.RGBD[0].bdyPart.shape[0]+1):
            # Get the tranform matrix from the local coordinates system to the global system 
            Tglo = self.RGBD[0].TransfoBB[bp]
            Tg.append(Tglo.astype(np.float32))

        # Sum of the number of vertices and faces of all body parts
        nb_verticesGlo = 0
        nb_facesGlo = 0
        # Number of body part +1 since the counting starts from 1
        bpstart = 1
        nbBdyPart = self.RGBD[0].bdyPart.shape[0]+1
        #Initiate stitcher object 
        StitchBdy = Stitcher.Stitch(nbBdyPart)
        Parts = []
        Parts.append(BdyPrt.BodyParts(self.GPUManager,self.RGBD[0],self.RGBD[0], Tg[0]))
        # Creating mesh of each body part
        for bp in range(bpstart,nbBdyPart):
            Parts.append(BdyPrt.BodyParts(self.GPUManager, self.RGBD[0], self.RGBD[bp], Tg[bp]))
            Parts[bp].Model3D_init(bp)

            # Update number of vertices and faces in the stitched mesh
            nb_verticesGlo = nb_verticesGlo + Parts[bp].MC.nb_vertices[0]
            nb_facesGlo = nb_facesGlo +Parts[bp].MC.nb_faces[0]

            #Put the Global transfo in PoseBP so that the dtype entered in the GPU is correct
            for i in range(4):
                for j in range(4):
                    PoseBP[i][j] = Tg[bp][i][j]
            # Stitch all the body parts
            if bp == bpstart :
                StitchBdy.StitchedVertices = StitchBdy.TransformVtx(Parts[bp].MC.Vertices,PoseBP,1)
                StitchBdy.StitchedNormales = StitchBdy.TransformNmls(Parts[bp].MC.Normales,PoseBP,1)
                StitchBdy.StitchedFaces = Parts[bp].MC.Faces
            else:
                StitchBdy.NaiveStitch(Parts[bp].MC.Vertices,Parts[bp].MC.Normales,Parts[bp].MC.Faces,PoseBP)



        # save with the number of the body part
        #bpStr = str(idx)   #"+bpStr+"      
        start_time3 = time.time()
        Parts[1].MC.SaveToPlyExt("wholeBody.ply",nb_verticesGlo,nb_facesGlo,StitchBdy.StitchedVertices,StitchBdy.StitchedFaces)
        elapsed_time = time.time() - start_time3
        print "SaveToPly: %f" % (elapsed_time)                     

        #"""
        Tracker = TrackManager.Tracker(0.001, 0.5, 1, [10])
        TimeStart = time.time()

        for imgk in range(self.Index+1,nunImg):
            #Time counting
            start = time.time()

            '''
            New Image 
            '''
            # Current Depth Image (i.e: i+1)
            newRGBD = []
            # separate  each body parts of the image into different object -> each object have just the body parts in its depth image
            for bp in range(nbBdyPart):
                newRGBD.append(RGBD.RGBD(path + '/Depth.tiff', path + '/RGB.tiff', self.intrinsic, fact))
                newRGBD[bp].LoadMat(lImages,self.pos2d,self.connection,bdyIdx )              
                # Get new current image
                newRGBD[bp].ReadFromMat(imgk) 
                newRGBD[bp].BilateralFilter(-1, 0.02, 3)
                if bp == 0:
                    # segmenting the body
                    newRGBD[bp].RGBDSegmentation()
                    # select the body part
                    newRGBD[bp].depth_image *= (newRGBD[bp].labels > 0) 
                else:
                    newRGBD[bp].depth_image *= (newRGBD[0].labels == bp)

                newRGBD[bp].Vmap_optimize()   
                newRGBD[bp].NMap_optimize()        
            # create the transform matrix from local to global coordinate
            newRGBD[0].myPCA()

            # Transform the stitch body in the current image (alignment current image mesh) 
            # New pose estimation
            NewPose = Tracker.RegisterRGBDMesh_optimize(newRGBD[0],StitchBdy.StitchedVertices,StitchBdy.StitchedNormales, T_Pose)

            # Transfert NewPose in T_Pose which can be used by GPU
            for k in range(4):
                for l in range(4):
                    T_Pose[k,l] = NewPose[k,l]

            
            # restart processing of each body part for the current image.
            # Sum of the number of vertices and faces of all body parts
            nb_verticesGlo = 0
            nb_facesGlo = 0
            
            #Initiate stitcher object 
            StitchBdy = Stitcher.Stitch(nbBdyPart)      
            
            # Updating mesh of each body part
            for bp in range(1,nbBdyPart):
                # Transform in the current image
                Tg_new = np.dot(T_Pose,Tg[bp])
                # Put the Global transfo in PoseBP so that the dtype entered in the GPU is correct
                for i in range(4):
                    for j in range(4):
                        PoseBP[i][j] = Tg_new[i][j]#Tg[bp][i][j]#
    
                # TSDF Fusion of the body part
                Parts[bp].TSDFManager.FuseRGBD_GPU(newRGBD[bp], PoseBP)

                # Create Mesh
                Parts[bp].MC = My_MC.My_MarchingCube(Parts[bp].TSDFManager.Size, Parts[bp].TSDFManager.res, 0.0, self.GPUManager)
                # Mesh rendering
                Parts[bp].MC.runGPU(Parts[bp].TSDFManager.TSDFGPU)
    
                # Update number of vertices and faces in the stitched mesh
                nb_verticesGlo = nb_verticesGlo + Parts[bp].MC.nb_vertices[0]
                nb_facesGlo = nb_facesGlo +Parts[bp].MC.nb_faces[0]
                
                # Stitch all the body parts
                if bp ==1 :
                    StitchBdy.StitchedVertices = StitchBdy.TransformVtx(Parts[bp].MC.Vertices,PoseBP,1)
                    StitchBdy.StitchedNormales = StitchBdy.TransformNmls(Parts[bp].MC.Normales,PoseBP,1)
                    StitchBdy.StitchedFaces = Parts[bp].MC.Faces
                else:
                    StitchBdy.NaiveStitch(Parts[bp].MC.Vertices,Parts[bp].MC.Normales,Parts[bp].MC.Faces,PoseBP)
            time_lapsed = time.time() - start
            print "numero %d finished : %f" %(imgk,time_lapsed)
                    

            # save with the number of the body part
            imgkStr = str(imgk)
            Parts[bp].MC.SaveToPlyExt("wholeBody"+imgkStr+".ply",nb_verticesGlo,nb_facesGlo,StitchBdy.StitchedVertices,StitchBdy.StitchedFaces,1)

        
        TimeStart_Lapsed = time.time() - TimeStart
        print "total timw: %f" %(TimeStart_Lapsed)
        #"""

        # projection in 2d space to draw it
        rendering =np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
        # projection of the current image/ Overlay
        #rendering = self.RGBD.Draw_optimize(rendering,Id4, 1, self.color_tag)

        for bp in range(bpstart,nbBdyPart):
            bou = bp - bpstart + 1
            for i in range(4):
                for j in range(4):
                    PoseBP[i][j] = Parts[bou].Tlg[i][j]
            rendering = self.RGBD[0].DrawMesh(rendering,Parts[bou].MC.Vertices,Parts[bou].MC.Normales,PoseBP, 1, self.color_tag)

        # 3D reconstruction of the whole image
        self.canvas = tk.Canvas(self, bg="black", height=self.Size[0], width=self.Size[1])
        self.canvas.pack()        
        #rendering = self.DrawColors2D(self.RGBD[0],rendering,self.Pose)
        img = Image.fromarray(rendering, 'RGB')
        self.imgTk=ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgTk)
        self.DrawSkeleton2D(self.Pose)
        #self.DrawCenters2D(self.Pose)
        #self.DrawSys2D(self.Pose)
        #self.DrawOBBox2D(self.Pose)

        #enable keyboard and mouse monitoring
        self.root.bind("<Key>", self.key)
        self.root.bind("<Button-1>", self.mouse_press)
        self.root.bind("<ButtonRelease-1>", self.mouse_release)
        self.root.bind("<B1-Motion>", self.mouse_motion)

        self.w = tk.Scale(master, from_=1, to=10, orient=tk.HORIZONTAL)
        self.w.pack()
        

