# File created by Diego Thomas the 21-11-2016
# Second Author Inoe AMDRE

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
            #print 'c'
            #print c
            pt0 = self.RGBD.drawNewSys[i][0]
            pt1 = self.RGBD.drawNewSys[i][1]
            pt2 = self.RGBD.drawNewSys[i][2]    
            self.canvas.create_line(pt0[0],pt0[1],c[0],c[1],fill="gray",width = 2)
            self.canvas.create_line(pt1[0],pt1[1],c[0],c[1],fill="gray",width = 2)
            self.canvas.create_line(pt2[0],pt2[1],c[0],c[1],fill="gray",width = 2)

    def DrawOBBox2D(self,Pose):
        '''Draw in the canvas the oriented bounding boxes for each body part''' 
        self.OBBcoords2D = []
        # for each body part
        for i in range(len(self.RGBD.coordsL)):
            self.OBBcoords2D.append(self.RGBD.GetProjPts2D_optimize(self.RGBD.coordsL[i],Pose))
            pt = self.OBBcoords2D[i]
            #print 'self.OBBcoords2D[]'
            #print pt
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
        self.bdyIdx = mat['BodyIndex']


        connectionMat = scipy.io.loadmat(self.path + '/SkeletonConnectionMap.mat')
        self.connection = connectionMat['SkeletonConnectionMap']
        self.Pose = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        self.T_Pose = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        Id4 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        
        # initialize lists

        self.TSDF = []
        self.TSDFGPU = []
        self.Weight = []
        self.WeightGPU = []
        self.MC = []
 
        # Loop for each image
        i = 10

        # Current Depth Image (i.e: i)
        start_time = time.time()
        self.RGBD = RGBD.RGBD(self.path + '/Depth.tiff', self.path + '/RGB.tiff', self.intrinsic, self.fact)
        self.RGBD.LoadMat(self.lImages,self.pos2d,self.connection,self.bdyIdx )   
        self.Index = i
        self.RGBD.ReadFromMat(self.Index) 
        self.RGBD.BilateralFilter(-1, 0.02, 3) 
        # segmenting the body
        self.RGBD.Crop2Body() 
        self.RGBD.BodySegmentation() 
        self.RGBD.BodyLabelling()   
        # select the body part
        self.RGBD.depth_image *= (self.RGBD.labels >0) # 9 = head; 10 = torso 
        self.RGBD.Vmap_optimize()   
        self.RGBD.NMap_optimize() 
        # create the transform matrix from local to global coordinate
        self.RGBD.myPCA()
        elapsed_time = time.time() - start_time
        print "depth conversion: %f s" % (elapsed_time)
        
        
        Vertices2 = np.zeros((self.Size[0], self.Size[1], 4), dtype = np.float32)
        Normales2 = np.zeros((self.Size[0], self.Size[1], 3), dtype = np.float32)
            
        # Loop for each body part bp
        for bp in range(1,self.RGBD.bdyPart.shape[0]+1):
            #bp = 10              
            # Compute the dimension of the body part to create the volume
            #Compute for axis X,Y and Z
            pt = self.RGBD.GetProjPts2D_optimize(self.RGBD.coordsL[bp],Id4)
            X = int(round(LA.norm(pt[1]-pt[0])))
            Y = int(round(LA.norm(pt[3]-pt[0])))
            print self.RGBD.coordsL[bp][4]
            Z = int(round(self.fact*(self.RGBD.coordsL[bp][4,2]-self.RGBD.coordsL[bp][0,2])))
    
            print "X= %d; Y= %d; Z= %d" %(X,Y,Z)
    
    
            # Create the volume
#==============================================================================
#             mf = cl.mem_flags
#             self.TSDF.append(np.zeros((X,Y,Z), dtype = np.int16))
#             self.TSDFGPU.append(cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.TSDF[0].nbytes))
#             self.Weight.append(np.zeros((X,Y,Z), dtype = np.int16))
#             self.WeightGPU.append(cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Weight[0].nbytes))
#==============================================================================        
    
            
            # extract one body part
            depth_image = np.zeros((self.Size[0],self.Size[1]))
            depth_image = (self.RGBD.labels == bp) #+ (self.RGBD.labels == 10) # 9 = head; 10 = torso               
            mask = (self.RGBD.labels == bp)# + (self.RGBD.labels == 10)
            mask3D = np.stack( (mask,mask,mask),axis=2)            
            Vertices = self.RGBD.Vtx
            Vertices = Vertices *mask3D
            Normales = self.RGBD.Nmls 
            Normales = Normales * mask3D
            print "max vtx : %lf" %(np.max(Vertices))
            print np.max(Normales)
            print np.max(depth_image)
            nbPts = sum(sum(mask))
            print nbPts
            
    
            
            # Get the tranform from the local coordinates system to the global system Tl??
            Tg =self.RGBD.TransfoBB[bp]
            Tl = self.InvPose(Tg)#self.RGBD.TransfoBB[bp-1]#T_l2g#
            #Tl[0:3,0:3] = LA.inv(self.RGBD.TransfoBB[bp][0:3,0:3])#np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype = np.float32)#T_l2g#self.RGBD.TransfoBB[bp]#
            #Tl[0:3,3] = [-0.5,0.0,0.0]
            # Compute the whole transform (local transform + image alignment transform)
            

            
#==============================================================================
#             ref_pose = np.dot(self.T_Pose,Tl)
#             for k in range(4):
#                 for l in range(4):
#                     self.T_Pose[k,l] = ref_pose[k,l]
#             print 'self.T_Pose'
#             print self.T_Pose
#==============================================================================
            
            stackVtx = np.ones((Vertices.shape[0],Vertices.shape[1]))
            Vertices = np.stack((Vertices[:,:,0],Vertices[:,:,1],Vertices[:,:,2],stackVtx),axis = 2)

            #Vertices2 += np.dot(Tl,Vertices.transpose(0,2,1)).transpose(1,2,0)
            Normales2 += np.dot(Tl[0:3,0:3],Normales.transpose(0,2,1)).transpose(1,2,0)
            
            
            
            pt2D = self.RGBD.GetProjPts2D_optimize(self.RGBD.TVtxBB[bp],Id4)
            maskLine = (pt2D[:,0] > -1)*(pt2D[:,0] < self.Size[0])
            maskCol = (pt2D[:,1] > -1)*(pt2D[:,1] < self.Size[1])
            mask = maskCol*maskLine
            print pt2D.shape
            print pt2D[:,0]
            print np.max(pt2D[:,0])
            print np.min(pt2D[:,0])
            pt2D[:,0] =  pt2D[:,0]*mask
            print pt2D[:,0]
            print pt2D[:,1]
            pt2D[:,1] =  pt2D[:,1]*mask
            print pt2D[:,1]
            print np.max(pt2D[:,0])
            print np.min(pt2D[:,0])
            print np.max(pt2D[:,1])
            print np.min(pt2D[:,1])
            Vertices2[pt2D[:,0],pt2D[:,1]][:,0:3] += self.RGBD.TVtxBB[bp]
            #Points of clouds in the TSDF local coordinates system
    
            #rescale the points of clouds       
            
            # compute TSDF indexes
    #==============================================================================
    #         param = np.array([X/2 , X /5.0, Y/2 , Y /5.0, -0.1, Z/5.0], dtype = np.float32)
    #         
    #         x = np.arange(X )
    #         y = np.arange(Y )
    #         
    #         x_init = np.zeros([int(X) ,int(Y)  ], np.int)
    #         y_init = np.zeros([int(X) ,int(Y)  ], np.int)
    #         
    #         for i in range(int(Y) ):
    #             x_init[0:int(X ),i] = x
    #         
    #         for i in range( int(X) ):
    #             y_init[i,0:int(Y) ] = y    
    #             
    #         print "x_init"          
    #         print x_init
    #         print "y_init"          
    #         print y_init
    #         nmle = np.random.rand(self.Size[0],self.Size[1],3)
    #         init = np.zeros((self.Size[0],self.Size[1],3))
    #         init[x_init[:,:], y_init[:,:]]= np.dstack(  (nmle[x_init[:,:], y_init[:,:]][ :, :,0]*255., \
    #                                                   (nmle[x_init[:,:], y_init[:,:]][ :, :,1]*255.), \
    #                                                   (nmle[x_init[:,:], y_init[:,:]][ :, :,2]*255.) ) ).astype(int) 
    #         print "init"          
    #         print init
    #==============================================================================
            
    #==============================================================================
    #         ptx = (x-param[0])/param[1]
    #         pty = (y-param[2])/param[3]
    # 
    # 
    #         x_raw = np.zeros([int(X) ,int(Y)  ], np.float32)
    #         y_raw = np.zeros([int(X) ,int(Y)  ], np.float32)
    #         
    #         for i in range(int(Y) ):
    #             x_raw[0:int(X ),i] = ptx
    #         
    #         for i in range( int(X) ):
    #             y_raw[i,0:int(Y) ] = pty
    #             
    # 
    #         
    #         
    #         # transform local to global = Tg
    #         x_T = self.T_Pose[0,0]*x_raw + self.T_Pose[0,1]*y_raw + self.T_Pose[0,3]
    #         y_T = self.T_Pose[1,0]*x_raw + self.T_Pose[1,1]*y_raw + self.T_Pose[1,3]
    #         z_T = self.T_Pose[2,0]*x_raw + self.T_Pose[2,1]*y_raw + self.T_Pose[2,3]
    #         
    #         z = np.arange(int(Z))
    #         ptz = (z-param[4])/param[5]
    #         
    #         pt_Tx = np.zeros([int(X) ,int(Y) ,int(Z) ], np.float32)
    #         pt_Ty = np.zeros([int(X) ,int(Y) ,int(Z) ], np.float32)
    #         pt_Tz = np.zeros([int(X) ,int(Y) ,int(Z) ], np.float32)
    #         
    #         for i in range(int(Z)):
    #         	pt_Tx[:,:,i] = x_T + self.T_Pose[0,2]*ptz[i]
    #         	pt_Ty[:,:,i] = y_T + self.T_Pose[1,2]*ptz[i]
    #         	pt_Tz[:,:,i] = z_T + self.T_Pose[2,2]*ptz[i]
    # 
    # 
    #         pixX = (np.round((pt_Tx/np.abs(pt_Tz))*self.intrinsic[0][0] + self.intrinsic[0][2])).astype(np.int)
    #         pixY = (np.round((pt_Ty/np.abs(pt_Tz))*self.intrinsic[1][1] + self.intrinsic[1][2])).astype(np.int)
    #         print "pixX,pixY"
    #         print pixX
    #         print pixY
    #         
    #         # compare centers, axis
    #         maskX = (pixX > 0) * (pixX < self.Size[0])
    #         maskY = (pixY > 0) * (pixY < self.Size[1])
    #         mask = maskX*maskY
    #         nbPts = np.sum(mask)
    #         print nbPts
    #         pixX = pixX* mask
    #         pixY = pixY* mask
    #         x_res = pixX[~(pixX==0)]
    #         y_res = pixY[~(pixY==0)]
    #         print x_res.shape
    #         print y_res.shape
    #         
    # 
    #         result = np.zeros((self.Size[0],self.Size[1],3))
    #         result[x_res[:], y_res[:]]= np.dstack(  (nmle[x_res[:], y_res[:]][ :,0]*255., \
    #                                                 (nmle[x_res[:], y_res[:]][ :,1]*255.), \
    #                                                 (nmle[x_res[:], y_res[:]][ :,2]*255.) ) ).astype(int)        
    #==============================================================================
            
    
            # TSDF of the body part
    #==============================================================================
    #         TSDFManager = TSDFtk.TSDFManager((X,Y,Z), self.RGBD, self.GPUManager,self.TSDFGPU[0],self.WeightGPU[0]) 
    #         TSDFManager.FuseRGBD_GPU(self.RGBD, Tg)   
    #         print TSDFManager.TSDF.shape
    #         print np.min(TSDFManager.TSDF)
    #         tsdfmax = np.max(TSDFManager.TSDF)
    #         print tsdfmax 
    #         tmp = ~(TSDFManager.TSDF ==tsdfmax)
    #         tmp = tmp*TSDFManager.TSDF
    #         print np.max(tmp)
    #==============================================================================
    
          
            # Create Mesh
    #==============================================================================
    #         self.MC = My_MC.My_MarchingCube(TSDFManager.Size, TSDFManager.res, 0.0, self.GPUManager)     
    #         # Mesh rendering
    #         self.MC.runGPU(TSDFManager.TSDFGPU) 
    #         start_time3 = time.time()
    #         # save
    #         self.MC.SaveToPly("torso.ply")
    #         elapsed_time = time.time() - start_time3
    #         print "SaveToPly: %f" % (elapsed_time)             
    #==============================================================================
            # Get new current image
            
            # Once it done for all part
            # Transform the segmented part in the current image (alignment current image mesh)
            #Tracker = TrackManager.Tracker(0.01, 0.5, 1, [10], 0.001)  
            # restart processing of each body part for the current image.
            
            
            '''
            TEST 1 Visualize it 
            '''
            
            '''
            # projection in 2d space to draw it
            rendering =np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
            # projection of the current image/ Overlay
            rendering = self.RGBD.Draw_optimize(rendering,Tl2, 1, self.color_tag)
            
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
            '''
    
            
            '''
            # For global Fusion
            mf = cl.mem_flags
            self.TSDF = np.zeros((512,512,512), dtype = np.int16)
            self.TSDFGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.TSDF.nbytes)
            self.Weight = np.zeros((512,512,512), dtype = np.int16)
            self.WeightGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Weight.nbytes)
            
            TSDFManager = TSDFtk.TSDFManager((512,512,512), self.RGBD, self.GPUManager,self.TSDFGPU,self.WeightGPU) 
            self.MC = My_MC.My_MarchingCube(TSDFManager.Size, TSDFManager.res, 0.0, self.GPUManager)
            Tracker = TrackManager.Tracker(0.01, 0.5, 1, [10], 0.001)
            
    
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
            '''
        Vertices2 = Vertices2.reshape(Vertices2.shape[0]*Vertices2.shape[1],Vertices2.shape[2])
        Normales2 = Normales2.reshape(Normales2.shape[0]*Normales2.shape[1],Normales2.shape[2])
        
        
        #Vertices2 = TVtxBB
        # projection in 2d space to draw it
        rendering =np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
        # projection of the current image/ Overlay
        #rendering = self.RGBD.Draw_optimize(rendering,Id4, 1, self.color_tag)
        rendering = self.RGBD.DrawMesh(rendering,Vertices2[:,0:3],Normales2,Id4, 1, self.color_tag)
        #rendering = self.RGBD.Draw_optimize(rendering,Id4, 1, self.color_tag)


        

        # Show figure and images
            
        # 3D reconstruction of the whole image
        self.canvas = tk.Canvas(self, bg="black", height=self.Size[0], width=self.Size[1])
        self.canvas.pack()        
        #rendering = self.DrawColors2D(self.RGBD,rendering,self.Pose)
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


