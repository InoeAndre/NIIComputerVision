# File created by Diego Thomas the 16-11-2016

# Define functions to manipulate RGB-D data
import cv2
import numpy as np
from numpy import linalg as LA
import random


def normalized_cross_prod(a,b):
    res = np.zeros(3, dtype = "float")
    if (LA.norm(a) == 0.0 or LA.norm(b) == 0.0):
        return res
    a = a/LA.norm(a)
    b = b/LA.norm(b)
    res[0] = a[1]*b[2] - a[2]*b[1]
    res[1] = -a[0]*b[2] + a[2]*b[0]
    res[2] = a[0]*b[1] - a[1]*b[0]
    if (LA.norm(res) > 0.0):
        res = res/LA.norm(res)
    return res

#Nurbs class to handle NURBS curves (Non-uniform rational B-spline)
class RGBD():

    # Constructor
    def __init__(self, depthname, colorname, intrinsic, fact):
        self.depthname = depthname
        self.colorname = colorname
        self.intrinsic = intrinsic
        self.fact = fact
        
    def LoadMat(self, Images):
        self.lImages = Images
        self.numbImages = len(self.lImages.transpose())
        self.Index = -1
        
        
    def ReadFromDisk(self): #Read an RGB-D image from the disk
        print(self.depthname)
        self.depth_in = cv2.imread(self.depthname, -1)
        self.color_image = cv2.imread(self.colorname, -1)
        
        self.Size = self.depth_in.shape
        self.depth_image = np.zeros((self.Size[0], self.Size[1]), np.float32)
        for i in range(self.Size[0]): # line index (i.e. vertical y axis)
            for j in range(self.Size[1]):
                self.depth_image[i,j] = float(self.depth_in[i,j][0]) / self.fact
                                
    def ReadFromMat(self, idx = -1):
        if (idx == -1):
            self.Index = self.Index + 1
        else:
            self.Index = idx
            
        depth_in = self.lImages[0][self.Index]
        print "Input depth image is of size: ", depth_in.shape
        size_depth = depth_in.shape
        self.Size = (size_depth[0], size_depth[1], 3)
        self.depth_image = np.zeros((self.Size[0], self.Size[1]), np.float32)
        for i in range(self.Size[0]): # line index (i.e. vertical y axis)
            for j in range(self.Size[1]):
                self.depth_image[i,j] = float(depth_in[i,j]) / self.fact

    def Vmap(self): # Create the vertex image from the depth image and intrinsic matrice
        self.Vtx = np.zeros(self.Size, np.float32)
        for i in range(self.Size[0]): # line index (i.e. vertical y axis)
            for j in range(self.Size[1]): # column index (i.e. horizontal x axis)
                d = self.depth_image[i,j]
                if d > 0.0:
                    x = d*(j - self.intrinsic[0,2])/self.intrinsic[0,0]
                    y = d*(i - self.intrinsic[1,2])/self.intrinsic[1,1]
                    self.Vtx[i,j] = (x, y, d)
        
    ##### Compute normals
    def NMap(self):
        self.Nmls = np.zeros(self.Size, np.float32)
        for i in range(1,self.Size[0]-1):
            for j in range(1, self.Size[1]-1):
                nmle1 = normalized_cross_prod(self.Vtx[i+1, j]-self.Vtx[i, j], self.Vtx[i, j+1]-self.Vtx[i, j])
                nmle2 = normalized_cross_prod(self.Vtx[i, j+1]-self.Vtx[i, j], self.Vtx[i-1, j]-self.Vtx[i, j])
                nmle3 = normalized_cross_prod(self.Vtx[i-1, j]-self.Vtx[i, j], self.Vtx[i, j-1]-self.Vtx[i, j])
                nmle4 = normalized_cross_prod(self.Vtx[i, j-1]-self.Vtx[i, j], self.Vtx[i+1, j]-self.Vtx[i, j])
                nmle = (nmle1 + nmle2 + nmle3 + nmle4)/4.0
                if (LA.norm(nmle) > 0.0):
                    nmle = nmle/LA.norm(nmle)
                self.Nmls[i, j] = (nmle[0], nmle[1], nmle[2])

    def Draw(self, Pose, s, color = 0) :
        result = np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
        line_index = 0
        column_index = 0
        pix = np.array([0., 0., 1.])
        pt = np.array([0., 0., 0., 1.])
        nmle = np.array([0., 0., 0.])
        for i in range(self.Size[0]/s):
            for j in range(self.Size[1]/s):
                pt[0] = self.Vtx[i*s,j*s][0]
                pt[1] = self.Vtx[i*s,j*s][1]
                pt[2] = self.Vtx[i*s,j*s][2]
                pt = np.dot(Pose, pt)
                nmle[0] = self.Nmls[i*s,j*s][0]
                nmle[1] = self.Nmls[i*s,j*s][1]
                nmle[2] = self.Nmls[i*s,j*s][2]
                nmle = np.dot(Pose[0:3,0:3], nmle)
                if (pt[2] != 0.0):
                    pix[0] = pt[0]/pt[2]
                    pix[1] = pt[1]/pt[2]
                    pix = np.dot(self.intrinsic, pix)
                    column_index = int(round(pix[0]))
                    line_index = int(round(pix[1]))
                    if (column_index > -1 and column_index < self.Size[1] and line_index > -1 and line_index < self.Size[0]):
                        if (color == 0):
                            result[line_index, column_index] = (self.color_image[i*s,j*s][2], self.color_image[i*s,j*s][1], self.color_image[i*s,j*s][0])
                        else:
                            result[line_index, column_index] = (int((nmle[0] + 1.0)*(255./2.)), int((nmle[1] + 1.0)*(255./2.)), int((nmle[2] + 1.0)*(255./2.)))

        return result
    
##################################################################
###################Bilateral Smooth Funtion#######################
##################################################################
    def BilateralFilter(self, d, sigma_color, sigma_space):
        self.depth_image = cv2.bilateralFilter(self.depth_image, d, sigma_color, sigma_space)
    
    
    
                
