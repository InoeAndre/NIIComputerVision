# File created by Diego Thomas the 16-11-2016
# improved by Inoe Andre 02-2017

# Define functions to manipulate RGB-D data
import cv2
import numpy as np
from numpy import linalg as LA
import random
import imp
import time
import scipy.ndimage.measurements as spm

segm = imp.load_source('segmentation', './lib/segmentation.py')

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


def in_mat_zero2one(mat):
    """This fonction replace in the matrix all the 0 to 1"""
    mat_tmp = (mat != 0.0)
    res = mat * mat_tmp + ~mat_tmp
    return res

def division_by_norm(mat,norm):
    """This fonction divide a n by m by p=3 matrix, point by point, by the norm made through the p dimension>
    It ignores division that makes infinite values or overflow to replace it by the former mat values or by 0"""
    for i in range(3):
        with np.errstate(divide='ignore', invalid='ignore'):
            mat[:,:,i] = np.true_divide(mat[:,:,i],norm)
            mat[:,:,i][mat[:,:,i] == np.inf] = 0
            mat[:,:,i] = np.nan_to_num(mat[:,:,i])
    return mat
                
def normalized_cross_prod_optimize(a,b):
    #res = np.zeros(a.Size, dtype = "float")
    norm_mat_a = np.sqrt(np.sum(a*a,axis=2))
    norm_mat_b = np.sqrt(np.sum(b*b,axis=2))
    #changing every 0 to 1 in the matrix so that the division does not generate nan or infinite values
    norm_mat_a = in_mat_zero2one(norm_mat_a)
    norm_mat_b = in_mat_zero2one(norm_mat_b)
    # compute a/ norm_mat_a
    a = division_by_norm(a,norm_mat_a)
    b = division_by_norm(b,norm_mat_b)
    #compute cross product with matrix
    res = np.cross(a,b)
    #compute the norm of res using the same method for a and b 
    norm_mat_res = np.sqrt(np.sum(res*res,axis=2))
    norm_mat_res = in_mat_zero2one(norm_mat_res)
    #norm division
    res = division_by_norm(res,norm_mat_res)
    return res

#Nurbs class to handle NURBS curves (Non-uniform rational B-spline)
class RGBD():

    # Constructor
    def __init__(self, depthname, colorname, intrinsic, fact):
        self.depthname = depthname
        self.colorname = colorname
        self.intrinsic = intrinsic
        self.fact = fact
        
    def LoadMat(self, Images,Images_filtered,Pos_2D,BodyConnection,binaryBodyPart,binImage):
        self.lImages = Images
        self.lImages_filtered = Images_filtered
        self.numbImages = len(self.lImages.transpose())
        self.Index = -1
        self.pos2d = Pos_2D
        self.connection = BodyConnection
        self.binBody = binaryBodyPart
        self.bw = binImage
        
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
        self.depth_image = depth_in.astype(np.float32) / self.fact
        self.skel = self.depth_image.copy()

    def DrawSkeleton(self, idx = -1):
        #this function draw the Skeleton of a human and make connections between each part
        if (idx == -1):
            self.Index = self.Index + 1
        else:
            self.Index = idx
        pos = self.pos2d[0][self.Index]
        for i in range(np.size(self.connection,0)):
            pt1 = (pos[self.connection[i,0]-1,0],pos[self.connection[i,0]-1,1])
            pt2 = (pos[self.connection[i,1]-1,0],pos[self.connection[i,1]-1,1])
            cv2.line( self.skel,pt1,pt2,(0,0,255),2) # color space = BGR
            cv2.circle(self.skel,pt1,1,(0,0,255),2)
            cv2.circle(self.skel,pt2,1,(0,0,255),2)


    def Vmap(self): # Create the vertex image from the depth image and intrinsic matrice
        self.Vtx = np.zeros(self.Size, np.float32)
        for i in range(self.Size[0]): # line index (i.e. vertical y axis)
            for j in range(self.Size[1]): # column index (i.e. horizontal x axis)
                d = self.depth_image[i,j]
                if d > 0.0:
                    x = d*(j - self.intrinsic[0,2])/self.intrinsic[0,0]
                    y = d*(i - self.intrinsic[1,2])/self.intrinsic[1,1]
                    self.Vtx[i,j] = (x, y, d)
        
    
    def Vmap_optimize(self): # Create the vertex image from the depth image and intrinsic matrice
        self.Vtx = np.zeros(self.Size, np.float32)
        d = self.skel[0:self.Size[0]][0:self.Size[1]]
        d_pos = d * (d > 0.0)
        x_raw = np.zeros([self.Size[0],self.Size[1]], np.float32)
        y_raw = np.zeros([self.Size[0],self.Size[1]], np.float32)
        # change the matrix so that the first row is on all rows for x respectively colunm for y.
        x_raw[0:-1,:] = ( np.arange(self.Size[1]) - self.intrinsic[0,2])/self.intrinsic[0,0]
        y_raw[:,0:-1] = np.tile( ( np.arange(self.Size[0]) - self.intrinsic[1,2])/self.intrinsic[1,1],(1,1)).transpose()
        # multiply point by point d_pos and raw matrices
        x = d_pos * x_raw
        y = d_pos * y_raw
        self.Vtx = np.dstack((x, y,d))
    
                
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
                
    def NMap_optimize(self):
        self.Nmls = np.zeros(self.Size, np.float32)        
        nmle1 = normalized_cross_prod_optimize(self.Vtx[2:self.Size[0]  ][:,1:self.Size[1]-1] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1], \
                                               self.Vtx[1:self.Size[0]-1][:,2:self.Size[1]  ] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1])        
        nmle2 = normalized_cross_prod_optimize(self.Vtx[1:self.Size[0]-1][:,2:self.Size[1]  ] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1], \
                                               self.Vtx[0:self.Size[0]-2][:,1:self.Size[1]-1] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1])
        nmle3 = normalized_cross_prod_optimize(self.Vtx[0:self.Size[0]-2][:,1:self.Size[1]-1] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1], \
                                               self.Vtx[1:self.Size[0]-1][:,0:self.Size[1]-2] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1])
        nmle4 = normalized_cross_prod_optimize(self.Vtx[1:self.Size[0]-1][:,0:self.Size[1]-2] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1], \
                                               self.Vtx[2:self.Size[0]  ][:,1:self.Size[1]-1] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1])
        nmle = (nmle1 + nmle2 + nmle3 + nmle4)/4.0
        norm_mat_nmle = np.sqrt(np.sum(nmle*nmle,axis=2))
        norm_mat_nmle = in_mat_zero2one(norm_mat_nmle)
        #norm division 
        nmle = division_by_norm(nmle,norm_mat_nmle)
        self.Nmls[1:self.Size[0]-1][:,1:self.Size[1]-1] = nmle

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


    def Draw_optimize(self, Pose, s, color = 0) :   
        result = np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
        stack_pix = np.ones((self.Size[0], self.Size[1]), dtype = np.float32)
        stack_pt = np.ones((np.size(self.Vtx[ ::s, ::s,:],0), np.size(self.Vtx[ ::s, ::s,:],1)), dtype = np.float32)
        pix = np.zeros((self.Size[0], self.Size[1],2), dtype = np.float32)
        pix = np.dstack((pix,stack_pix))
        pt = np.dstack((self.Vtx[ ::s, ::s, :],stack_pt))
        pt = np.dot(Pose,pt.transpose(0,2,1)).transpose(1,2,0)
        nmle = np.zeros((self.Size[0], self.Size[1],self.Size[2]), dtype = np.float32)
        nmle[ ::s, ::s,:] = np.dot(Pose[0:3,0:3],self.Nmls[ ::s, ::s,:].transpose(0,2,1)).transpose(1,2,0)
        #if (pt[2] != 0.0):
        lpt = np.dsplit(pt,4)
        lpt[2] = in_mat_zero2one(lpt[2])
        # if in 1D pix[0] = pt[0]/pt[2]
        pix[ ::s, ::s,0] = (lpt[0]/lpt[2]).reshape(np.size(self.Vtx[ ::s, ::s,:],0), np.size(self.Vtx[ ::s, ::s,:],1))
        # if in 1D pix[1] = pt[1]/pt[2]
        pix[ ::s, ::s,1] = (lpt[1]/lpt[2]).reshape(np.size(self.Vtx[ ::s, ::s,:],0), np.size(self.Vtx[ ::s, ::s,:],1))
        pix = np.dot(self.intrinsic,pix[0:self.Size[0],0:self.Size[1]].transpose(0,2,1)).transpose(1,2,0)
        column_index = (np.round(pix[:,:,0])).astype(int)
        line_index = (np.round(pix[:,:,1])).astype(int)
        # create matrix that have 0 when the conditions are not verified and 1 otherwise
        cdt_column = (column_index > -1) * (column_index < self.Size[1])
        cdt_line = (line_index > -1) * (line_index < self.Size[0])
        line_index = line_index*cdt_line
        column_index = column_index*cdt_column
        if (color == 0):
            result[line_index[:][:], column_index[:][:]]= np.dstack((self.color_image[ ::s, ::s,2], \
                                                                     self.color_image[ ::s, ::s,1]*cdt_line, \
                                                                     self.color_image[ ::s, ::s,0]*cdt_column) )
        else:
            result[line_index[:][:], column_index[:][:]]= np.dstack( ( (nmle[ :, :,0]+1.0)*(255./2.), \
                                                                       ((nmle[ :, :,1]+1.0)*(255./2.))*cdt_line, \
                                                                       ((nmle[ :, :,2]+1.0)*(255./2.))*cdt_column ) ).astype(int)
        return result
    
    
        
##################################################################
###################Bilateral Smooth Funtion#######################
##################################################################
    def BilateralFilter(self, d, sigma_color, sigma_space):
        self.depth_image = cv2.bilateralFilter(self.depth_image, d, sigma_color, sigma_space)
        self.skel = cv2.bilateralFilter(self.skel, d, sigma_color, sigma_space)
    



##################################################################
################### Segmentation Function #######################
##################################################################
    def removeBG(self,binaryImage):
        ''' This function delete all the little group unwanted from the binary image'''
        labeled, n = spm.label(binaryImage)
        size = np.bincount(labeled.ravel())
        #do not consider the background
        size2 = np.delete(size,0)
        threshold = max(size2)-1
        keep_labels = size >= threshold
        # Make sure the background is left as 0/False
        keep_labels[0] = 0
        filtered_labeled = keep_labels[labeled]
        return filtered_labeled


    def BodySegmentation(self, idx = -1):
        #this function calls the function in segmentation.py to process the segmentation of the body
        if (idx == -1):
            self.Index = self.Index + 1
        else:
            self.Index = idx
        self.segm = segm.Segmentation(self.depth_image,self.colorname,self.pos2d[0][self.Index])
        segImg = (np.zeros([self.Size[0],self.Size[1],self.Size[2],self.numbImages])).astype(np.int8)
        I =  (np.zeros([self.Size[0],self.Size[1]])).astype(np.int8)
        start_time = time.time()
        #for j  in range(self.numbImages):
        pose = self.pos2d[0][self.Index]
        #depth_image = self.depth_image[0][self.Index]
        imageWBG = (self.bw[0][self.Index]>0)#self.removeBG(self.bw[0][self.Index])
        B = self.lImages_filtered[0][self.Index]
        arm = self.segm.forearmLeft(imageWBG,B)
        

#==============================================================================
#         self.binBody[0] = forearmL      color=[0,0,255]
#         self.binBody[1] = upperarmL     color=[200,200,255]
#         self.binBody[2] = forearmR     color=[0,255,0]
#         self.binBody[3] = upperarmR     color=[200,255,200]
#         self.binBody[4] = thighR     color=[255,0,255]
#         self.binBody[5] = calfR     color=[255,180,255]
#         self.binBody[6] = thighL     color=[255,255,0]
#         self.binBody[7] = calfL     color=[255,255,180]
#         self.binBody[8] = headB     color=[255,0,0]
#         self.binBody[9] = body     color=[255,255,255]
#==============================================================================
        
        # For Channel color R
#==============================================================================
#         I = I +255*self.binBody[8][self.Index]
#         #I = I +0*self.binBody[0][self.Index]
#         #I = I +200*self.binBody[1][self.Index]
#         I = I +0*self.binBody[2][self.Index]
#         I = I +200*self.binBody[3][self.Index]
#         I = I +255*self.binBody[6][self.Index]
#         I = I +255*self.binBody[7][self.Index]
#         I = I +255*self.binBody[4][self.Index]
#         I = I +255*self.binBody[5][self.Index]
#         I = I +255*self.binBody[9][self.Index]
#==============================================================================
        I = I +0*arm[0]
        I = I +200*arm[1]
        segImg[:,:,0,self.Index]=I
    
        # For Channel color G
        I =  (np.zeros([self.Size[0],self.Size[1]])).astype(np.int8)
#==============================================================================
#         I = I +0*self.binBody[8][self.Index]
#         #I = I +0*self.binBody[0][self.Index]
#         #I = I +200*self.binBody[1][self.Index]
#         I = I +255*self.binBody[2][self.Index]
#         I = I +255*self.binBody[3][self.Index]
#         I = I +255*self.binBody[6][self.Index]
#         I = I +255*self.binBody[7][self.Index]
#         I = I +0*self.binBody[4][self.Index]
#         I = I +180*self.binBody[5][self.Index]
#         I = I +255*self.binBody[9][self.Index]
#==============================================================================
        I = I +0*arm[0]
        I = I +200*arm[1]
        segImg[:,:,1,self.Index] = I
    
        # For Channel color B
        I =  (np.zeros([self.Size[0],self.Size[1]])).astype(np.int8)#I[(self.bw[0,0]>0)]=255
#==============================================================================
#         I = I +0*self.binBody[8][self.Index]
#         #I = I +255*self.binBody[0][self.Index]
#         #I = I +255*self.binBody[1][self.Index]
#         I = I +0*self.binBody[2][self.Index]
#         I = I +200*self.binBody[3][self.Index]
#         I = I +0*self.binBody[6][self.Index]
#         I = I +180*self.binBody[7][self.Index]
#         I = I +255*self.binBody[4][self.Index]
#         I = I +255*self.binBody[5][self.Index]
#         I = I +255*self.binBody[9][self.Index]
#==============================================================================
        I = I +255*arm[0]
        I = I +255*arm[1]
        segImg[:,:,2,self.Index] = I
        #I = segImg[:,:,:,0]
    
        elapsed_time = time.time() - start_time
        print "Segmentation: %f" % (elapsed_time)
        return segImg[:,:,:,self.Index]
#==============================================================================
#         markers = self.pos2d[0][self.Index]
#         self.depth_image = cv2.watershed(self.depth_image,markers)
#==============================================================================
    
                
