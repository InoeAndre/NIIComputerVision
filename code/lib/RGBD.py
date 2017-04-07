# File created by Diego Thomas the 16-11-2016
# improved by Inoe Andre from 02-2017

# Define functions to manipulate RGB-D data
import cv2
import numpy as np
from numpy import linalg as LA
import random
import imp
import time
import scipy.ndimage.measurements as spm
import pdb
from skimage import img_as_ubyte
from scipy import ndimage


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
        
    def LoadMat(self, Images,Pos_2D,BodyConnection,binImage):
        self.lImages = Images
        self.numbImages = len(self.lImages.transpose())
        self.Index = -1
        self.pos2d = Pos_2D
        self.connection = BodyConnection
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


    def DrawSkeleton(self):
        '''this function draw the Skeleton of a human and make connections between each part'''
        pos = self.pos2d[0][self.Index]
        for i in range(np.size(self.connection,0)):
            pt1 = (pos[self.connection[i,0]-1,0],pos[self.connection[i,0]-1,1])
            pt2 = (pos[self.connection[i,1]-1,0],pos[self.connection[i,1]-1,1])
            cv2.line( self.skel,pt1,pt2,(0,0,255),2) # color space = BGR
            cv2.circle(self.skel,pt1,1,(0,0,255),2)
            cv2.circle(self.skel,pt2,1,(0,0,255),2)

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
    
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
        #self.Vtx = np.zeros(self.Size, np.float32)
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

    def VmapBB(self,im=0): # Create the vertex image from the segmented part of the body and intrinsic matrice
        self.VtxBB = []
        #self.corners3D =[]
        for i in range(self.bdyPart.shape[0]):
            if im==0:            
                Size = self.bdyPart[i].shape
                partBox = self.bdyPart[i]
            else:
                Size = self.PartBox[i].shape
                partBox = self.PartBox[i]
            #shift = self.transBB
            d = partBox#[0:Size[0]][0:Size[1]]
            d_pos = d * (d > 0.0)
            x_raw = np.zeros(Size, np.float32)
            y_raw = np.zeros(Size, np.float32)
            # change the matrix so that the first row is on all rows for x respectively colunm for y.
            x_raw[0:-1,:] = ( np.arange(Size[1]) - self.intrinsic[0,2] )/self.intrinsic[0,0]
            y_raw[:,0:-1] = np.tile( ( np.arange(Size[0]) - self.intrinsic[1,2])/self.intrinsic[1,1],(1,1)).transpose()
            # multiply point by point d_pos and raw matrices
            x = d_pos * x_raw
            y = d_pos * y_raw
            self.VtxBB.append(np.dstack( (x, y,d) ))

                
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
        
    def NMapBB(self, im = 0):
        self.NmlsBB = []

        for i in range(self.bdyPart.shape[0]):
            if im==0:
                Size = self.bdyPart[i].shape
            else:
                Size = self.PartBox[i].shape
            Vtx = self.VtxBB[i]
            NmlsBB = np.zeros([Size[0],Size[1],3], np.float32)  
            nmle1 = normalized_cross_prod_optimize(Vtx[2:Size[0]  ][:,1:Size[1]-1] - Vtx[1:Size[0]-1][:,1:Size[1]-1], \
                                                   Vtx[1:Size[0]-1][:,2:Size[1]  ] - Vtx[1:Size[0]-1][:,1:Size[1]-1])        
            nmle2 = normalized_cross_prod_optimize(Vtx[1:Size[0]-1][:,2:Size[1]  ] - Vtx[1:Size[0]-1][:,1:Size[1]-1], \
                                                   Vtx[0:Size[0]-2][:,1:Size[1]-1] - Vtx[1:Size[0]-1][:,1:Size[1]-1])
            nmle3 = normalized_cross_prod_optimize(Vtx[0:Size[0]-2][:,1:Size[1]-1] - Vtx[1:Size[0]-1][:,1:Size[1]-1], \
                                                   Vtx[1:Size[0]-1][:,0:Size[1]-2] - Vtx[1:Size[0]-1][:,1:Size[1]-1])
            nmle4 = normalized_cross_prod_optimize(Vtx[1:Size[0]-1][:,0:Size[1]-2] - Vtx[1:Size[0]-1][:,1:Size[1]-1], \
                                                   Vtx[2:Size[0]  ][:,1:Size[1]-1] - Vtx[1:Size[0]-1][:,1:Size[1]-1])
            nmle = (nmle1 + nmle2 + nmle3 + nmle4)/4.0
            norm_mat_nmle = np.sqrt(np.sum(nmle*nmle,axis=2))
            norm_mat_nmle = in_mat_zero2one(norm_mat_nmle)
            #norm division 
            nmle = division_by_norm(nmle,norm_mat_nmle)
            NmlsBB[1:Size[0]-1][:,1:Size[1]-1] = nmle
            self.NmlsBB.append(NmlsBB)

                
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
    
    def DrawBB(self, Pose, s, color = 0,im=0) :   
        self.drawBB = []
        for i in range(self.bdyPart.shape[0]):
            if im == 0:
                Size = self.bdyPart[i].shape
            else:
                Size = self.PartBox[i].shape
            Vtx = self.VtxBB[i]
            Nmls = self.NmlsBB[i]          
            result = np.zeros((Size[0], Size[1], 3), dtype = np.uint8)
            stack_pix = np.ones((Size[0], Size[1]), dtype = np.float32)
            stack_pt = np.ones((np.size(Vtx[ ::s, ::s,:],0), np.size(Vtx[ ::s, ::s,:],1)), dtype = np.float32)
            pix = np.zeros((Size[0], Size[1],2), dtype = np.float32)
            pix = np.dstack((pix,stack_pix))
            pt = np.dstack((Vtx[ ::s, ::s, :],stack_pt))
            pt = np.dot(Pose,pt.transpose(0,2,1)).transpose(1,2,0)
            nmle = np.zeros((Size[0], Size[1],3), dtype = np.float32)
            nmle[ ::s, ::s,:] = np.dot(Pose[0:3,0:3],Nmls[ ::s, ::s,:].transpose(0,2,1)).transpose(1,2,0)
            #if (pt[2] != 0.0):
            lpt = np.dsplit(pt,4)
            lpt[2] = in_mat_zero2one(lpt[2])
            # if in 1D pix[0] = pt[0]/pt[2]
            pix[ ::s, ::s,0] = (lpt[0]/lpt[2]).reshape(np.size(Vtx[ ::s, ::s,:],0), np.size(Vtx[ ::s, ::s,:],1))
            # if in 1D pix[1] = pt[1]/pt[2]
            pix[ ::s, ::s,1] = (lpt[1]/lpt[2]).reshape(np.size(Vtx[ ::s, ::s,:],0), np.size(Vtx[ ::s, ::s,:],1))
            pix = np.dot(self.intrinsic,pix[0:Size[0],0:Size[1]].transpose(0,2,1)).transpose(1,2,0)
            column_index = (np.round(pix[:,:,0])).astype(int)
            line_index = (np.round(pix[:,:,1])).astype(int)
            # create matrix that have 0 when the conditions are not verified and 1 otherwise
            cdt_column = (column_index > -1) * (column_index < Size[1])
            cdt_line = (line_index > -1) * (line_index < Size[0])
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
            #for others points
            self.drawBB.append(result)


##################################################################
###################Bilateral Smooth Funtion#######################
##################################################################
    def BilateralFilter(self, d, sigma_color, sigma_space):
        self.depth_image = (self.depth_image[:,:] > 0.0) * cv2.bilateralFilter(self.depth_image, d, sigma_color, sigma_space)
        self.skel = (self.skel[:,:] > 0.0) *cv2.bilateralFilter(self.skel, d, sigma_color, sigma_space)
        
##################################################################
###################Transformation Funtion#######################
##################################################################
    def Transform(self, Pose):
        stack_pt = np.ones((np.size(self.Vtx,0), np.size(self.Vtx,1)), dtype = np.float32)
        pt = np.dstack((self.Vtx, stack_pt))
        self.Vtx = np.dot(Pose,pt.transpose(0,2,1)).transpose(1,2,0)[:, :, 0:3]
        self.Nmls = np.dot(Pose[0:3,0:3],self.Nmls.transpose(0,2,1)).transpose(1,2,0)
        
    



##################################################################
################### Segmentation Function #######################
##################################################################
    def RemoveBG(self,binaryImage):
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

    
    def EntireBdy(self):
        '''this function threshold the depth image in order to to get the whole body alone'''
        pos2D = self.pos2d[0][self.Index]
        max_value = np.iinfo(np.uint16).max # = 65535 for uint16
        tmp = self.depth_image*max_value
        self.depth_image = tmp.astype(np.uint16)
        
        # Threshold according to detph of the body
        bdyVals = self.depth_image[pos2D[:,0]-1,pos2D[:,1]-1]
        #only keep values different from 0
        bdy = bdyVals[np.nonzero(bdyVals != 0)]
        mini =  np.min(bdy)
        print "mini: %u" % (mini)
        maxi = np.max(bdy)
        print "max: %u" % (maxi)        
        moy = np.mean(bdy)
        print "moy: %u" % (moy)
        std = np.std(bdy)
        print "std: %u" % (std)
        bwmin = (self.depth_image > moy-std)# mini+0.08*mini)#
        bwmax = (self.depth_image < moy+std)#maxi-0.35*maxi)#
        bw0 = bwmin*bwmax
        # Compare with thenoised binary image given by the kinect
        thresh2,tmp = cv2.threshold(self.bw[0,self.Index],0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        res = tmp *bw0 
        # Remove all stand alone object
        res = ( self.RemoveBG(res)>0)
        return res#bw0#tmp#
    
    def EntireBdyBB(self):
        '''this function threshold the depth image in order to to get the whole body alone with the bounding box (BB)'''
        pos2D = self.BBPos
        max_value = np.iinfo(np.uint16).max # = 65535 for uint16
        #tmp = self.BBox*max_value
        #self.BBox = tmp.astype(np.uint16)
        self.BBox = self.BBox.astype(np.uint16)
        # Threshold according to detph of the body
        bdyVals = self.BBox[pos2D[self.connection[:,0]-1,1]-1,pos2D[self.connection[:,0]-1,0]-1]
        #only keep vales different from 0
        bdy = bdyVals[np.nonzero(bdyVals != 0)]
        mini =  np.min(bdy)
        print "mini: %u" % (mini)
        maxi = np.max(bdy)
        print "max: %u" % (maxi)
        bwmin = (self.BBox > mini-0.01*max_value) 
        bwmax = (self.BBox < maxi+0.01*max_value)
        bw0 = bwmin*bwmax
        # Compare with thenoised binary image given by the kinect
        thresh2,tmp = cv2.threshold(self.BBbw,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        res = tmp * bw0        
        # Remove all stand alone object
        bw0 = ( self.RemoveBG(bw0)>0)

        return res


    def BodySegmentation(self):
        '''this function calls the function in segmentation.py to process the segmentation of the body'''
        start_time = time.time()
#==============================================================================
#         self.segm = segm.Segmentation(self.lImages[0,self.Index],self.pos2d[0,self.Index])
#         segImg = (np.zeros([self.Size[0],self.Size[1],self.Size[2],self.numbImages])).astype(np.int8)
#         bdyImg = (np.zeros([self.Size[0],self.Size[1],self.Size[2],self.numbImages])).astype(np.int8) 
#         I =  (np.zeros([self.Size[0],self.Size[1]])).astype(np.int8)
#         #segmentation of the whole body 
#         imageWBG = (self.EntireBdy()>0)
#         B = self.lImages[0][self.Index]
#==============================================================================
        #Bounding box version
        self.segm = segm.Segmentation(self.BBox,self.BBPos) 
        segImg = (np.zeros([self.BBox.shape[0],self.BBox.shape[1],self.Size[2],self.numbImages])).astype(np.int8)
        bdyImg = (np.zeros([self.BBox.shape[0],self.BBox.shape[1],self.Size[2],self.numbImages])).astype(np.int8) 
        I =  (np.zeros([self.BBox.shape[0],self.BBox.shape[1]])).astype(np.int8)
        #segmentation of the whole body 
        imageWBG = (self.EntireBdyBB()>0)
        B = self.BBox
        
          # Visualize the body
#==============================================================================
#         M = np.max(self.depth_image) #self.depth_image*(255./M)#
#         bdyImg[:,:,0,self.Index]=imageWBG*255#self.depth_image*(255./M)#
#         bdyImg[:,:,1,self.Index]=imageWBG*255#self.depth_image*(255./M)#
#         bdyImg[:,:,2,self.Index]=imageWBG*255#self.depth_image*(255./M)#
#         return bdyImg[:,:,:,self.Index]
#==============================================================================
    
    
        right = 0
        left = 1
        armLeft = self.segm.armSeg(imageWBG,B,left)
        armRight = self.segm.armSeg(imageWBG,B,right)
        legRight = self.segm.legSeg(imageWBG,right)
        legLeft = self.segm.legSeg(imageWBG,left)
        head = self.segm.headSeg(imageWBG)
        
        tmp = armLeft[0]+armLeft[1]+armRight[0]+armRight[1]+legRight[0]+legRight[1]+legLeft[0]+legLeft[1]+head
        MidBdyImage =((imageWBG-(tmp>0))>0)

        body = ( self.segm.GetBody( MidBdyImage)>0)
        handRight = ( self.segm.GetHand( MidBdyImage,right)>0)
        handLeft = ( self.segm.GetHand( MidBdyImage,left)>0)
        footRight = ( self.segm.GetFoot( MidBdyImage,right)>0)
        footLeft = ( self.segm.GetFoot( MidBdyImage,left)>0)
        #pdb.set_trace()

        self.bdyPart = B*np.array( [armLeft[0], armLeft[1], armRight[0], armRight[1],\
                                  legLeft[0], legLeft[1], legRight[0],legRight[1],\
                                  head, body])#,  handRight, handLeft, footRight, footLeft])
        self.bdyColor = np.array( [np.array([0,0,255]), np.array([200,200,255]), np.array([0,255,0]), np.array([200,255,200]),\
                                   np.array([255,0,255]), np.array([255,180,255]), np.array([255,255,0]), np.array([255,255,180]),\
                                   np.array([255,0,0]), np.array([255,255,255])])#,  handRight, handLeft, footRight, footLeft])    
        '''
        correspondance between number and body parts and color
        armLeft[0] = forearmL      color = [0,0,255]          blue
        armLeft[1] = upperarmL     color = [200,200,255]      very light blue
        armRight[0]= forearmR      color = [0,255,0]          green
        armRight[1] = upperarmR    color = [200,255,200]      very light green
        legRight[0] = thighR       color = [255,0,255]        purple
        legRight[1] = calfR        color = [255,180,255]      pink
        legLeft[0] = thighL        color = [255,255,0]        yellow
        legLeft[1] = calfL         color = [255,255,180]      very light yellow
        head = headB               color = [255,0,0]          red
        body = body                color = [255,255,255]      white
        handRight = right hand     color = [0,191,255]        turquoise
        handLeft = left hand       color = [0,100,0]          dark green
        footRight = right foot     color = [199,21,133]       dark purple
        footLeft = left foot       color = [255,165,0]        orange
        '''
        
        # For Channel color R
        I = I +self.bdyColor[0,0]*armLeft[0]
        I = I +self.bdyColor[1,0]*armLeft[1]
        I = I +self.bdyColor[2,0]*armRight[0]
        I = I +self.bdyColor[3,0]*armRight[1]
        I = I +self.bdyColor[4,0]*legRight[0]
        I = I +self.bdyColor[5,0]*legRight[1]
        I = I +self.bdyColor[6,0]*legLeft[0]
        I = I +self.bdyColor[7,0]*legLeft[1]
        I = I +self.bdyColor[8,0]*head
        I = I +self.bdyColor[9,0]*body
        I = I +0*handRight
        I = I +0*handLeft
        I = I +199*footRight
        I = I +255*footLeft
        segImg[:,:,0,self.Index]=I
    
        # For Channel color G
        #I =  (np.zeros([self.Size[0],self.Size[1]])).astype(np.int8)
        I =  (np.zeros([self.BBox.shape[0],self.BBox.shape[1]])).astype(np.int8)
        I = I +self.bdyColor[0,1]*armLeft[0]
        I = I +self.bdyColor[1,1]*armLeft[1]
        I = I +self.bdyColor[2,1]*armRight[0]
        I = I +self.bdyColor[3,1]*armRight[1]
        I = I +self.bdyColor[4,1]*legRight[0]
        I = I +self.bdyColor[5,1]*legRight[1]
        I = I +self.bdyColor[6,1]*legLeft[0]
        I = I +self.bdyColor[7,1]*legLeft[1]
        I = I +self.bdyColor[8,1]*head
        I = I +self.bdyColor[9,1]*body
        I = I +100*handRight
        I = I +191*handLeft
        I = I +21*footRight
        I = I +165*footLeft        
        segImg[:,:,1,self.Index] = I
    
        # For Channel color B
        #I =  (np.zeros([self.Size[0],self.Size[1]])).astype(np.int8)
        I =  (np.zeros([self.BBox.shape[0],self.BBox.shape[1]])).astype(np.int8)
        I = I +self.bdyColor[0,2]*armLeft[0]
        I = I +self.bdyColor[1,2]*armLeft[1]
        I = I +self.bdyColor[2,2]*armRight[0]
        I = I +self.bdyColor[3,2]*armRight[1]
        I = I +self.bdyColor[4,2]*legRight[0]
        I = I +self.bdyColor[5,2]*legRight[1]
        I = I +self.bdyColor[6,2]*legLeft[0]
        I = I +self.bdyColor[7,2]*legLeft[1]
        I = I +self.bdyColor[8,2]*head
        I = I +self.bdyColor[9,2]*body
        I = I +0*handRight
        I = I +255*handLeft
        I = I +133*footRight
        I = I +0*footLeft        
        segImg[:,:,2,self.Index] = I
    
        elapsed_time = time.time() - start_time
        print "Segmentation: %f" % (elapsed_time)
        return segImg[:,:,:,self.Index]

    
###################################################################
################### Bounding boxes Function #######################
##################################################################      
    def BodyBBox(self):       
        '''This will generate a new depthframe but focuses on the human body'''
        pos2D = self.pos2d[0,self.Index].astype(np.int16)
        # extremes points of the bodies
        minV = np.min(pos2D[:,1])
        maxV = np.max(pos2D[:,1])
        minH = np.min(pos2D[:,0])
        maxH = np.max(pos2D[:,0])
        # distance head to neck. Let us assume this is enough for all borders
        distH2N = LA.norm( (pos2D[self.connection[0,1]-1]-pos2D[self.connection[0,0]-1])).astype(np.int16)
        Box = self.lImages[0,self.Index]
        bwBox = self.bw[0,self.Index]
        ############ Should check whether the value are in the frame #####################
        colStart = (minH-distH2N).astype(np.int16)
        lineStart = (minV-distH2N).astype(np.int16)
        colEnd = (maxH+distH2N).astype(np.int16)
        lineEnd = (maxV+distH2N).astype(np.int16)  
        self.transBB = np.array([colStart,lineStart,colEnd,lineEnd])
        self.BBox = Box[lineStart:lineEnd,colStart:colEnd]
        self.BBPos = (pos2D -self.transBB[0:2]).astype(np.int16)
        self.BBbw = bwBox[lineStart:lineEnd,colStart:colEnd]
        

    def SetTransfoMat(self,evecs,i,ctrMass):       
        '''Generate the transformation matrix '''
        e1 = evecs[0]
        e2 = evecs[1]
        e3 = evecs[2]
#==============================================================================
#         x = np.min(np.dot(self.VtxBB[i],e1))*e1
#         y = np.min(np.dot(self.VtxBB[i],e2))*e2
#         z = np.min(np.dot(self.VtxBB[i],e3))*e3
#         center = x + y + z
#==============================================================================
        e1b = np.array( [e1[0],e1[1],e1[2],0])
        e2b = np.array( [e2[0],e2[1],e2[2],0])
        e3b = np.array( [e3[0],e3[1],e3[2],0])
        center = (ctrMass[0]+ctrMass[1]+ctrMass[2])/3
        origine = np.array( [center[0],center[1],center[2],1])
        Transfo = np.stack( (e1b,e2b,e3b,origine),axis = 0 )
        self.TransfoBB.append(Transfo.transpose())
        print self.TransfoBB[i]           
           
    def myPCA(self, dims_rescaled_data=3):
        """
        returns: data transformed 
        """
        self.TVtxBB = []
        self.TransfoBB = []
        shift = self.transBB
        for i in range(self.bdyPart.shape[0]):
            ctrMass = []
            data = np.zeros(self.VtxBB[i].shape)
            for j in range(3):
                # center of mass the data
                idx = ndimage.measurements.center_of_mass(self.VtxBB[i][:,:,j])
                ctrMass.append(np.array([idx[1],idx[1],j]))
                data[:,:,j] = self.VtxBB[i][:,:,j]-self.VtxBB[i][int(round(idx[0])),int(round(idx[1])),j]
            data_cov = np.zeros( [3,3])                
            # compute the covariance matrix  
            for j in range(3):
                for k in range(3):  
                    #compute each term of the covariancecovariance matrix
                    data_cov[j,k] = np.sum(np.dot(data[:,:,j],data[:,:,k].T))/data.shape[0]
            # calculate eigenvectors & eigenvalues of the covariance matrix
            # use 'eigh' rather than 'eig' since data_cov is symmetric, 
            # the performance gain is substantial
            uu,s,vv = np.linalg.svd(data_cov)
            # sort eigenvalue in decreasing order
            idx = np.argsort(s)[::-1]
            vv = vv[:,idx]
            # sort eigenvectors according to same index
            s = s[idx]
            # select the first n eigenvectors (n is desired dimension
            # of rescaled data array, or dims_rescaled_data)
            vv = vv[:, :dims_rescaled_data]
            # carry out the transformation on the data using eigenvectors
            # and return the re-scaled data, eigenvalues, and eigenvectors
            self.TVtxBB.append( np.dot(self.VtxBB[i],vv))
#==============================================================================
#             print 'TVtxBB[%d]' %(i)
#             print self.TVtxBB[i]
#==============================================================================
            self.SetTransfoMat(uu,i,ctrMass)       

            
    def FindCoord(self, dims_rescaled_data=3):       
        '''
        draw the bounding boxes in 3D for each part of the human body
        '''     
        self.coords=[]
        self.coordsT=[]
        self.borders = []
        for i in range(self.bdyPart.shape[0]):
            # extremes planes of the bodies
            
            minX = np.min(self.TVtxBB[i][:,:,0][np.nonzero(self.TVtxBB[i][:,:,0])])#np.min(self.TVtxBB[i][:,:,0])
            maxX = np.max(self.TVtxBB[i][:,:,0][np.nonzero(self.TVtxBB[i][:,:,0])])#np.max(self.TVtxBB[i][:,:,0])
            minY = np.min(self.TVtxBB[i][:,:,1][np.nonzero(self.TVtxBB[i][:,:,1])])#np.min(self.TVtxBB[i][:,:,1])
            maxY = np.max(self.TVtxBB[i][:,:,1][np.nonzero(self.TVtxBB[i][:,:,1])])#np.max(self.TVtxBB[i][:,:,1])
            minZ = np.min(self.TVtxBB[i][:,:,2][np.nonzero(self.TVtxBB[i][:,:,2])])#np.min(self.TVtxBB[i][:,:,2])
            maxZ = np.max(self.TVtxBB[i][:,:,2][np.nonzero(self.TVtxBB[i][:,:,2])])#np.max(self.TVtxBB[i][:,:,2])
            self.borders.append( np.array([minX,maxX,minY,maxY,minZ,maxZ]) )
            # extremes points of the bodies
            xymz = np.array([minX,minY,minZ]).astype(np.int16)
            xYmz = np.array([minX,maxY,minZ]).astype(np.int16)            
            Xymz = np.array([maxX,minY,minZ]).astype(np.int16)
            XYmz = np.array([maxX,maxY,minZ]).astype(np.int16)
            xymZ = np.array([minX,minY,maxZ]).astype(np.int16)
            xYmZ = np.array([minX,maxY,maxZ]).astype(np.int16)
            XymZ = np.array([maxX,minY,maxZ]).astype(np.int16)
            XYmZ = np.array([maxX,maxY,maxZ]).astype(np.int16)           
            # New coordinates and new images
            self.coordsT.append( np.array([xymz,xYmz,XYmz,Xymz,xymZ,xYmZ,XYmZ,XymZ]) )
            print "coordsT[%d]" %(i)
            print self.coordsT[i]
            inv = np.linalg.inv(self.TransfoBB[i][0:3,0:3])
            self.coords.append(np.dot(self.coordsT[i],inv.T))
            print "coord[%d]" %(i)
            print self.coords[i]
            

    def GetCorners(self, Pose, s=1, color = 0) :   
        self.drawCorners = []
        self.drawCenter = []
        for k in range(self.bdyPart.shape[0]):
            # coordinates of boxes corners
            Size = self.coords[k].shape
            tmp = np.zeros([Size[0],2])
            Vtx = self.coords[k]  
            line_index = 0
            column_index = 0
            pix = np.array([0., 0., 1.])
            pt = np.array([0., 0., 0., 1.])
            for i in range(Size[0]/s):
                pt[0] = Vtx[i*s][0]
                pt[1] = Vtx[i*s][1]
                pt[2] = Vtx[i*s][2]
                pt = np.dot(Pose, pt)
                if (pt[2] != 0.0):
                    pix[0] = pt[0]/pt[2]
                    pix[1] = pt[1]/pt[2]
                    pix = np.dot(self.intrinsic, pix)
                    column_index = int(round(pix[0]))
                    line_index = int(round(pix[1]))
                    tmp[i,0] = line_index
                    tmp[i,1] = column_index
            self.drawCorners.append(tmp) 
            print "drawCorners[%d]" %(k)
            print self.drawCorners[k]
            # coordinates system
            tmp = np.zeros([1,2])
            bord= self.TransfoBB[k]  
            line_index = 0
            column_index = 0
            pix = np.array([0., 0., 1.])
            pt = np.array([0., 0., 0., 1.])
            pt[0] = bord[0][3]
            pt[1] = bord[1][3]
            pt[2] = bord[2][3]
            pt = np.dot(Pose, pt)
            if (pt[2] != 0.0):
                pix[0] = pt[0]/pt[2]
                pix[1] = pt[1]/pt[2]
                pix = np.dot(self.intrinsic, pix)
                column_index = int(round(pix[0]))
                line_index = int(round(pix[1]))
                tmp[0,0] = line_index
                tmp[0,1] = column_index
            self.drawCenter.append(tmp) 
            print "drawCenter[%d]" %(k)
            print self.drawCenter[k]


    def Cvt2RGBA(self,im_im):
        '''
        convert an RGB image in RGBA to put all zeros as transparent
        '''
        img = im_im.convert("RGBA")
        datas = img.getdata()     
        newData = []
        for item in datas:
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                newData.append((0, 0, 0, 0))
            else:
                newData.append(item)
        
        img.putdata(newData)
        return img      

                
    def ChangeColors(self,im_im,color):
        '''
        take an RGBA image to color it 
        '''
        img = im_im.convert("RGBA")
        datas = img.getdata()     
        newData = []
        for item in datas:
            if item[0] != 0 and item[1] != 0 and item[2] != 0:
                newData.append((color[0], color[1], color[2], item[3]))
            else:
                newData.append(item)
        
        img.putdata(newData)
        return img                
            
            

            
                