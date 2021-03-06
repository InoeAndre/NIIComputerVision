"""
 File created by Diego Thomas the 16-11-2016
 improved by Inoe Andre from 02-2017

 Define functions to manipulate RGB-D data
"""
import cv2
import numpy as np
from numpy import linalg as LA
import random
import imp
import time
import scipy.ndimage.measurements as spm
import pdb
from skimage import img_as_ubyte
from sklearn.decomposition import PCA


segm = imp.load_source('segmentation', './lib/segmentation.py')
General = imp.load_source('General', './lib/General.py')



class RGBD():
    """
    Class to handle any processing on depth image and the image breed from the depth image
    """

    def __init__(self, depthname, colorname, intrinsic, fact):
        """
        Constructor
        :param depthname: path to a depth image
        :param colorname: path to a RGBD image
        :param intrinsic: matrix with calibration parameters
        :param fact: factor for converting pixel value to meter or conversely
        """
        self.depthname = depthname
        self.colorname = colorname
        self.intrinsic = intrinsic
        self.fact = fact
        
    def LoadMat(self, Images,Pos_2D,BodyConnection,binImage):
        """
        Load information in datasets into the RGBD object
        :param Images: List of depth images put in function of time
        :param Pos_2D: List of junctions position for each depth image
        :param BodyConnection: list of doublons that contains the number of pose that represent adjacent body parts
        :param binImage: Binary image with the body suppodly in white
        :return:  none
        """
        self.lImages = Images
        self.numbImages = len(self.lImages.transpose())
        self.Index = -1
        self.pos2d = Pos_2D
        self.connection = BodyConnection
        self.bw = binImage
        
    def ReadFromDisk(self):
        """
        Read an RGB-D image from the disk
        :return: none
        """
        print(self.depthname)
        self.depth_in = cv2.imread(self.depthname, -1)
        self.color_image = cv2.imread(self.colorname, -1)
        
        self.Size = self.depth_in.shape
        self.depth_image = np.zeros((self.Size[0], self.Size[1]), np.float32)
        for i in range(self.Size[0]): # line index (i.e. vertical y axis)
            for j in range(self.Size[1]):
                self.depth_image[i,j] = float(self.depth_in[i,j][0]) / self.fact
                                
    def ReadFromMat(self, idx = -1):
        """
        Read an RGB-D image from matrix (dataset)
        :param idx: number of the
        :return:
        """
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

    #####################################################################
    ################### Map Conversion Functions #######################
    #####################################################################
    
    def Vmap(self):
        """
        Create the vertex image from the depth image and intrinsic matrice
        :return: none
        """
        self.Vtx = np.zeros(self.Size, np.float32)
        for i in range(self.Size[0]): # line index (i.e. vertical y axis)
            for j in range(self.Size[1]): # column index (i.e. horizontal x axis)
                d = self.depth_image[i,j]
                if d > 0.0:
                    x = d*(j - self.intrinsic[0,2])/self.intrinsic[0,0]
                    y = d*(i - self.intrinsic[1,2])/self.intrinsic[1,1]
                    self.Vtx[i,j] = (x, y, d)
        
    
    def Vmap_optimize(self):
        """
        Create the vertex image from the depth image and intrinsic matrice
        :return: none
        """
        #self.Vtx = np.zeros(self.Size, np.float32)
        #matrix containing depth value of all pixel
        d = self.depth_image[0:self.Size[0]][0:self.Size[1]]
        d_pos = d * (d > 0.0)
        # create matrix that contains index values
        x_raw = np.zeros([self.Size[0],self.Size[1]], np.float32)
        y_raw = np.zeros([self.Size[0],self.Size[1]], np.float32)
        # change the matrix so that the first row is on all rows for x respectively colunm for y.
        x_raw[0:-1,:] = ( np.arange(self.Size[1]) - self.intrinsic[0,2])/self.intrinsic[0,0]
        y_raw[:,0:-1] = np.tile( ( np.arange(self.Size[0]) - self.intrinsic[1,2])/self.intrinsic[1,1],(1,1)).transpose()
        # multiply point by point d_pos and raw matrices
        x = d_pos * x_raw
        y = d_pos * y_raw
        self.Vtx = np.dstack((x, y,d))
        return self.Vtx

    def NMap(self):
        """
        Compute normal map
        :return: none
        """
        self.Nmls = np.zeros(self.Size, np.float32)
        for i in range(1,self.Size[0]-1):
            for j in range(1, self.Size[1]-1):
                # normal for each direction
                nmle1 = General.normalized_cross_prod(self.Vtx[i+1, j]-self.Vtx[i, j], self.Vtx[i, j+1]-self.Vtx[i, j])
                nmle2 = General.normalized_cross_prod(self.Vtx[i, j+1]-self.Vtx[i, j], self.Vtx[i-1, j]-self.Vtx[i, j])
                nmle3 = General.normalized_cross_prod(self.Vtx[i-1, j]-self.Vtx[i, j], self.Vtx[i, j-1]-self.Vtx[i, j])
                nmle4 = General.normalized_cross_prod(self.Vtx[i, j-1]-self.Vtx[i, j], self.Vtx[i+1, j]-self.Vtx[i, j])
                nmle = (nmle1 + nmle2 + nmle3 + nmle4)/4.0
                # normalized
                if (LA.norm(nmle) > 0.0):
                    nmle = nmle/LA.norm(nmle)
                self.Nmls[i, j] = (nmle[0], nmle[1], nmle[2])
                
    def NMap_optimize(self):
        """
        Compute normal map, CPU optimize algo
        :return: none
        """
        self.Nmls = np.zeros(self.Size, np.float32)
        # matrix of normales for each direction
        nmle1 = General.normalized_cross_prod_optimize(self.Vtx[2:self.Size[0]  ][:,1:self.Size[1]-1] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1], \
                                               self.Vtx[1:self.Size[0]-1][:,2:self.Size[1]  ] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1])        
        nmle2 = General.normalized_cross_prod_optimize(self.Vtx[1:self.Size[0]-1][:,2:self.Size[1]  ] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1], \
                                               self.Vtx[0:self.Size[0]-2][:,1:self.Size[1]-1] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1])
        nmle3 = General.normalized_cross_prod_optimize(self.Vtx[0:self.Size[0]-2][:,1:self.Size[1]-1] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1], \
                                               self.Vtx[1:self.Size[0]-1][:,0:self.Size[1]-2] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1])
        nmle4 = General.normalized_cross_prod_optimize(self.Vtx[1:self.Size[0]-1][:,0:self.Size[1]-2] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1], \
                                               self.Vtx[2:self.Size[0]  ][:,1:self.Size[1]-1] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1])
        nmle = (nmle1 + nmle2 + nmle3 + nmle4)/4.0
        # normalized
        norm_mat_nmle = np.sqrt(np.sum(nmle*nmle,axis=2))
        norm_mat_nmle = General.in_mat_zero2one(norm_mat_nmle)
        #norm division 
        nmle = General.division_by_norm(nmle,norm_mat_nmle)
        self.Nmls[1:self.Size[0]-1][:,1:self.Size[1]-1] = nmle
        return self.Nmls

    #############################################################################
    ################### Projection and transform Functions #######################
    #############################################################################

    def Draw(self, Pose, s, color = 0) :
        """
        Project vertices and normales in 2D images
        :param Pose: camera pose
        :param s: subsampling the cloud of points
        :param color: if there is a color image put color in the image
        :return: scene projected in 2D space
        """
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


    def Draw_optimize(self, rendering,Pose, s, color = 0) :
        """
        Project vertices and normales from an RGBD image in 2D images
        :param rendering : 2D image for overlay purpose or black image
        :param Pose: camera pose
        :param s: subsampling the cloud of points
        :param color: if there is a color image put color in the image
        :return: scene projected in 2D space
        """
        result = rendering#np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
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
        lpt[2] = General.in_mat_zero2one(lpt[2])
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


    def DrawMesh(self, rendering,Vtx,Nmls,Pose, s, color = 0) :
        """
        Project vertices and normales from a mesh in 2D images
        :param rendering : 2D image for overlay purpose or black image
        :param Pose: camera pose
        :param s: subsampling the cloud of points
        :param color: if there is a color image put color in the image
        :return: scene projected in 2D space
        """
        result = rendering#np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)#
        stack_pix = np.ones( (np.size(Vtx[ ::s,:],0)) , dtype = np.float32)
        stack_pt = np.ones( (np.size(Vtx[ ::s,:],0)) , dtype = np.float32)
        pix = np.zeros( (np.size(Vtx[ ::s,:],0),2) , dtype = np.float32)
        pix = np.stack((pix[:,0],pix[:,1],stack_pix),axis = 1)
        pt = np.stack( (Vtx[ ::s,0],Vtx[ ::s,1],Vtx[ ::s,2],stack_pt),axis =1 )
        pt = np.dot(pt,Pose.T)

        nmle = np.zeros((Nmls.shape[0], Nmls.shape[1]), dtype = np.float32)
        nmle[ ::s,:] = np.dot(Nmls[ ::s,:],Pose[0:3,0:3].T)
        

        # projection in 2D space
        lpt = np.split(pt,4,axis=1)
        lpt[2] = General.in_mat_zero2one(lpt[2])
        pix[ ::s,0] = (lpt[0]/lpt[2]).reshape(np.size(Vtx[ ::s,:],0))
        pix[ ::s,1] = (lpt[1]/lpt[2]).reshape(np.size(Vtx[ ::s,:],0))
        pix = np.dot(pix,self.intrinsic.T)

        column_index = (np.round(pix[:,0])).astype(int)
        line_index = (np.round(pix[:,1])).astype(int)
        # create matrix that have 0 when the conditions are not verified and 1 otherwise
        cdt_column = (column_index > -1) * (column_index < self.Size[1])
        cdt_line = (line_index > -1) * (line_index < self.Size[0])
        line_index = line_index*cdt_line
        column_index = column_index*cdt_column
        if (color == 0):
            result[line_index[:], column_index[:]]= np.dstack((self.color_image[ ::s, ::s,2], \
                                                                    self.color_image[ ::s, ::s,1]*cdt_line, \
                                                                    self.color_image[ ::s, ::s,0]*cdt_column) )
        else:
            result[line_index[:], column_index[:]]= np.dstack( ( (nmle[ :,0]+1.0)*(255./2.), \
                                                                       ((nmle[ :,1]+1.0)*(255./2.))*cdt_line, \
                                                                       ((nmle[ :,2]+1.0)*(255./2.))*cdt_column ) ).astype(int)
        return result  


    def Transform(self, Pose):
        """
        Transform Vertices and Normales with the Pose matrix (generally camera pose matrix)
        :param Pose: 4*4 Transformation Matrix
        :return: none
        """
        stack_pt = np.ones((np.size(self.Vtx,0), np.size(self.Vtx,1)), dtype = np.float32)
        pt = np.dstack((self.Vtx, stack_pt))
        self.Vtx = np.dot(Pose,pt.transpose(0,2,1)).transpose(1,2,0)[:, :, 0:3]
        self.Nmls = np.dot(Pose[0:3,0:3],self.Nmls.transpose(0,2,1)).transpose(1,2,0)

##################################################################
###################Bilateral Smooth Funtion#######################
##################################################################
    def BilateralFilter(self, d, sigma_color, sigma_space):
        """
        Bilateral filtering the depth image
        see cv2 documentation
        """
        self.depth_image = (self.depth_image[:,:] > 0.0) * cv2.bilateralFilter(self.depth_image, d, sigma_color, sigma_space)
        

##################################################################
################### Segmentation Function #######################
##################################################################
    def RemoveBG(self,binaryImage):
        """
        Delete all the little group (connected component) unwanted from the binary image
        :param binaryImage: a binary image containing several connected component
        :return: A binary image containing only big connected component
        """
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

    def Crop2Body(self):
        """
        Generate a cropped depthframe from the previous one. The new frame focuses on the human body
        :return: none
        """
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
        self.transCrop = np.array([colStart,lineStart,colEnd,lineEnd])
        self.CroppedBox = Box[lineStart:lineEnd,colStart:colEnd]
        self.CroppedPos = (pos2D -self.transCrop[0:2]).astype(np.int16)
        self.Croppedbw = bwBox[lineStart:lineEnd,colStart:colEnd]
        
    def BdyThresh(self):
        """
        Threshold the depth image in order to to get the whole body alone with the bounding box (BB)
        :return: The connected component that contain the body
        """
        pos2D = self.CroppedPos
        max_value = np.iinfo(np.uint16).max # = 65535 for uint16
        self.CroppedBox = self.CroppedBox.astype(np.uint16)
        # Threshold according to detph of the body
        bdyVals = self.CroppedBox[pos2D[self.connection[:,0]-1,1]-1,pos2D[self.connection[:,0]-1,0]-1]
        #only keep vales different from 0
        bdy = bdyVals[np.nonzero(bdyVals != 0)]
        mini =  np.min(bdy)
        #print "mini: %u" % (mini)
        maxi = np.max(bdy)
        #print "max: %u" % (maxi)
        # double threshold according to the value of the depth
        bwmin = (self.CroppedBox > mini-0.01*max_value) 
        bwmax = (self.CroppedBox < maxi+0.01*max_value)
        bw0 = bwmin*bwmax
        # Compare with the noised binary image given by the kinect
        # to use this put res instead of bw0 as the return argument
        thresh2,tmp = cv2.threshold(self.Croppedbw,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        res = tmp * bw0
        # Remove all stand alone object
        bw0 = ( self.RemoveBG(bw0)>0)
        return bw0#res

    def BodySegmentation(self):
        """
        Calls the function in segmentation.py to process the segmentation of the body
        :return:  none
        """
        #Initialized segmentation with the cropped image
        self.segm = segm.Segmentation(self.CroppedBox,self.CroppedPos) 
        # binary image without bqckground
        imageWBG = (self.BdyThresh()>0)

        # Cropped image
        B = self.CroppedBox

        right = 0
        left = 1
        # Process to segmentation algorithm
        armLeft = self.segm.armSeg(imageWBG,B,left)
        armRight = self.segm.armSeg(imageWBG,B,right)
        legRight = self.segm.legSeg(imageWBG,right)
        legLeft = self.segm.legSeg(imageWBG,left)
        head = self.segm.headSeg(imageWBG)
        
        # Retrieve every already segmentated part to the main body.
        tmp = armLeft[0]+armLeft[1]+armRight[0]+armRight[1]+legRight[0]+legRight[1]+legLeft[0]+legLeft[1]+head
        MidBdyImage =((imageWBG-(tmp>0))>0)

        # display result
        # cv2.imshow('trunk' , MidBdyImage.astype(np.float))
        # cv2.waitKey(0)

        # continue segmentation for hands and feet
        handRight = ( self.segm.GetHand( MidBdyImage,right)>0)
        handLeft = ( self.segm.GetHand( MidBdyImage,left)>0)
        footRight = ( self.segm.GetFoot( MidBdyImage,right)>0)
        footLeft = ( self.segm.GetFoot( MidBdyImage,left)>0)

        # display the trunck
        # cv2.imshow('trunk' , MidBdyImage.astype(np.float))
        # cv2.waitKey(0)

        # Retrieve again every newly computed segmentated part to the main body.
        tmp2 = handRight+handLeft+footRight+footLeft
        MidBdyImage2 =((MidBdyImage-(tmp2>0))>0)

        # Display result
        # cv2.imshow('MidBdyImage2' , MidBdyImage2.astype(np.float))
        # cv2.waitKey(0)
        body = ( self.segm.GetBody( MidBdyImage2)>0)

        # cv2.imshow('body' , body.astype(np.float))
        # cv2.waitKey(0)
        #pdb.set_trace()

        # list of each body parts
        self.bdyPart = np.array( [ armLeft[0], armLeft[1], armRight[0], armRight[1], \
                                   legRight[0], legRight[1], legLeft[0], legLeft[1], \
                                   head, body, handRight, handLeft, footLeft,footRight ]).astype(np.int)#]).astype(np.int)#]).astype(np.int)#
        # list of color for each body parts
        self.bdyColor = np.array( [np.array([0,0,255]), np.array([200,200,255]), np.array([0,255,0]), np.array([200,255,200]),\
                                   np.array([255,0,255]), np.array([255,180,255]), np.array([255,255,0]), np.array([255,255,180]),\
                                   np.array([255,0,0]), np.array([255,255,255]),np.array([0,100,0]),np.array([0,191,255]),\
                                   np.array([255,165,0]),np.array([199,21,133]) ])    
        self.labelColor = np.array( ["#0000ff", "#ffc8ff", "#00ff00","#c8ffc8","#ff00ff","#ffb4ff",\
                                   "#ffff00","#ffffb4","#ff0000","#ffffff","#00bfff","#006400",\
                                   "#c715ff","#ffa500"])    

        '''
        correspondance between number and body parts and color
        background should have :   color = [0,0,0]       = #000000     black                 label = 0
        armLeft[0] = forearmL      color = [0,0,255]     = #0000ff     blue                  label = 1
        armLeft[1] = upperarmL     color = [200,200,255] = #ffc8ff     very light blue       label = 2
        armRight[0]= forearmR      color = [0,255,0]     = #00ff00     green                 label = 3
        armRight[1] = upperarmR    color = [200,255,200] = #c8ffc8     very light green      label = 4
        legRight[0] = thighR       color = [255,0,255]   = #ff00ff     purple                label = 5
        legRight[1] = calfR        color = [255,180,255] = #ffb4ff     pink                  label = 6
        legLeft[0] = thighL        color = [255,255,0]   = #ffff00     yellow                label = 7
        legLeft[1] = calfL         color = [255,255,180] = #ffffb4     very light yellow     label = 8
        head = headB               color = [255,0,0]     = #ff0000     red                   label = 9
        body = body                color = [255,255,255] = #ffffff     white                 label = 10
        handRight = right hand     color = [0,191,255]   = #00bfff     turquoise             label = 11
        handLeft = left hand       color = [0,100,0]     = #006400     dark green            label = 12
        footRight = right foot     color = [199,21,133]  = #c715ff     dark purple           label = 13
        footLeft = left foot       color = [255,165,0]   = #ffa500     orange                label = 14
        '''
        

    def BodyLabelling(self):
        '''Create label for each body part in the depth_image'''
        Size = self.depth_image.shape
        self.labels = np.zeros(Size,np.int)   
        Txy = self.transCrop
        for i in range(self.bdyPart.shape[0]): 
            self.labels[Txy[1]:Txy[3],Txy[0]:Txy[2]] += (i+1)*self.bdyPart[i]
            # if some parts overlay, the number of this part will bigger
            overlap = np.where(self.labels > (i+1) )
            #put the overlapping part in the following body part
            self.labels[overlap] = i+1

    def RGBDSegmentation(self):
        """
        Call every method to have a complete segmentation
        :return: none
        """
        self.Crop2Body()
        self.BodySegmentation()
        self.BodyLabelling()



#######################################################################
################### Bounding boxes Function #######################
##################################################################             

    def GetCenter3D(self,i):
        """
        Compute the mean for one segmented part
        :param i: number of the body part
        :return: none
        """
        mean3D = np.mean(self.PtCloud[i],axis = 0)
        return mean3D

        
    def SetTransfoMat3D(self,evecs,i):
        """
        Generate the transformation matrix
        :param evecs: eigen vectors
        :param i: number of the body part
        :return: none
        """
        ctr = self.ctr3D[i]#self.coordsGbl[i][0]#[0.,0.,0.]#
        e1 = evecs[0]
        e2 = evecs[1]
        e3 = evecs[2]
        # axis of coordinates system
        e1b = np.array( [e1[0],e1[1],e1[2],0])
        e2b = np.array( [e2[0],e2[1],e2[2],0])
        e3b = np.array( [e3[0],e3[1],e3[2],0])
        #center of coordinates system
        origine = np.array( [ctr[0],self.ctr3D[i][1],ctr[2],1])
        # concatenate it in the right order.
        Transfo = np.stack( (e1b,e2b,e3b,origine),axis = 0 )
        self.TransfoBB.append(Transfo.transpose())
        #display
        #print "TransfoBB[%d]" %(i)
        #print self.TransfoBB[i]        
        
        
    def bdyPts3D(self, mask):
        """
        create of cloud of point from part of the RGBD image
        :param mask: a matrix containing one only in the body parts indexes, 0 otherwise
        :return:  list of vertices = cloud of points
        """
        start_time2 = time.time()
        nbPts = sum(sum(mask))
        res = np.zeros((nbPts, 3), dtype = np.float32)
        k = 0
        for i in range(self.Size[0]):
            for j in range(self.Size[1]):
                if(mask[i,j]):
                    res[k] = self.Vtx[i,j]
                    k = k+1
        elapsed_time3 = time.time() - start_time2
        print "making pointcloud process time: %f" % (elapsed_time3)       
        return res

    def bdyPts3D_optimize(self, mask):
        """
        create of cloud of point from part of the RGBD image
        :param mask: a matrix containing one only in the body parts indexes, 0 otherwise
        :return:  list of vertices = cloud of points
        """
        #start_time2 = time.time()
        nbPts = sum(sum(mask))

        # threshold with the mask
        x = self.Vtx[:,:,0]*mask
        y = self.Vtx[:,:,1]*mask
        z = self.Vtx[:,:,2]*mask

        #keep only value that are different from 0 in the list
        x_res = x[~(x==0)]
        y_res = y[~(y==0)]
        z_res = z[~(z==0)]

        #concatenate each axis
        res = np.dstack((x_res,y_res,z_res)).reshape(nbPts,3)

        #elapsed_time3 = time.time() - start_time2
        #print "making pointcloud process time: %f" % (elapsed_time3)    

        return res
    
    
           
    def myPCA(self, dims_rescaled_data=3):
        """
        Compute the principal component analysis on a cloud of points
        to get the coordinates system local to the cloud of points
        :param dims_rescaled_data: 3 per default, number of dimension wanted
        :return:  none
        """
        # list of center in the 3D space
        self.ctr3D = []
        self.ctr3D.append([0.,0.,0.])
        # list of transformed Vtx of each bounding boxes
        self.TVtxBB = []
        self.TVtxBB.append([0.,0.,0.])
        # list of coordinates sys with center
        self.TransfoBB = []
        self.TransfoBB.append([0.,0.,0.])
        self.vects3D = []
        self.vects3D.append([0.,0.,0.])
        self.PtCloud = []
        self.PtCloud.append([0.,0.,0.])
        self.pca = []
        self.pca.append(PCA(n_components=3))
        self.coordsL=[]
        self.coordsL.append([0.,0.,0.])
        self.coordsGbl=[]
        self.coordsGbl.append([0.,0.,0.])
        self.mask=[]
        self.mask.append([0.,0.,0.])
        for i in range(1,self.bdyPart.shape[0]+1):
            self.mask.append( (self.labels == i) )
            # compute center of 3D
            self.PtCloud.append(self.bdyPts3D_optimize(self.mask[i]))
            self.pca.append(PCA(n_components=3))
            self.pca[i].fit(self.PtCloud[i]) 
            
            # Compute 3D centers
            self.ctr3D.append(self.GetCenter3D(i))         
            #print "ctr3D indexes :"
            #print self.ctr3D[i]

            # eigen vectors
            self.vects3D.append(self.pca[i].components_)
            #global to local transform of the cloud of point
            self.TVtxBB.append( self.pca[i].transform(self.PtCloud[i]))

            #Coordinates of the bounding boxes
            self.FindCoord3D(i)
            #Create local to global transform
            self.SetTransfoMat3D(self.pca[i].components_,i)  

            
    def FindCoord3D(self,i):       
        '''
        draw the bounding boxes in 3D for each part of the human body
        :param i : number of the body parts
        '''     
        # Adding a space so that the bounding boxes are wider
        VoxSize = 0.005
        wider = 5*VoxSize
        # extremes planes of the bodies
        minX = np.min(self.TVtxBB[i][:,0]) - wider
        maxX = np.max(self.TVtxBB[i][:,0]) + wider
        minY = np.min(self.TVtxBB[i][:,1]) - wider
        maxY = np.max(self.TVtxBB[i][:,1]) + wider
        minZ = np.min(self.TVtxBB[i][:,2]) - wider
        maxZ = np.max(self.TVtxBB[i][:,2]) + wider
        # extremes points of the bodies
        xymz = np.array([minX,minY,minZ])
        xYmz = np.array([minX,maxY,minZ])           
        Xymz = np.array([maxX,minY,minZ])
        XYmz = np.array([maxX,maxY,minZ])
        xymZ = np.array([minX,minY,maxZ])
        xYmZ = np.array([minX,maxY,maxZ])
        XymZ = np.array([maxX,minY,maxZ])
        XYmZ = np.array([maxX,maxY,maxZ])           

        # New coordinates and new images
        self.coordsL.append( np.array([xymz,xYmz,XYmz,Xymz,xymZ,xYmZ,XYmZ,XymZ]) )
        #print "coordsL[%d]" %(i)
        #print self.coordsL[i]
        
        # transform back
        self.coordsGbl.append( self.pca[i].inverse_transform(self.coordsL[i]))
        #print "coordsGbl[%d]" %(i)
        #print self.coordsGbl[i]
            

    def GetProjPts2D(self, vects3D, Pose, s=1) :
        """
        Project a list of vertexes in the image RGBD
        :param vects3D: list of 3 elements vector
        :param Pose: Transformation matrix
        :param s: subsampling coefficient
        :return: transformed list of 3D vector
        """
        pix = np.array([0., 0., 1.])
        pt = np.array([0., 0., 0., 1.])
        drawVects = []
        for i in range(len(vects3D)):
            pt[0] = vects3D[i][0]
            pt[1] = vects3D[i][1]
            pt[2] = vects3D[i][2]
            # transform list
            pt = np.dot(Pose, pt)
            #Project it in the 2D space
            if (pt[2] != 0.0):
                pix[0] = pt[0]/pt[2]
                pix[1] = pt[1]/pt[2]
                pix = np.dot(self.intrinsic, pix)
                column_index = pix[0].astype(np.int)
                line_index = pix[1].astype(np.int)
            else :
                column_index = 0
                line_index = 0
            #print "line,column index : (%d,%d)" %(line_index,column_index) 
            drawVects.append(np.array([column_index,line_index]))
        return drawVects
            
    def GetProjPts2D_optimize(self, vects3D, Pose, s=1) :
        """
        Project a list of vertexes in the image RGBD. Optimize for CPU version.
        :param vects3D: list of 3 elements vector
        :param Pose: Transformation matrix
        :param s: subsampling coefficient
        :return: transformed list of 3D vector
        """
        '''Project a list of vertexes in the image RGBD'''
        pix = np.array([0., 0., 1.])
        pt = np.array([0., 0., 0., 1.])
        pix = np.stack((pix for i in range(len(vects3D)) ))
        pt = np.stack((pt for i in range(len(vects3D)) ))
        pt[:,0:3] = vects3D
        # transform list
        pt = np.dot(pt,Pose.T)
        # Project it in the 2D space
        pt[:,2] = General.in_mat_zero2one(pt[:,2])
        pix[:,0] = pt[:,0]/pt[:,2]
        pix[:,1] = pt[:,1]/pt[:,2]
        pix = np.dot( pix,self.intrinsic.T)
        column_index = pix[:,0].astype(np.int)
        line_index = pix[:,1].astype(np.int)        
        drawVects = np.array([column_index,line_index]).T
        return drawVects            


            
    def GetNewSys(self, Pose,ctr2D,nbPix, s=1) : 
        '''
        compute the coordinates of the points that will create the coordinates system
        '''
        self.drawNewSys = []
        maxDepth = max(0.0001, np.max(self.Vtx[:,:,2]))

        for i in range(1,len(self.vects3D)):
            self.vects3D[i] = np.dot(self.vects3D[i],Pose[0:3,0:3].T )
            vect = self.vects3D[i]
            newPt = np.zeros(vect.shape)
            for j in range(vect.shape[0]):
                newPt[j][0] = ctr2D[i][0]-nbPix*vect[j][0]
                newPt[j][1] = ctr2D[i][1]-nbPix*vect[j][1]
                newPt[j][2] = vect[j][2]-nbPix*vect[j][2]/maxDepth
            self.drawNewSys.append(newPt)

            

    def Cvt2RGBA(self,im_im):
        '''
        convert an RGB image in RGBA to put all zeros as transparent
        THIS FUNCTION IS NOT USED IN THE PROJECT
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
                