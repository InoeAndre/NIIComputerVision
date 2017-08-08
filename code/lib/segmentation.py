"""
File created by Inoe ANDRE the 01-03-2017
Define functions to do the segmentation in a depthmap image
"""
import cv2
import numpy as np
from numpy import linalg as LA
import imp
import scipy as sp
import scipy.ndimage
import math
import time
import itertools
import scipy.ndimage.measurements as spm


'''These are the order of joints returned by the kinect adaptor.
    SpineBase = 0
    SpineMid = 1
    Neck = 2
    Head = 3
    ShoulderLeft = 4
    ElbowLeft = 5
    WristLeft = 6
    HandLeft = 7
    ShoulderRight = 8
    ElbowRight = 9
    WristRight = 10
    HandRight = 11
    HipLeft = 12
    KneeLeft = 13
    AnkleLeft = 14
    FootLeft = 15
    HipRight = 16
    KneeRight = 17
    AnkleRight = 18
    FootRight = 19
    SpineShoulder = 20
    HandTipLeft = 21
    ThumbLeft = 22
    HandTipRight = 23
    ThumbRight = 24
   ''' 
    
class Segmentation(object):
    """
    Segmentation process concerning body parts
    """
    def __init__(self, depthImage, pos2D):
        """
        Constructor
        :param depthImage: Cropped depth image of the current image
        :param pos2D: list of position of the junction
        """
        self.depthImage = depthImage
        self.pos2D = pos2D
        self.bodyPts = []


    def findSlope(self,A,B):
        """
        Get the slope of a line made from two point A and B or the distance in one axes
        :param A: point 1
        :param B: point 2
        :return: an array of coefficient
        a is the normalized distance in the x axis
        b is the normalized distance in the y axis
        c is the slope between the two points
        """
        A = A.astype(np.float32)
        B = B.astype(np.float32)
        # distance in Y axis
        diffY = B[1]-A[1]
        # distance in X axis
        diffX = A[0]-B[0]
        dist = np.sqrt(np.square(diffY) + np.square(diffX)) 
        a = diffY/dist # normalized distance
        b = diffX/dist # normalized distance
        #slope between two points
        c = -a*A[0]-b*A[1]
        return np.array([a,b,c])
    
    def inferedPoint(self,A,a,b,c,point,T=100):
        """
        Find two points that are the corners of the segmented part
        :param A: Depth Image
        :param a: dist x axe between two points
        :param b: dist y axe between two points
        :param c: slope between two points
        :param point: a junction
        :param T: max distance to find intersection
        :return: two intersection points between a slope and the edges of the body part
        """
        line = self.depthImage.shape[0]
        col = self.depthImage.shape[1]
        process_y = abs(a) > abs(b)
        # searching in y axis
        if process_y:
            y = int(point[1])
            # search an edge running through a slope with decreasing y
            while 1:
                y = y-1
                # keep track of the perpendicular slope
                x = int(np.round(-(b*y+c)/a))
                # if an edge is reached
                if A[y,x]==0:
                    x_up = x
                    y_up = y
                    break
                else:
                    # if the max distance is reached
                    distCdt = LA.norm([x,y]-point)>T
                    if distCdt:#sqrt((x-point(1))^2+(y-point(2))^2)>T:
                        x_up = x
                        y_up = y
                        break
            y = int(point[1])
            # search an edge running through a slope with increasing y
            while 1:
                y = y+1
                # keep track of the perpendicular slope
                x = int(np.round(-(b*y+c)/a))
                # if an edge is reached
                if A[y,x]==0:
                    x_down = x
                    y_down = y
                    break
                else:
                    # if the max distance is reached
                    distCdt = LA.norm([x,y]-point)>T
                    if distCdt:#math.sqrt((x-point(1))^2+(y-point(2))^2)>T:
                        x_down = x
                        y_down = y
                        break
    
            if x_up>x_down:
                right = [x_up, y_up]
                left = [x_down, y_down]
            else:
                left = [x_up, y_up]
                right = [x_down, y_down]
        # searching in x axis
        else:#process_x
            x = int(point[0])
            while 1:
                x = x-1
                # keep the track of the perpendicular slope
                y = int(np.round(-(a*x+c)/b))
                inImage = (x>0) and (x<=col) and (y>0) and (y<=line)
                if inImage:
                    # if an edge is reached
                    if A[int(y),int(x)]==0:
                        x_left = x
                        y_left = y
                        break
                    else:
                        # if the max distance is reached
                        distCdt = LA.norm([x,y]-point)>T
                        if distCdt:#sqrt((x-point(1))^2+(y-point(2))^2)>T
                            x_left = x
                            y_left = y
                            break
                else:
                    x_left = x+1
                    y_left = np.round(-(a*x_left+c)/b)
                    break
        
            x = int(point[0])
            while 1:
                x = x+1
                # keep the track of the perpendicular slope
                y = int(np.round(-(a*x+c)/b))
                inImage = (x>0) and (x<=col) and (y>0) and (y<=line)
                if inImage:
                    # if an edge is reached
                    if A[int(y),int(x)]==0:
                        x_right = x
                        y_right = y
                        break
                    else:
                        # if the max distance is reached
                        distCdt = LA.norm([x,y]-point)>T
                        if distCdt:#sqrt((x-point(1))^2+(y-point(2))^2)>T
                            x_right = x
                            y_right = y
                            break
                else:
                    x_right = x-1
                    y_right = int(np.round(-(a*x_right+c)/b))
                    break
            left = [x_left, y_left]
            right = [x_right, y_right]
        return [left, right]
    
    

    def polygon(self,slopes,ref,  limit  ):
        """
        Test the sign of alpha = (a[k]*j+b[k]*i+c[k])*ref[k]
        to know whether a point is within a polygon or not
        :param slopes: list of slopes defining a the border lines of the polygone
        :param ref:  a point inside the polygon
        :param limit: number of slopes
        :return: the body part filled with true.
        """
        start_time = time.time()
        line = self.depthImage.shape[0]
        col = self.depthImage.shape[1]
        res = np.zeros([line,col],np.bool)
        alpha = np.zeros([1,limit])
        # for each point in the image
        for i in range(line):
           for j in range(col):
               for k in range(limit):
                   # compare distance of the point to a slope
                   alpha[0][k] = (slopes[0][k]*j+slopes[1][k]*i+slopes[2][k])*ref[0,k]
               alpha_positif = (alpha >= 0)
               if alpha_positif.all():
                   res[i,j]=True
        elapsed_time = time.time() - start_time
        print "polygon: %f" % (elapsed_time)
        return res
   
    def polygon_optimize(self,slopes,ref,  limit  ):
        """
        Test the sign of alpha = (a[k]*j+b[k]*i+c[k])*ref[k]
        to know whether a point is within a polygon or not
        :param slopes: list of slopes defining a the border lines of the polygone
        :param ref:  a point inside the polygon
        :param limit: number of slopes
        :return: the body part filled with true.
        """
        #start_time = time.time()
        line = self.depthImage.shape[0]
        col = self.depthImage.shape[1]
        res = np.ones([line,col])
        
        #create a matrix containing in each pixel its indices
        lineIdx = np.array([np.arange(line) for _ in range(col)]).reshape(col,line).transpose()
        colIdx = np.array([np.arange(col) for _ in range(line)]).reshape(line,col)
        depthIdx = np.ones([line,col])
        ind = np.stack( (colIdx,lineIdx,depthIdx), axis = 2)
        
        alpha = np.zeros([line,col,limit])
        alpha= np.dot(ind,slopes)
        # for each k (line) if the points (ref and the current point in alpha) are on the same side then the operation is positiv
        for k in range(limit):
            alpha[:,:,k]=( (np.dot(alpha[:,:,k],ref[0][k])) >= 0)
        # make sure that each point are on the same side as the reference for all line of the polygon
        for k in range(limit):
            res = res*alpha[:,:,k ]
        #threshold the image so that only positiv values (same side as reference point) are kept.
        res = (res>0)
        #elapsed_time = time.time() - start_time
        #print "polygon_optimize: %f" % (elapsed_time)
        return res
        
    def polygonOutline(self,points):
        """
        Find a polygon on the image through the points given in points
        :param points: array of points which are the corners of the polygon to find
        :return:  the body part filled with true.
        """

        line = self.depthImage.shape[0]
        col = self.depthImage.shape[1]
        im_out = np.zeros([line,col],np.uint8)
        points = points.astype(np.float64)
        n = points.shape[0]
        i = 2
        d = 0

        # copy the point but with a circular permutation
        ptB = np.zeros(points.shape)
        ptB[-1]=points[0]
        for i in range(0,points.shape[0]-1):
            ptB[i] = points[i+1]
        # trace the segment
        M = np.zeros([line,col],np.uint8)
        for i in range(n-d):        
            A = points[i,:]
            B = ptB[i,:]
            slopes = self.findSlope(A,B)
            # if dist in x is longer than dist in y
            if np.abs(slopes[0]) > np.abs(slopes[1]):
                # if Ay have a higher value than By permute A and B
                if A[1] > B[1]:
                    tmp = B
                    B = A 
                    A = tmp
                # trace the slope between the two points
                for y in range(int(A[1]),int(B[1])+1):
                    x = np.round(-(slopes[1]*y+slopes[2])/slopes[0])
                    M[int(y),int(x)]= 1
            else :
                # if Ax have a higher value than Bx permute A and B
                if A[0] > B[0]:
                    tmp = B
                    B = A 
                    A = tmp
                # trace the slope between the two points
                for x in range(int(A[0]),int(B[0])+1):
                    y = np.round(-(slopes[0]*x+slopes[2])/slopes[1])
                    M[int(y),int(x)]= 1  
        ## Fill the polygon
        # Copy the thresholded image.
        im_floodfill = M.copy()
        im_floodfill = im_floodfill.astype(np.uint8)
         
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = M.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
         
        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0,0), 255)
         
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
         
        # Combine the two images to get the foreground.
        im_out = M | im_floodfill_inv 
        return im_out
   
    def nearestPeak(self,A,hipLeft,hipRight,knee_right):
        """
        In the case of upper legs, find the point in between the two upper legs that is at a edge of the hip
        :param A: binary image
        :param hipLeft: left hip junctions
        :param hipRight:  right hip junctions
        :param knee_right: right knee junctions
        :return: return a point at the edge and between the two legs
        Make drawing will help to understand
        """
        # check which hip is lower
        if (int(hipLeft[0])<int(hipRight[0])):
            # check which hip is lower
            # extract rectangle from the tree points depending on each other position
            if (int(hipLeft[1])<int(knee_right)):
                region = A[int(hipLeft[1]):int(knee_right),int(hipLeft[0]):int(hipRight[0])]
            else:
                region = A[int(knee_right):int(hipLeft[1]),int(hipLeft[0]):int(hipRight[0])]
        else:
            # check which hip is lower
            # extract rectangle from the tree points depending on each other position
            if (int(hipRight[1])<int(knee_right)):
                region = A[int(hipLeft[1]):int(knee_right),int(hipRight[0]):int(hipLeft[0])]
            else:
                region = A[int(knee_right):int(hipLeft[1]),int(hipRight[0]):int(hipLeft[0])]
        f = np.nonzero( (region==0) )
        # Get the highest point among the point that not in the body
        d = np.argmin(f[0])
        return np.array([f[1][d]-1+hipLeft[0],f[1][d]-1+hipLeft[1]])

    
    def armSeg(self,A,B,side):
        """
        Segment the left arm into two body parts
        :param A: depthImag
        :param B: depthImg after bilateral filtering
        :param side: if side = 0 the segmentation will be done for the right arm
                  otherwise it will be for the left arm
        :return: an array containing two body parts : an upper arm and a lower arm
        """
        
        # pos2D[4] = Shoulder_Left
        # pos2D[5] = Elbow_Left
        # pos2D[6] = Wrist_Left
        # pos2D[8] = Shoulder_Right
        # pos2D[9] = Elbow_Right
        # pos2D[10] = Wrist_Right

        # junction position (-1 adapted for python)
        pos2D = self.pos2D.astype(np.float64)-1
        # Right arm
        if side == 0 :
            shoulder =8
            elbow = 9
            wrist = 10
        # Left arm
        else :
            shoulder =4
            elbow = 5
            wrist = 6
        

        # First let us see the down limit thanks to the elbow and the wrist

        # FindSlopes give the slope of a line made by two points
        # Forearm
        slopesForearm=self.findSlope(pos2D[elbow],pos2D[wrist])
        a_pen67 = -slopesForearm[1]
        b_pen67 = slopesForearm[0]
        # Upperarm
        slopesUpperarm=self.findSlope(pos2D[elbow],pos2D[shoulder])


        a_pen = slopesForearm[0] + slopesUpperarm[0]
        b_pen = slopesForearm[1] + slopesUpperarm[1]
        if (a_pen == b_pen) and (a_pen==0):
            a_pen = slopesUpperarm[1]
            b_pen =-slopesUpperarm[0]

        # Perpendicular slopes
        c_pen = -(a_pen*pos2D[elbow,0]+b_pen*pos2D[elbow,1])
        

        # find lenght of arm
        bone1 = LA.norm(pos2D[elbow]-pos2D[wrist])
        bone2 = LA.norm(pos2D[elbow]-pos2D[shoulder])
        bone = max(bone1,bone2)
        
        # compute the intersection between the slope and the extremety of the body
        # And get two corners of the segmented body parts
        intersection_elbow=self.inferedPoint(A,a_pen,b_pen,c_pen,pos2D[elbow],0.5*bone)
        vect_elbow = intersection_elbow[0]-pos2D[elbow]
              
        # Slope forearm
        c_pen67=-(a_pen67*pos2D[wrist,0]+b_pen67*pos2D[wrist,1])
        # get intersection near the wrist
        intersection_wrist=self.inferedPoint(A,a_pen67,b_pen67,c_pen67,pos2D[wrist],bone/3)
        vect_wrist = intersection_wrist[0]-pos2D[wrist]
        vect67 = pos2D[wrist]-pos2D[elbow]
        vect67_pen = np.array([vect67[1], -vect67[0]])
        # reorder points if necessary
        if sum(vect67_pen*vect_elbow)*sum(vect67_pen*vect_wrist)<0:
           x = intersection_elbow[0]
           intersection_elbow[0] = intersection_elbow[1]
           intersection_elbow[1] = x

        # list of the 4 points defining the corners the forearm
        pt4D = np.array([intersection_elbow[0],intersection_elbow[1],intersection_wrist[1],intersection_wrist[0]])
        # list of the 4 points defining the corners the forearm permuted
        pt4D_bis = np.array([intersection_wrist[0],intersection_elbow[0],intersection_elbow[1],intersection_wrist[1]])
        if side != 0 :
            self.foreArmPtsR = pt4D
        else:
            self.foreArmPtsL = pt4D
        # Get slopes for each line of the polygon
        finalSlope=self.findSlope(pt4D.transpose(),pt4D_bis.transpose())
        x = np.isnan(finalSlope[0])
        #erase all NaN in the array
        polygonSlope = np.zeros([3,finalSlope[0][~np.isnan(finalSlope[0])].shape[0]])
        polygonSlope[0]=finalSlope[0][~np.isnan(finalSlope[0])]
        polygonSlope[1]=finalSlope[1][~np.isnan(finalSlope[1])]
        polygonSlope[2]=finalSlope[2][~np.isnan(finalSlope[2])]
        # get reference point
        midpoint = [(pos2D[elbow,0]+pos2D[wrist,0])/2, (pos2D[elbow,1]+pos2D[wrist,1])/2]
        ref= np.array([polygonSlope[0]*midpoint[0] + polygonSlope[1]*midpoint[1] + polygonSlope[2]]).astype(np.float32)
        #fill the polygon
        bw_up = ( A*self.polygon_optimize(polygonSlope,ref,x.shape[0]-sum(x)) > 0 )
        
        # pos2D[2] = Neck
        # pos2D[3] = Head
        
        #compute slopes Neck Head (SH)spine
        slopesSH=self.findSlope(pos2D[2],pos2D[3])
        a_pen = slopesSH[1]
        b_pen = - slopesSH[0]
        c_pen = -(a_pen*pos2D[2,0]+b_pen*pos2D[2,1])
        
        # compute the intersection between the slope and the extremety of the body
        intersection_head=self.inferedPoint(A,a_pen,b_pen,c_pen,pos2D[2])
        
        slopesTorso=self.findSlope(pos2D[20],pos2D[shoulder])
        
        a_pen = slopesTorso[0]+slopesUpperarm[0]
        b_pen = slopesTorso[1]+slopesUpperarm[1]
        if (a_pen == b_pen) and (a_pen==0):
            a_pen = slopesTorso[1]
            b_pen = -slopesTorso[0]

        #slope of the shoulder
        c_pen = -(a_pen*pos2D[shoulder,0]+b_pen*pos2D[shoulder,1])


        intersection_shoulder =self.inferedPoint(A,a_pen,b_pen,c_pen,pos2D[shoulder])
        vect65 = pos2D[shoulder]-pos2D[elbow]
        
        #
        vect_215 = intersection_shoulder[0]-pos2D[shoulder]
        #cross product to know which point to select
        t = np.cross(np.insert(vect_elbow, vect_elbow.shape[0],0),np.insert(vect65, vect65.shape[0],0))
        t1 = np.cross(np.insert(vect_215,vect_215.shape[0],0),np.insert(-vect65,vect65.shape[0],0))
        if t1[2]>0:
            intersection_shoulder[0] = intersection_shoulder[1]

        if t[2]<0:
            tmp = intersection_elbow[0]
            intersection_elbow[0] = intersection_elbow[1]
            intersection_elbow[1] = tmp

        # the upper arm need a fifth point -> Let us find it by finding the lowest x value
        # that meet the background in half of the body part
        B1 = np.logical_and( (A==0),self.polygonOutline(pos2D[[elbow, shoulder, 20, 0],:]))
        # transform the background into 1 and body into 0
        f = np.nonzero(B1)
        # find the minimum in x value (vertical line up = low value, down = high value) in f
        d = np.argmin(np.sum( np.square(np.array([pos2D[20,0]-f[1], pos2D[20,1]-f[0]]).transpose()),axis=1 ))
        peakArmpit = np.array([f[1][d],f[0][d]])
        # create the upperarm polygon out the five point defining it
        if side != 0 :
            ptA = np.stack((intersection_elbow[0],intersection_shoulder[0],intersection_head[0],peakArmpit,intersection_elbow[1]))
            self.upperArmPtsR = ptA
        else:
            ptA = np.stack((intersection_elbow[1],intersection_shoulder[1],intersection_head[1],peakArmpit,intersection_elbow[0]))
            self.upperArmPtsL = ptA
        bw_upper = (A*self.polygonOutline(ptA)>0)

        return np.array([bw_up,bw_upper])


    def legSeg(self,A,side):
        """
        Segment the leg into two body parts
        :param A: depthImag
        :param side: if side = 0 the segmentation will be done for the right leg
                  otherwise it will be for the left leg
        :return: an array containing two body parts : an upper leg and a lower leg
        """
        
        pos2D = self.pos2D.astype(np.float64)-1

        # Right
        if side == 0 :
            knee =17
            hip = 16
            ankle = 18
        else : # Left
            knee =13
            hip = 12
            ankle = 14
        
        #check which knee is higher
        if pos2D[17,1] > pos2D[13,1]:
            P = pos2D[17,1]
        else:
            P = pos2D[13,1]
        ## Find the Thigh
        # find the fifth point that can not be deduce simply with Slopes or intersection using the entire hip
        peak1 = self.nearestPeak(A,pos2D[12],pos2D[16],P)

        # compute slopes related to the leg position
        slopeThigh = self.findSlope(pos2D[hip],pos2D[knee])
        slopeCalf = self.findSlope(pos2D[ankle],pos2D[knee])
        a_pen = slopeThigh[0] + slopeCalf[0]
        b_pen = slopeThigh[1] + slopeCalf[1]
        if (a_pen == b_pen) and (a_pen==0):
            a_pen = slopeThigh[1]
            b_pen =-slopeThigh[0]
        c_pen = -(a_pen*pos2D[knee,0]+b_pen*pos2D[knee,1])

        # find 2 points corner of the knee
        intersection_knee=self.inferedPoint(A,a_pen,b_pen,c_pen,pos2D[knee])

        # find right side of the hip rsh
        c_rsh = -(slopeThigh[1]*pos2D[hip,0]-slopeThigh[0]*pos2D[hip,1])
        intersection_rsh=self.inferedPoint(A,slopeThigh[1],-slopeThigh[0],c_rsh,pos2D[hip])

        if side == 0:
            v1 = pos2D[knee] - pos2D[hip]
            v2 = pos2D[0] - pos2D[hip]
            # Put the point in the good order in function of the angle alpha
            alpha = np.arccos(np.sum(v1*v2)/(np.sqrt(sum(v1*v1))*np.sqrt(sum(v2*v2)) ))/math.pi*180
            if abs(alpha-90)>45:
                B = np.logical_and( (A==0),self.polygonOutline(pos2D[[20, 0, knee],:]))
                f = np.nonzero(B)
                d = np.argmin(np.sum( np.square(np.array([pos2D[hip,0]-f[1], pos2D[hip,1]-f[0]]).transpose()),axis=1 ))
                intersection_rsh[1] = np.array([f[1][d],f[0][d]])
            ptA = np.stack((pos2D[0],intersection_rsh[1],intersection_knee[1],intersection_knee[0],peak1))
            self.thighPtsR = ptA
        else :
            ptA = np.stack((pos2D[0],intersection_rsh[0],intersection_knee[0],intersection_knee[1],peak1))  
            self.thighPtsL = ptA
        # Fill up the polygon
        bw_up = ( (A*self.polygonOutline(ptA))>0 )
        
        ## Find Calf
        # Define slopes
        a_pen = slopeCalf[1]
        b_pen = -slopeCalf[0]
        c_pen = -(a_pen*pos2D[ankle,0]+b_pen*pos2D[ankle,1])

        # find 2 points corner of the ankle
        intersection_ankle=self.inferedPoint(A,a_pen,b_pen,c_pen,pos2D[ankle])  
        ptA = np.stack((intersection_ankle[1],intersection_ankle[0],intersection_knee[0],intersection_knee[1]))  
        if side == 0 :
            self.calfPtsR = ptA
        else:
            self.calfPtsL = ptA
        # Fill up the polygon
        bw_down = (A*self.polygonOutline(ptA)>0)
        return np.array([bw_up,bw_down])
    
    def headSeg(self,A):
        """
        Segment the head
        :param A: binary depthImag
        :return: head body part
        """

        pos2D = self.pos2D.astype(np.float64)-1    
        
        #compute slopes Shoulder Head (SH)spine
        slopesSH=self.findSlope(pos2D[2],pos2D[3])
        a_pen = slopesSH[1]
        b_pen = - slopesSH[0]
        c_pen = -(a_pen*pos2D[2,0]+b_pen*pos2D[2,1])

        # find left
        x = int(pos2D[2,0])
        while 1:
            x = x-1
            # follow the slopes
            y =int(np.round(-(a_pen*x+c_pen)/b_pen))
            # reach an edges
            if A[y,x]==0:
                headLeft = np.array([x,y])
                break
            
        # find right
        x = int(pos2D[2,0])
        while 1:
            x = x+1
            # follow the slopes
            y =int(np.round(-(a_pen*x+c_pen)/b_pen))
            # reach an edges
            if A[y,x]==0:
                headRight = np.array([x,y])
                break

        # distance head - neck
        h = 2*(pos2D[2,1]-pos2D[3,1])
        # create point that higher than the head
        headUp_right = np.array([pos2D[8,0],pos2D[2,1]-h])
        headUp_left = np.array([pos2D[5,0],pos2D[2,1]-h])
        # stock corner of the polyogne
        pt4D = np.array([headUp_right,headUp_left,headLeft,headRight])
        self.headPts = pt4D
        pt4D_bis = np.array([headUp_left,headLeft,headRight,headUp_right])
        # Compute slope of each line of the polygon
        HeadSlope=self.findSlope(pt4D.transpose(),pt4D_bis.transpose())
        # reference point
        midpoint = [pos2D[3,0], pos2D[3,1]]
        ref= np.array([HeadSlope[0]*midpoint[0] + HeadSlope[1]*midpoint[1] + HeadSlope[2]]).astype(np.float32)
        # fill up the polygon
        bw_head = ( A*self.polygon_optimize(HeadSlope,ref,HeadSlope.shape[0]) > 0 )  
        return bw_head

    def GetBody(self,binaryImage):
        """
        Delete all the unwanted connected component from the binary image
        It focuses on the group having the right pos2D, for now the body
        :param binaryImage: binary image of the body but all body part are substracted to the body leaving only the trunck and noise
        :return: trunk
        """

        pos2D = self.pos2D
        # find all connected component and label it
        labeled, n = spm.label(binaryImage)
        # Get the labelled  of the connected component that have the trunk
        threshold = labeled[pos2D[1,1],pos2D[1,0]]
        # erase all connected component that are not the trunk
        labeled = (labeled==threshold)
        return labeled
    
    def GetHand(self,binaryImage,side):
        """
        Delete all the little group unwanted from the binary image
        It focuses on the group having the right pos2D, here the hands
        :param binaryImage: binary image of the body without limbs
        :param side: if side = 0 the segmentation will be done for the right hand
                  otherwise it will be for the left hand
        :return: one hand
        """
        # Right side
        if side == 0 :
            idx =11
        # Left side
        else :
            idx =7
        pos2D = self.pos2D

        #create a sphere of radius 12 so that anything superior does not come in the feet label
        handDist = 12# LA.norm( (pos2D[16]-pos2D[12])/1.5).astype(np.int16)
        #since feet are on the same detph as the floor some processing are required before using cc
        line = self.depthImage.shape[0]
        col = self.depthImage.shape[1]
        mask = np.ones([line,col,2])
        mask = mask*pos2D[idx]
        #create a matrix containing in each pixel its index
        lineIdx = np.array([np.arange(line) for _ in range(col)]).reshape(col,line).transpose()
        colIdx = np.array([np.arange(col) for _ in range(line)]).reshape(line,col)
        ind = np.stack( (colIdx,lineIdx), axis = 2)
        #compute the distance between the skeleton point of feet and each pixel
        mask = np.sqrt(np.sum( (ind-mask)*(ind-mask),axis = 2))
        mask = (mask < handDist)
        mask = mask * binaryImage

        # compute the body part as it is done for the head
        labeled, n = spm.label(mask)
        threshold = labeled[pos2D[idx,1],pos2D[idx,0]]
        labeled = (labeled==threshold)
        return labeled
    
    
    def GetFoot(self,binaryImage,side):
        """
        Delete all the little group unwanted from the binary image
        It focuses on the group having the right pos2D, here the feet
        :param binaryImage: binary image of the body without limbs
        :param side: if side = 0 the segmentation will be done for the right feet
                  otherwise it will be for the left feet
        :return: one feet
        """


        #Right Side
        if side == 0 :
            idx =19
        # Left Side
        else :
            idx =15
        pos2D = self.pos2D

        #create a sphere mask of radius 12 so that anything superior does not come in the feet label
        footDist = 12# LA.norm( (pos2D[16]-pos2D[12])/1.5).astype(np.int16)
        #since feet are on the same detph as the floor some processing are required before using cc
        line = self.depthImage.shape[0]
        col = self.depthImage.shape[1]
        mask = np.ones([line,col,2])
        mask = mask*pos2D[idx]
        #create a matrix containing in each pixel its index
        lineIdx = np.array([np.arange(line) for _ in range(col)]).reshape(col,line).transpose()
        colIdx = np.array([np.arange(col) for _ in range(line)]).reshape(line,col)
        ind = np.stack( (colIdx,lineIdx), axis = 2)
        #compute the distance between the skeleton point of feet and each pixel
        mask = np.sqrt(np.sum( (ind-mask)*(ind-mask),axis = 2))
        mask = (mask < footDist)
        mask = mask * binaryImage

        # compute the body part as it is done for the head
        labeled, n = spm.label(mask)
        threshold = labeled[pos2D[idx,1],pos2D[idx,0]]
        labeled = (labeled==threshold)
        return labeled
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    