# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 17:50:19 2017

@author: Inoe ANDRE
"""

import numpy as np
import math
from math import cos, sin
from numpy import linalg as LA
PI = math.pi

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


class Stitch():
    
    def __init__(self, number_bodyPart):
        self.nb_bp = number_bodyPart
        self.StitchedVertices = 0
        self.StitchedFaces = 0

    def InvPose(self, Pose):
        '''Compute the inverse transform of Pose'''
        PoseInv = np.zeros(Pose.shape, Pose.dtype)
        PoseInv[0:3, 0:3] = LA.inv(Pose[0:3, 0:3])
        PoseInv[0:3, 3] = -np.dot(PoseInv[0:3, 0:3], Pose[0:3, 3])
        PoseInv[3, 3] = 1.0
        return PoseInv
        
    def NaiveStitch(self, PartVtx,PartNmls,PartFaces,PoseBP):
        '''
        This function will just add the vertices and faces of each body parts 
        together after transforming them in the global coordinates system
        '''
        #Initialize values from the list of
        ConcatVtx = self.StitchedVertices
        ConcatFaces = self.StitchedFaces
        ConcatNmls = self.StitchedNormales
        # tranform the vertices in the global coordinates system
        PartVertices = self.TransformVtx(PartVtx,PoseBP,1)
        PartNormales = self.TransformNmls(PartNmls,PoseBP,1)
        PartFacets = PartFaces + np.max(ConcatFaces)+1
        self.StitchedVertices = np.concatenate((ConcatVtx,PartVertices))
        self.StitchedNormales = np.concatenate((ConcatNmls,PartNormales))
        self.StitchedFaces = np.concatenate((ConcatFaces,PartFacets))
        

        
    def TransformVtx(self, Vtx,Pose, s):
        """
        Transform the vertices in a system to another system.
        Here it will be mostly used to transform from local system to global coordiantes system
        """
        stack_pt = np.ones(np.size(Vtx,0), dtype = np.float32)
        pt = np.stack( (Vtx[ ::s,0],Vtx[ ::s,1],Vtx[ ::s,2],stack_pt),axis =1 )
        Vtx = np.dot(pt,Pose.T)
        return Vtx[:,0:3]
        
    def TransformNmls(self, Nmls,Pose, s):
        """
        Transform the normales in a system to another system.
        Here it will be mostly used to transform from local system to global coordiantes system
        """
        nmle = np.zeros((Nmls.shape[0], Nmls.shape[1]), dtype = np.float32)
        nmle[ ::s,:] = np.dot(Nmls[ ::s,:],Pose[0:3,0:3].T)
        return nmle[:,0:3]
        
    def RArmsTransform(self, angle,bp, pos2d,RGBD,Tg):
        """
        Transform Pose matrix to move the model of the right arm
        For now just a rotation in the z axis
        bp : numero of the body parts
        pos2d : position in 2D of the junctions
        RGBD : an RGBD object containing the image
        Tg : local to global transform
        """

        # Rotate skeleton right arm
        angley = angle  # pi * 2. * delta_x / float(Size[0])
        RotZ = np.array([[cos(angley), -sin(angley), 0.0, 0.0], \
                         [sin(angley), cos(angley), 0.0, 0.0], \
                         [0.0, 0.0, 1.0, 0.0], \
                         [0.0, 0.0, 0.0, 1.0]], np.float32)

        ctr = pos2d[4].astype(np.int)
        rotAxe = Tg[7][0:3, 3]
        ctr3D = Tg[bp][0:3, 3]
        # z = rotAxe[2]
        # ctr3D[0] = z * (ctr[0] - RGBD.intrinsic[0, 2]) / RGBD.intrinsic[0, 0]
        # ctr3D[1] = z * (ctr[1] - RGBD.intrinsic[1, 2]) / RGBD.intrinsic[1, 1]
        # ctr3D[1] = z
        # # Rotation of about 30
        if bp == 1 :
            # # transform joints
            print pos2d[5:8]
            rotat = RotZ[0:2, 0:2]
            for rot in range(3):
                pt = (pos2d[5 + rot]).astype(np.int)- ctr
                pt = np.dot(rotat[0:2, 0:2], pt.T).T
                pos2d[5+rot] = pt + ctr



        if bp ==1 or bp == 2 or bp==12:
            if bp == 12:
                #pos = 4 # should left
                pos = 7  # hand left
                Xm = pos2d[pos,0]#(pos2d[pos,0] + pos2d[pos-2,0])/2
                Ym = pos2d[pos,1]#(pos2d[pos,1] + pos2d[pos-2,1])/2
            elif bp == 2:
                pos = 5 #elbow left
                Xm = (pos2d[pos,0] + pos2d[pos-1,0])/2
                Ym = (pos2d[pos,1] + pos2d[pos-1,1])/2
            elif bp == 1:
                pos = 6  # wrist left
                Xm = (pos2d[pos,0] + pos2d[pos-1,0])/2
                Ym = (pos2d[pos,1] + pos2d[pos-1,1])/2

            ctr = Tg[bp][0:3,3]

            # for rot in range(3):
            #     pt = (pos2d[5 + rot]).astype(np.int)- ctr
            #     pt = np.dot(rotat[0:2, 0:2], pt.T).T
            #     pos2d[5+rot] = pt + ctr

            print ctr
            print ctr
            print RGBD.Vtx[pos2d[pos,0],pos2d[pos,1]]
            print RGBD.Vtx[pos2d[pos,0],pos2d[pos,1]]
            z = ctr[2]

            ctr[0] = z * (Xm - RGBD.intrinsic[0, 2]) / RGBD.intrinsic[0, 0]
            ctr[1] = z * (Ym - RGBD.intrinsic[1, 2]) / RGBD.intrinsic[1, 1]
            print ctr
            print ctr
            RotZ[0:3, 3] = ctr
            Tg[bp][:,0:3] = np.dot(RotZ, Tg[bp][:,0:3])
            #Tg[bp][0:3, 3] = ctr
            print RotZ




    def GetBBTransfo(self, pos2d,cur,prev,RGBD,bp):
        """
        Transform Pose matrix to move the model body parts according to the position of the skeleton
        For now just a rotation in the z axis
        bp : numero of the body parts
        pos2d : position in 2D of the junctions
        cur : index for the current frame
        prev : index for the previous frame
        RGBD : an RGBD object containing the image
        """
        PosCur = pos2d[0,cur]
        PosPrev = pos2d[0,prev]

        print prev
        print cur
        # get the junctions of the current body parts
        pos = self.GetPos(bp)

        # Compute the transform between the two skeleton : Tbb = A^-1 * B



        # compute the 3D centers point of the bounding boxes using the skeleton for B
        ctr = np.array([0.0, 0.0, 0.0], np.float)
        z = RGBD.TransfoBB[bp][3,3]
        if bp < 9:
            Xm = (PosCur[pos[0], 0] + PosCur[pos[1], 0]) / 2
            Ym = (PosCur[pos[0], 1] + PosCur[pos[1], 1]) / 2
        else:
            Xm = PosCur[pos[2], 0]
            Ym = PosCur[pos[2], 1]

        ctr[0] = z * (Xm - RGBD.intrinsic[0, 2]) / RGBD.intrinsic[0, 0]
        ctr[1] = z * (Ym - RGBD.intrinsic[1, 2]) / RGBD.intrinsic[1, 1]
        ctr[2] = z

        # Compute B Transfo from camera to local system of current frame
        pt = np.array([0.0, 0.0, 0.0], np.float)
        pt[0] = z * (PosCur[pos[1], 0] - RGBD.intrinsic[0, 2]) / RGBD.intrinsic[0, 0]
        pt[1] = z * (PosCur[pos[1], 1] - RGBD.intrinsic[1, 2]) / RGBD.intrinsic[1, 1]
        pt[2] = z
        axeX = (pt - ctr)/LA.norm(pt - ctr)
        signX = np.sign(axeX)
        axeX = signX[1]*axeX
        axeZ = np.array([0.0, 0.0, 1.0], np.float)
        axeY = normalized_cross_prod(axeZ, axeX)

        # Bounding boxes tracking matrix for B
        e1b = np.array( [axeX[0],axeX[1],axeX[2],0])
        e2b = np.array( [axeY[0],axeY[1],axeY[2],0])
        e3b = np.array( [axeZ[0],axeZ[1],axeZ[2],0])
        origine = np.array( [ctr[0],ctr[1],ctr[2],1])
        B = np.stack( (e1b,e2b,e3b,origine),axis = 0 ).T

        # # Compute A Transfo from camera to local system of previous frame
        # # compute the 3D centers point of the bounding boxes using the skeleton for A
        # ctr = np.array([0.0, 0.0, 0.0], np.float)
        # z = RGBD.TransfoBB[bp][3,3]
        # if bp < 9:
        #     Xm = (PosPrev[pos[0], 0] + PosPrev[pos[1], 0]) / 2
        #     Ym = (PosPrev[pos[0], 1] + PosPrev[pos[1], 1]) / 2
        # else:
        #     Xm = PosPrev[pos[2], 0]
        #     Ym = PosPrev[pos[2], 1]
        #
        # ctr[0] = z * (Xm - RGBD.intrinsic[0, 2]) / RGBD.intrinsic[0, 0]
        # ctr[1] = z * (Ym - RGBD.intrinsic[1, 2]) / RGBD.intrinsic[1, 1]
        # ctr[2] = z
        #
        # # Compute A Transfo from camera to local system of current frame
        # pt = np.array([0.0, 0.0, 0.0], np.float)
        # pt[0] = z * (PosPrev[pos[1], 0] - RGBD.intrinsic[0, 2]) / RGBD.intrinsic[0, 0]
        # pt[1] = z * (PosPrev[pos[1], 1] - RGBD.intrinsic[1, 2]) / RGBD.intrinsic[1, 1]
        # pt[2] = z
        # axeX = (pt - ctr)/LA.norm(pt - ctr)
        # axeZ = np.array([0.0, 0.0, 1.0], np.float)
        # axeY = normalized_cross_prod(axeZ, axeX)
        #
        # # Bounding boxes tracking matrix for B
        # e1b = np.array( [axeX[0],axeX[1],axeX[2],0])
        # e2b = np.array( [axeY[0],axeY[1],axeY[2],0])
        # e3b = np.array( [axeZ[0],axeZ[1],axeZ[2],0])
        # origine = np.array( [ctr[0],ctr[1],ctr[2],1])
        # A = np.stack( (e1b,e2b,e3b,origine),axis = 0 ).T


        # Compute Tbb : skeleton tracking transfo
        Tbb =B# np.dot(self.InvPose(A),B)

        #print A
        print B
        print Tbb

        return Tbb

    def GetPos(self,bp):
        '''
        :param bp:
        :return: return the junctions corresponding to the body parts
        '''
        mid = 0
        if bp ==1 :
            pos1 = 6  # wrist left
            pos2 = 5 # elbow left
        elif bp == 2:
            pos1 = 4  # elbow left
            pos2 =  5#  shoulder left
        elif bp == 3:
            pos1 = 10 # wrist right
            pos2 =  9 # elbow left
        elif bp == 4:
            pos1 = 9 # elbow left
            pos2 =  8 # shoulder left
        elif bp == 5:
            pos1 = 13 # knee left
            pos2 = 12 # hip left
        elif bp == 6:
            pos1 = 14 # ankle left
            pos2 =  13 # knee left
        elif bp == 7:
            pos1 = 17  #knee right
            pos2 =  16 #hip right
        elif bp == 8:
            pos1 = 18  # ankle right
            pos2 =  17 # knee right
        elif bp == 9:
            pos1 = 3 # head
            pos2 = 2 # neck
            mid = 3
        elif bp == 10:
            pos1 = 0  # spine base
            pos2 = 20 # spine should
            mid = 1 # spine mid
        elif bp == 11:
            pos1 = 11  # hand right
            pos2 = 10 # wrist right
            mid = 11
        elif bp == 12:
            pos1 = 7  # hand left
            pos2 = 6  # wrist left
            mid == 7
        elif bp == 13:
            pos1 = 19 # foot right
            pos2 = 18  # ankle right
            mid = 19
        elif bp == 14:
            pos1 = 15  # foot left
            pos2 = 14 # ankle left
            mid = 15

        return np.array([pos1,pos2,mid])