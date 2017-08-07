"""
7 August 2017
@author: Inoe ANDRE
All process within a body part
"""


import numpy as np
import math
from numpy import linalg as LA
import imp
import time

PI = math.pi
RGBD = imp.load_source('RGBD', './lib/RGBD.py')
RGBDimg = imp.load_source('RGBDimg', './lib/RGBDimg.py')
TrackManager = imp.load_source('TrackManager', './lib/tracking.py')
TSDFtk = imp.load_source('TSDFtk', './lib/TSDF.py')
GPU = imp.load_source('GPUManager', './lib/GPUManager.py')
My_MC = imp.load_source('My_MarchingCube', './lib/My_MarchingCube.py')
Stitcher = imp.load_source('Stitcher', './lib/Stitching.py')
General = imp.load_source('General', './lib/General.py')


class BodyParts():
    """
    Body parts
    """
    def __init__(self, GPUManager,RGBD,RGBD_BP, Tlg):
        """
        Init a body parts
        :param GPUManager: GPU environment
        :param RGBD: Image having all the body
        :param RGBD_BP: Image containing just the body part
        :param Tlg: Transform local to global for the concerned body parts
        """
        self.GPUManager = GPUManager
        self.RGBD = RGBD
        self.Tlg =Tlg
        self.RGBD_BP = RGBD_BP
        self.VoxSize = 0.005


    def Model3D_init(self,bp):
        """
        Create a 3D model of the body parts
        :param bp: numero of the body part
        :return:  none
        """

        # need to put copy transform amtrix in PoseBP for GPU
        PoseBP = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)

        # Compute the dimension of the body part to create the volume
        Xraw = int(round(LA.norm(self.RGBD.coordsGbl[bp][3] - self.RGBD.coordsGbl[bp][0]) / self.VoxSize)) + 1
        Yraw = int(round(LA.norm(self.RGBD.coordsGbl[bp][1] - self.RGBD.coordsGbl[bp][0]) / self.VoxSize)) + 1
        Zraw = int(round(LA.norm(self.RGBD.coordsGbl[bp][4] - self.RGBD.coordsGbl[bp][0]) / self.VoxSize)) + 1

        # Dimensions of body part volume
        X = max(Xraw, Zraw)
        Y = Yraw
        Z = max(Xraw, Zraw)
        # show result
        print "bp = %d, X= %d; Y= %d; Z= %d" % (bp, X, Y, Z)

        # Put the Global transfo in PoseBP so that the dtype entered in the GPU is correct
        for i in range(4):
            for j in range(4):
                PoseBP[i][j] = self.Tlg[i][j]

        # TSDF Fusion of the body part
        self.TSDFManager = TSDFtk.TSDFManager((X, Y, Z), self.RGBD_BP, self.GPUManager)
        self.TSDFManager.FuseRGBD_GPU(self.RGBD_BP, PoseBP)

        # Create Mesh
        self.MC = My_MC.My_MarchingCube(self.TSDFManager.Size, self.TSDFManager.res, 0.0, self.GPUManager)
        # Mesh rendering
        self.MC.runGPU(self.TSDFManager.TSDFGPU)
        start_time3 = time.time()

        # save with the number of the body part
        bpStr = str(bp)
        self.MC.SaveToPly("body" + bpStr + ".ply")
        elapsed_time = time.time() - start_time3
        print "SaveBPToPly: %f" % (elapsed_time)





