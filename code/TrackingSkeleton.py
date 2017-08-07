# File created by Diego Thomas the 21-11-2016
# Second Author Inoe AMDRE

# File to handle program main loop
import sys
import cv2
from math import cos, sin
import numpy as np
from numpy import linalg as LA
from numpy.matlib import rand, zeros, ones, empty, eye
import Tkinter as tk
import tkMessageBox
from tkFileDialog import askdirectory
from PIL import Image, ImageTk
import imp
import scipy.io
import time
import pyopencl as cl

libPath = "/Users/nii-user/inoe/NIIComputerVision/code"
RGBD_ = imp.load_source('RGBD', libPath+'/lib/RGBD.py')
RGBDimg = imp.load_source('RGBDimg',libPath+ '/lib/RGBDimg.py')
TrackManager = imp.load_source('TrackManager', libPath+'/lib/tracking.py')
TSDFtk = imp.load_source('TSDFtk', libPath+'/lib/TSDF.py')
GPU = imp.load_source('GPUManager',libPath +'/lib/GPUManager.py')
My_MC = imp.load_source('My_MarchingCube',libPath+ '/lib/My_MarchingCube.py')
Stitcher = imp.load_source('Stitcher',libPath+ '/lib/Stitching.py')


def in_mat_zero2one(mat):
    """This fonction replace in the matrix all the 0 to 1"""
    mat_tmp = (mat != 0.0)
    res = mat * mat_tmp + ~mat_tmp
    return res



def DrawPoint2D(point, radius, color):
    if point[0] > 0 and point[1] > 0:
        x1, y1 = (point[0] - radius), (point[1] - radius)
        x2, y2 = (point[0] + radius), (point[1] + radius)
    else:
        x1, y1 = (point[0]), (point[1])
        x2, y2 = (point[0]), (point[1])
    canvas.create_oval(x1, y1, x2, y2, fill=color)

def DrawColors2D( RGBD, img, Pose):
    '''this function draw the color of each segmented part of the body'''
    newImg = img.copy()
    Txy = RGBD.transCrop
    label = RGBD.labels
    for k in range(1, RGBD.bdyPart.shape[0] + 1):
        color = RGBD.bdyColor[k - 1]
        for i in range(Txy[1], Txy[3]):
            for j in range(Txy[0], Txy[2]):
                if label[i][j] == k:
                    newImg[i, j] = color
                else:
                    newImg[i, j] = newImg[i, j]
    return newImg

def DrawSkeleton2D( Pose):
    '''this function draw the Skeleton of a human and make connections between each part'''
    pos = pos2d[0][Index]
    for i in range(np.size(connection, 0)):
        pt1 = (pos[connection[i, 0] - 1, 0], pos[connection[i, 0] - 1, 1])
        pt2 = (pos[connection[i, 1] - 1, 0], pos[connection[i, 1] - 1, 1])
        radius = 1
        color = "blue"
        DrawPoint2D(pt1, radius, color)
        DrawPoint2D(pt2, radius, color)
        canvas.create_line(pt1[0], pt1[1], pt2[0], pt2[1], fill="red")

def DrawSkeleton2Dbis( Pose,Index):
    '''this function draw the Skeleton of a human and make connections between each part'''
    pos = pos2d[0][Index]
    for i in range(np.size(connection, 0)):
        pt1 = (pos[connection[i, 0] - 1, 0], pos[connection[i, 0] - 1, 1])
        pt2 = (pos[connection[i, 1] - 1, 0], pos[connection[i, 1] - 1, 1])
        radius = 1
        color = "blue"
        DrawPoint2D(pt1, radius, color)
        DrawPoint2D(pt2, radius, color)
        canvas.create_line(pt1[0], pt1[1], pt2[0], pt2[1], fill="red")

def DrawCenters2D( Pose, s=1):
    '''this function draw the center of each oriented coordinates system for each body part'''
    ctr2D = RGBD.GetProjPts2D_optimize(RGBD.ctr3D, Pose)
    for i in range(1, len(RGBD.ctr3D)):
        c = ctr2D[i]
        DrawPoint2D(c, 2, "yellow")


def DrawOBBox2D( Pose):
    '''Draw in the canvas the oriented bounding boxes for each body part'''
    OBBcoords2D = []
    OBBcoords2D.append([0., 0., 0.])
    # for each body part
    for i in range(1, len(RGBD[0].coordsGbl)):
        OBBcoords2D.append(RGBD[0].GetProjPts2D_optimize(RGBD[0].coordsGbl[i], Pose))
        pt = OBBcoords2D[i]
        # print 'OBBcoords2D[]'
        # print pt.shape
        # create lines of the boxes
        for j in range(3):
            canvas.create_line(pt[j][0], pt[j][1], pt[j + 1][0], pt[j + 1][1], fill="red", width=2)
            canvas.create_line(pt[j + 4][0], pt[j + 4][1], pt[j + 5][0], pt[j + 5][1], fill="red", width=2)
            canvas.create_line(pt[j][0], pt[j][1], pt[j + 4][0], pt[j + 4][1], fill="red", width=2)
        canvas.create_line(pt[3][0], pt[3][1], pt[0][0], pt[0][1], fill="red", width=2)
        canvas.create_line(pt[7][0], pt[7][1], pt[4][0], pt[4][1], fill="red", width=2)
        canvas.create_line(pt[3][0], pt[3][1], pt[7][0], pt[7][1], fill="red", width=2)
        # draw points of the bounding boxes
        for j in range(8):
            DrawPoint2D(pt[j], 2, "blue")


def InvPose( Pose):
    '''Compute the inverse transform of Pose'''
    PoseInv = np.zeros(Pose.shape, Pose.dtype)
    PoseInv[0:3, 0:3] = LA.inv(Pose[0:3, 0:3])
    PoseInv[0:3, 3] = -np.dot(PoseInv[0:3, 0:3], Pose[0:3, 3])
    PoseInv[3, 3] = 1.0
    return PoseInv





GPUManager = GPU.GPUManager()
GPUManager.print_device_info()
GPUManager.load_kernels()
GPUManager = GPUManager
draw_bump = False
draw_spline = False


#top = tk.Tk()
pathdata= "/Users/nii-user/inoe/data/"
path = "/Users/nii-user/inoe/NIIComputerVision/data/"
color_tag = 1
calib_file = open(path + '/Calib.txt', 'r')
calib_data = calib_file.readlines()
Size = [int(calib_data[0]), int(calib_data[1])]
intrinsic = np.array([[float(calib_data[2]), float(calib_data[3]), float(calib_data[4])], \
                           [float(calib_data[5]), float(calib_data[6]), float(calib_data[7])], \
                           [float(calib_data[8]), float(calib_data[9]), float(calib_data[10])]],
                          dtype=np.float32)

print intrinsic

fact = 1000.0

name = 'NewData.mat'
name2 = 'String4b.mat'
mat = scipy.io.loadmat(pathdata + name)
mat2 = scipy.io.loadmat(path + name2)
lImages = mat['DepthImg']
pos2d = mat['Pos2D']
colors = mat['ColorImg']
bdyIdx = mat2['BodyIndex']
for i in range(100):
    bdyIdx[0,i] = 255*(np.ones(lImages[0,0].shape,np.uint8))

print bdyIdx.shape



connectionMat = scipy.io.loadmat(path + '/SkeletonConnectionMap.mat')
connection = connectionMat['SkeletonConnectionMap']
Pose = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype=np.float32)
T_Pose = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype=np.float32)
PoseBP = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype=np.float32)
Id4 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype=np.float32)

Index = 80
nunImg = 100

while (not lImages[0][Index].any() or \
       not pos2d[0][Index].any()  \
       # segmentation default
       or Index == 72 or Index == 87 or Index==17 or Index==31 or Index==0 or Index==1 or Index==2 or Index==3 or Index==4 or Index==5 \
       # coordinate system default
       or Index == 36 or Index == 37 or Index == 39 or Index == 46 or Index == 47 or Index == 49):
    print("List lImages is empty")
    Index += 1
print Index
# Former Depth Image (i.e: i)
RGBD = []
for bp in range(15):
    RGBD.append(RGBD_.RGBD(path + '/Depth.tiff', path + '/RGB.tiff', intrinsic, fact))
    RGBD[bp].LoadMat(lImages, pos2d, connection, bdyIdx)
    RGBD[bp].ReadFromMat(Index)
    RGBD[bp].BilateralFilter(-1, 0.02, 3)
    # segmenting the body
    RGBD[bp].Crop2Body()
    RGBD[bp].BodySegmentation()
    RGBD[bp].BodyLabelling()
    # select the body part
    if bp == 0:
        RGBD[bp].depth_image *= (RGBD[bp].labels > 0)
    # elif bp == 9:
    #     RGBD[bp].depth_image *= (RGBD[bp].labels == bp) + (RGBD[bp].labels == bp+1)
    # elif bp == 10:
    #     RGBD[bp].depth_image *= (RGBD[bp].labels == bp -1 ) + (RGBD[bp].labels == bp)
    else:
        RGBD[bp].depth_image *= (RGBD[bp].labels == bp)
    RGBD[bp].Vmap_optimize()
    RGBD[bp].NMap_optimize()
    # create the transform matrix from local to global coordinate

top = tk.Tk()
# projection in 2d space to draw it
rendering = np.zeros((Size[0], Size[1], 3), dtype=np.uint8)
# projection of the current image/ Overlay
rendering = RGBD[0].Draw_optimize(rendering,Id4, 1, color_tag)
# 3D reconstruction of the whole image
canvas = tk.Canvas(top, bg="black", height=Size[0], width=Size[1])
canvas.pack()
#rendering = DrawColors2D(RGBD[0],rendering,Pose)
img = Image.fromarray(rendering, 'RGB')
imgTk = ImageTk.PhotoImage(img)
canvas.create_image(0, 0, anchor=tk.NW, image=imgTk)
DrawSkeleton2D(Pose)
# DrawCenters2D(Pose)
# DrawOBBox2D(Pose)
canvas.pack()
top.mainloop()


# for i in range(RGBD[0].bdyPart.shape[0]):
#     istr = str(i)
#     cv2.imshow('bdyPart'+istr, RGBD[0].bdyPart[i].astype(np.float))
#     cv2.waitKey(0)
RGBD[0].myPCA()

'''
The first image is process differently from the other since it does not have any previous value.
'''
# Number of body part +1 since the counting starts from 1
up = 0
bpstart = 1 + up
nbBdyPart = RGBD[0].bdyPart.shape[0] + 1#2 + up#

# Init for Local to global Transform
Tg = []
Tg.append(Id4)
TgPrev = []
TgPrev.append(Id4)
for bp in range(bpstart, nbBdyPart):
    # Get the tranform matrix from the local coordinates system to the global system
    Tglo = RGBD[0].TransfoBB[bp]
    Tg.append(Tglo.astype(np.float32))
    TgPrev.append(Tglo.astype(np.float32))

# For TSDF output
X = []
X.append(0)
Y = []
Y.append(0)
Z = []
Z.append(0)
VoxSize = 0.005

# List of TSDF
TSDFManager = []
TSDFManager.append(TSDFtk.TSDFManager((10, 10, 10), RGBD[0], GPUManager))

# For Marching cubes output
MC = []
MC.append(My_MC.My_MarchingCube(TSDFManager[0].Size, TSDFManager[0].res, 0.0, GPUManager))
# Sum of the number of vertices and faces of all body parts
nb_verticesGlo = 0
nb_facesGlo = 0

# Initiate stitcher object
StitchBdy = Stitcher.Stitch(nbBdyPart)
# Creating mesh of each body part
for bp in range(bpstart, nbBdyPart):
    # MC = 0
    # TSDFManager = 0
    bou = bp - bpstart + 1
    # Compute the dimension of the body part to create the volume
    Xraw = int(round(LA.norm(RGBD[0].coordsGbl[bp][3] - RGBD[0].coordsGbl[bp][0]) / VoxSize)) + 1
    Yraw = int(round(LA.norm(RGBD[0].coordsGbl[bp][1] - RGBD[0].coordsGbl[bp][0]) / VoxSize)) + 1
    Zraw = int(round(LA.norm(RGBD[0].coordsGbl[bp][4] - RGBD[0].coordsGbl[bp][0]) / VoxSize)) + 1

    X.append(max(Xraw, Zraw))
    Y.append(Yraw)
    Z.append(max(Xraw, Zraw))
    # show result
    print "bp = %d, X= %d; Y= %d; Z= %d" % (bou, X[bou], Y[bou], Z[bou])

    # Put the Global transfo in PoseBP so that the dtype entered in the GPU is correct
    for i in range(4):
        for j in range(4):
            PoseBP[i][j] = Tg[bp][i][j]

    # TSDF Fusion of the body part
    TSDFManager.append(TSDFtk.TSDFManager((X[bou], Y[bou], Z[bou]), RGBD[bp], GPUManager))
    TSDFManager[bou].FuseRGBD_GPU(RGBD[bp], PoseBP)

    # Create Mesh
    MC.append(My_MC.My_MarchingCube(TSDFManager[bou].Size, TSDFManager[bou].res, 0.0, GPUManager))
    # Mesh rendering
    MC[bou].runGPU(TSDFManager[bou].TSDFGPU)
    start_time3 = time.time()
    # save with the number of the body part
    bpStr = str(bp)
    MC[bou].SaveToPly("body" + bpStr + ".ply")
    elapsed_time = time.time() - start_time3
    print "SaveBPToPly: %f" % (elapsed_time)

    # Update number of vertices and faces in the stitched mesh
    nb_verticesGlo = nb_verticesGlo + MC[bou].nb_vertices[0]
    nb_facesGlo = nb_facesGlo + MC[bou].nb_faces[0]

    #angle = 0.5
    #StitchBdy.RArmsTransform(angle,bp, pos2d[0,0],RGBD[0],Tg)


    # Put the Global transfo in PoseBP so that the dtype entered in the GPU is correct
    for i in range(4):
        for j in range(4):
            PoseBP[i][j] = Tg[bp][i][j]
    # Stitch all the body parts
    if bp == bpstart:
        StitchBdy.StitchedVertices = StitchBdy.TransformVtx(MC[bou].Vertices, PoseBP, 1)
        StitchBdy.StitchedNormales = StitchBdy.TransformNmls(MC[bou].Normales, PoseBP, 1)
        StitchBdy.StitchedFaces = MC[bou].Faces
    else:
        StitchBdy.NaiveStitch(MC[bou].Vertices, MC[bou].Normales, MC[bou].Faces, PoseBP)

# save with the number of the body part
# bpStr = str(idx)   #"+bpStr+"
start_time3 = time.time()
MC[0].SaveToPlyExt("wholeBody.ply", nb_verticesGlo, nb_facesGlo, StitchBdy.StitchedVertices,
                   StitchBdy.StitchedFaces)
elapsed_time = time.time() - start_time3
print "SaveToPly: %f" % (elapsed_time)

#
#
# Tracking BB
#
#
TimeStart = time.time()
formerIdx = Index
Tbbw = []
for bp in range(nbBdyPart+1):
    Tbbw.append(Id4)
for imgk in range(Index+1, nunImg):
    print imgk
    if not lImages[0][imgk].any():
        print lImages[0][imgk]
        print("List lImages is empty")
        continue
    if not pos2d[0][imgk].any():
        #print pos2d[0][imgk]
        print("List pos2d is empty")
        continue
    if imgk == 72 or imgk == 87 or imgk == 17 or imgk == 31 or imgk == 36\
            or imgk == 37 or imgk == 39 or imgk == 46 or imgk == 47 or imgk == 49 or imgk == 50:
        continue
    # Time counting
    start = time.time()

    '''
    New Image 
    '''
    # Current Depth Image (i.e: i+1)
    newRGBD = []
    #Tg = []
    #Tg.append(Id4)
    Tbb = []
    Tbb.append(Id4)

    for bp in range(nbBdyPart):
        newRGBD.append(RGBD_.RGBD(path + '/Depth.tiff', path + '/RGB.tiff', intrinsic, fact))
        newRGBD[bp].LoadMat(lImages, pos2d, connection, bdyIdx)
        # Get new current image
        newRGBD[bp].ReadFromMat(imgk)
        newRGBD[bp].BilateralFilter(-1, 0.02, 3)
        # segmenting the body
        newRGBD[bp].Crop2Body()
        newRGBD[bp].BodySegmentation()
        newRGBD[bp].BodyLabelling()
        # select the body part
        if bp == 0:
            newRGBD[bp].depth_image *= (newRGBD[bp].labels > 0)
        else:
            newRGBD[bp].depth_image *= (newRGBD[bp].labels == bp)

        newRGBD[bp].Vmap_optimize()
        newRGBD[bp].NMap_optimize()
        # create the transform matrix from local to global coordinate
    newRGBD[0].myPCA()



    # restart processing of each body part for the current image.
    # Sum of the number of vertices and faces of all body parts
    nb_verticesGlo = 0
    nb_facesGlo = 0
    # Initiate stitcher object
    StitchBdy = Stitcher.Stitch(nbBdyPart)


    # Updating mesh of each body part
    for bp in range(1, nbBdyPart):
        #Tglo = newRGBD[0].TransfoBB[bp]
        #Tg.append(Tglo.astype(np.float32))
        Tbb.append(StitchBdy.GetBBTransfo(pos2d,imgk,formerIdx,RGBD[0],bp))
        Tbbw[bp] = np.dot(Tbb[bp],Tbbw[bp])
        # Transform in the current image
        #Tg_new = np.dot(T_Pose, Tg[bp])
        Tg_new = np.dot(Tbbw[bp],Tg[bp])
        # Put the Global transfo in PoseBP so that the dtype entered in the GPU is correct
        for i in range(4):
            for j in range(4):
                PoseBP[i][j] = Tg_new[i][j]  #Tbb[bp][i][j]# TgPrev[bp][i][j]#Tg[bp][i][j]#Tbb[bp][i][j]#
        TgPrev.append(Tglo.astype(np.float32))

        # Update number of vertices and faces in the stitched mesh
        nb_verticesGlo = nb_verticesGlo + MC[bp].nb_vertices[0]
        nb_facesGlo = nb_facesGlo + MC[bp].nb_faces[0]

        # Stitch all the body parts
        if bp == 1:
            StitchBdy.StitchedVertices = StitchBdy.TransformVtx(MC[bp].Vertices, PoseBP, 1)
            StitchBdy.StitchedNormales = StitchBdy.TransformNmls(MC[bp].Normales, PoseBP, 1)
            StitchBdy.StitchedFaces = MC[bp].Faces
        else:
            StitchBdy.NaiveStitch(MC[bp].Vertices, MC[bp].Normales, MC[bp].Faces, PoseBP)
    formerIdx = imgk
    TgPrev = []
    TgPrev.append(Id4)
    for bp in range(1, nbBdyPart):
        Tglo = newRGBD[0].TransfoBB[bp]
        TgPrev.append(Tglo.astype(np.float32))

    time_lapsed = time.time() - start
    print "numero %d finished : %f" % (imgk, time_lapsed)

    # save with the number of the body part
    start_time3 = time.time()
    imgkStr = str(imgk)
    MC[0].SaveToPlyExt("wholeBody" + imgkStr + ".ply", nb_verticesGlo, nb_facesGlo, StitchBdy.StitchedVertices,
                       StitchBdy.StitchedFaces)
    elapsed_time = time.time() - start_time3
    print "SaveToPly: %f" % (elapsed_time)

    top = tk.Tk()
    # projection in 2d space to draw it
    rendering = np.zeros((Size[0], Size[1], 3), dtype=np.uint8)
    # projection of the current image/ Overlay
    # rendering = RGBD.Draw_optimize(rendering,Id4, 1, color_tag)
    '''
    for bp in range(bpstart, nbBdyPart):
        bou = bp - bpstart + 1
        for i in range(4):
            for j in range(4):
                PoseBP[i][j] = Tg[bp][i][j]
        rendering = RGBD[0].DrawMesh(rendering, MC[bou].Vertices, MC[bou].Normales, PoseBP, 1, color_tag)
    '''
    rendering = RGBD[0].DrawMesh(rendering, StitchBdy.StitchedVertices,StitchBdy.StitchedNormales , Id4, 1, color_tag)
    # 3D reconstruction of the whole image
    canvas = tk.Canvas(top, bg="black", height=Size[0], width=Size[1])
    canvas.pack()
    #rendering = DrawColors2D(RGBD[0],rendering,Pose)
    img = Image.fromarray(rendering, 'RGB')
    imgTk = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor=tk.NW, image=imgTk)
    DrawSkeleton2Dbis(Pose,imgk)
    # DrawCenters2D(Pose)
    # DrawOBBox2D(Pose)
    #DrawPoint2D([205,142],5,'blue')


    canvas.pack()
    top.mainloop()

    top = tk.Tk()
    # projection in 2d space to draw it
    rendering = np.zeros((Size[0], Size[1], 3), dtype=np.uint8)
    # projection of the current image/ Overlay
    rendering = newRGBD[0].Draw_optimize(rendering,Id4, 1, color_tag)
    # 3D reconstruction of the whole image
    canvas = tk.Canvas(top, bg="black", height=Size[0], width=Size[1])
    canvas.pack()
    #rendering = DrawColors2D(RGBD[0],rendering,Pose)
    img = Image.fromarray(rendering, 'RGB')
    imgTk = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor=tk.NW, image=imgTk)
    DrawSkeleton2Dbis(Pose,imgk)
    # DrawCenters2D(Pose)
    # DrawOBBox2D(Pose)
    canvas.pack()
    top.mainloop()
