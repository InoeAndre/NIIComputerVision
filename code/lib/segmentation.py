# File created by Inoe ANDRE the 01-03-2017

# Define functions to do the segmentation in a depthmap image
import cv2
import numpy as np
from numpy import linalg as LA
import math


class Segmentation():
    
        # Constructor
    def __init__(self, depthname, colorname, pos2D):
        self.depthname = depthname
        self.colorname = colorname
        self.pos2D = pos2D
        
        
    def findSlope(A,B):
        #Get the slope of a line made from two point A and B
        A = A.astype(np.float32)
        B = B.astype(np.float32)
        diffY = B[1]-A[1]
        diffX = B[0]-A[0]
        dist = math.sqrt(diffY*diffY + diffX*diffX)
        a = diffX/dist
        b = diffY/dist
        c = -a*A[0]+b*A[1]
        return [a,b,c]
    
    def inferredPoint(A,a,b,c,point,T=100):
        process_y = math.abs(a) > math.abs(b) 
        if process_y:
            y = point[1]
            while 1:
                y = y-1
                x = np.round(-(b*y+c)/a)
                if A[y,x]!=0:
                    x_up = x
                    y_up = y
                    break
                else:
                    distCdt = LA.norm([x,y]-point)>T
                    if distCdt:#sqrt((x-point(1))^2+(y-point(2))^2)>T:
                        x_up = x
                        y_up = y
                        break
            y = point[1]
            while 1:
                y = y+1
                x = np.round(-(b*y+c)/a)
                if A[y,x]!=0:
                    x_down = x
                    y_down = y
                    break
                else:
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
        else:#process_x
        x = point[0]
        while 1:
            x = x-1
            y = np.round(-(a*x+c)/b)
            if (x>0) && (x<=512) && (y>0) && (y<=424):
                if A[y,x]!=0
                    x_left = x;
                    y_left = y;
                    break;
                else:
                    distCdt = LA.norm([x,y]-point)>T
                    if distCdt:#sqrt((x-point(1))^2+(y-point(2))^2)>T
                        x_left = x;
                        y_left = y;
                        break
            else
                x_left = x+1;
                y_left = np.round(-(a*x_left+c)/b);
                break
    
        x = point[0]
        while 1:
            x = x+1;
            y = np.round(-(a*x+c)/b);
            if (x>0) && (x<=512) && (y>0) && (y<=424)
                if ~A(y,x)
                    x_right = x;
                    y_right = y;
                    break;
                else
                    if sqrt((x-point(1))^2+(y-point(2))^2)>T
                        x_right = x;
                        y_right = y;
                        break
            else
                x_right = x-1;
                y_right = np.round(-(a*x_right+c)/b);
                break
    
        left = [x_left y_left];
        right = [x_right y_right];
        return [left, right]
    
    def forearmLeft(A,pos2D,kernel,B):
        return [bw_up,bw_upper] 
    