# File created by Inoe ANDRE the 01-03-2017

# Define functions to do the segmentation in a depthmap image
import cv2
import numpy as np
from numpy import linalg as LA



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
        diffX = -B[0]+A[0]
        dist = np.sqrt(diffY*diffY + diffX*diffX)
        a = diffX/dist
        b = diffY/dist
        c = -a*A[0]-b*A[1]
        return np.array([a,b,c])
    
    def inferedPoint(A,a,b,c,point,T=100):
        process_y = abs(a) > abs(b) 
        if process_y:
            y = point[1]
            while 1:
                y = y-1
                x = int(np.round(-(b*y+c)/a))
                if A[y,x]==0:
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
                x = int(np.round(-(b*y+c)/a))
                if A[y,x]==0:
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
                y = int(np.round(-(a*x+c)/b))
                inImage = (x>0) and (x<=512) and (y>0) and (y<=424)
                if inImage:
                    if A[y,x]==0:
                        x_left = x
                        y_left = y
                        break
                    else:
                        distCdt = LA.norm([x,y]-point)>T
                        if distCdt:#sqrt((x-point(1))^2+(y-point(2))^2)>T
                            x_left = x
                            y_left = y
                            break
                else:
                    x_left = x+1
                    y_left = np.round(-(a*x_left+c)/b)
                    break
        
            x = point[0]
            while 1:
                x = x+1
                y = int(np.round(-(a*x+c)/b))
                inImage = (x>0) and (x<=512) and (y>0) and (y<=424)
                if inImage:
                    if A[y,x]==0:
                        x_right = x
                        y_right = y
                        break
                    else:
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
    
    def forearmLeft(A,pos2D,B):
        #this function segment the left arm
        
        # pos2D[4] = Shoulder_Left
        # pos2D[5] = Elbow_Left
        # pos2D[6] = Wrist_Left
        pos2D = pos2D.astype(np.float64)
        # First let us see the down limit thanks to the elbow and the wrist (left side)
        # FindSlopes give the slope of a line made by two point
        slopes0=findSlope(pos2D[5],pos2D[6])#[a67,b67,~]
        a_pen67 = -slopes0[1]#-b67;
        b_pen67 = slopes0[0]#a67;
        
        slopes1=findSlope(pos2D[5],pos2D[4])#[a65,b65,~];
        
        a_pen = slopes0[0] + slopes1[0]#a67+a65;
        b_pen = slopes0[1] + slopes1[1]#b67+b65;
        if (a_pen == b_pen) and (a_pen==0):
            a_pen = slopes1[1]#b65;
            b_pen =-slopes1[0]#a65;

        c_pen = -(a_pen*pos2D[5,0]+b_pen*pos2D[5,1]);
        
        
        # find 2 points elbow
        bone1 = LA.norm(pos0[5]-pos0[6])#sqrt( sum( (pos2D(6,:)-pos2D(7,:)).^2 ) );
        bone2 = LA.norm(pos0[5]-pos0[4])#sqrt( sum( (pos2D(6,:)-pos2D(5,:)).^2 ) );
        bone = max(bone1,bone2);
        p1=B[pos0[5,1],pos0[5,0]]#(pos2D(6,2),pos2D(6,1));
        p2=B[pos0[6,1],pos0[6,0]]#(pos2D(7,2),pos2D(7,1));
        #threshold the imqge to get just the interesting part of the body
        A1 = B*(B>(min(p1,p2)-50)) * (B<(max(p1,p2)+50));
        
        # compute the intersection between the slope and the extremety of the body
        intersection_elbow=inferedPoint(A1,a_pen,b_pen,c_pen,pos2D[5],0.5*bone)#[left,right];
        vect_elbow = -pos2D[5]+intersection_elbow[0]
        
        # find 2 points wrist
        c_pen67=-(a_pen67*pos2D[6,0]+b_pen67*pos2D[6,1]);
        intersection_wrist=inferedPoint(A,a_pen67,b_pen67,c_pen67,pos2D[6],bone/3);
        vect_wrist = -pos2D[6]+intersection_wrist[0]
        vect67 = pos2D[6]-pos2D[5]
        vect67_pen = [vect67[1] - vect67[0]]
        if sum(vect67_pen*vect_elbow)*sum(vect67_pen*vect_wrist)<0:
           x = intersection_elbow[0]
           intersection_elbow[0] = intersection_elbow[1]
           intersection_elbow[1] = x;   

        pt4D = [intersection_elbow[0],intersection_elbow[1],intersection_wrist[0],intersection_wrist[1]]
        pt4D_bis = [intersection_wrist[0],intersection_elbow[0],intersection_elbow[1],intersection_wrist[1]]
        finalSlope=findSlopes(pt4D,pt4D_bis)
        x = np.isnan(finalSlope[0])
        #erase all nan in the array
        finalSlope[0]=finalSlope[0][~np.isnan(finalSlope[0])]
        finalSlope[1]=finalSlope[1][~np.isnan(finalSlope[1])]
        finalSlope[2]=finalSlope[2][~np.isnan(finalSlope[2])]
        midpoint = [(pos2D[5,0]+pos2D[6,0])/2, (pos2D[5,1]+pos2D[6,1])/2]
        ref= np.array([finalSlope[0]*midpoint[0] + finalSlope[1]*midpoint[1] + finalSlope[2]]).astype(np.float32)
        bw_up = A*polygon(kernel,a,b,c,ref,4-sum(x))>0
        
        # pos2D[2] = Shoulder_Center
        # pos2D[3] = Head
        
        #compute slopes
        slopesSH=findSlope(pos2D[2],pos2D[3])
        a_pen = slopesSH[1]
        b_pen = - slopesSH[0]
        c_pen = -(a_pen*pos2D[2,0]+b_pen*pos2D[2,1]);
        
        # compute the intersection between the slope and the extremety of the body
        intersection_head=inferedPoint(A,a_pen,b_pen,c_pen,pos2D[2])
        
        slopes215=findSlope(pos2D[20],pos2D[4])
        
        a_pen = slopes215[0]+slopesSH[0]
        b_pen = slopes215[1]+slopesSH[1]
        if (a_pen == b_pen) and (a_pen==0):
            a_pen = slopes215[1]
            b_pen = -slopes215[0]

        c_pen = -(a_pen*pos2D[4,0]+b_pen*pos2D[4,1]);
        
        intersection215=inferedPoint(A,a_pen,b_pen,c_pen,pos2D[4])
        vect65 = pos2D[4]-pos2D[5]
        vect65_pen = [vect65[1] - vect65[0]]
        
        vect_elbow = -pos2D[5]-intersection_elbow[0]
        vect_elbow1 = -pos2D[4] - intersection215[0]
        #cross product 
        t = np.cross([vect_elbow, 0],[vect65, 0]);
        t1 = np.cross([vect_elbow1,0],[-vect65,0]);
        if t1[2]>0:
            intersection215[0] = intersection215[1]

        if t[2]<0:
            c = intersection_elbow[0]
            intersection_elbow[0] = intersection_elbow[1]
            intersection_elbow[1] = slopes215[2]

        
        B =  np.logical_and( (A==0),polygonOutline(pos2D[[5, 4, 20, 0],:]));
        f=find(B)# [fy,fx];
        d = np.argmin(np.sum( np.array([pos2D[20,0]-f[1], pos2D[20,1]-f[0]]).transpose()*np.array([pos2D[20,0]-f[1], pos2D[20,1]-f[0]]).transpose()  ))
        peakArmpit = np.array([f[1][d],fy[0][d]])
        ptA = np.concatenate((intersection_elbow[0],intersection215[0],intersection_head[0],peakArmpit,intersection_elbow[1]))
        bw_upper = A*polygonOutline(ptA)>0
        return np.array([bw_up,bw_upper])
    