"""Script to do dynamic fusion with topological changes."""
import sys
import cv2
import scipy.io
import numpy as np

'''Self made modules to structures the project'''
#==============================================================================
# sys.path.insert(0, '/home/nii-user/inoe/NIIComputerVision/code/lib')
# import depthMapConversion
# import tracking
# import volumetric
# import segmentation
#==============================================================================

def main():
    '''loading data'''
    print '''Choose your data
    By default Test2Box.mat will be select
    Enter 1 for FixedPose.mat
    '''
    select = raw_input()
    condition = select=='1'
    if condition:
        name = 'FixedPose.mat'
    else:
        name = 'Test2Box.mat'
    mat = scipy.io.loadmat('../data/'+ name)
    '''display the video stream'''
    lImages = mat['DepthImg']
    numbImages = len(lImages.transpose())
    for x in range(0,numbImages):
        cv2.imshow(name,lImages[0][x])  
        cv2.waitKey(0)
    cv2.destroyAllWindows()   
    return 0


if __name__ == '__main__':
    sys.exit(main())