"""Script to do dynamic fusion with topological changes."""
import sys
import cv2
import scipy.io
import numpy as np
import Tkinter as tk
import imp

'''Self made modules to structures the project'''
#==============================================================================
# sys.path.insert(0, '/home/nii-user/inoe/NIIComputerVision/code/lib')
# import depthMapConversion
# import tracking
# import volumetric
# import segmentation
#==============================================================================

def main():
    ''' Create Menu to load data '''
    M = imp.load_source('Menu', './lib/Menu.py')
    root = tk.Tk()
    menu_app = M.Menu(root)
    menu_app.mainloop()
    
    A = imp.load_source('Menu', './lib/Application.py')
    root = tk.Tk()
    app = A.Application(menu_app.filename, root)
    app.mainloop()
    
    return 0


if __name__ == '__main__':
    main()
    exit(0)