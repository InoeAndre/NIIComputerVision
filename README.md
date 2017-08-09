# NIIComputerVision

This project use python 2.7

To run the code you need to install several libraries :

OpenCV
numpy


Using GPU:
1) Install GPU drivers (depend on your GPU). Not needed on MAC
2) Install OpenCL. Not needed on MAC
3) Install pyopenCL
    for windows : https://anaconda.org/conda-forge/pyopencl
    for Mac or Linux : https://anaconda.org/timrudge/pyopencl
Be sure to adapt the code in GPUManager.py and KernelsOpenCL.py with your GPU devices.



To visualize mesh:

MeshLab (software)
http://www.meshlab.net/

mayavi (directly with the code but limited compare to MeshLab) :
conda install mayavi

./meshes contains all the .ply file that are created during the execution of the code
./lib contains all the .pu file that are module of the project
./images is used for displaying a menu.