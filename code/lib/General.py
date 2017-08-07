"""
@Author Inoe ANDRE
Cross module functions
"""

import numpy as np
from numpy import linalg as LA




def in_mat_zero2one(mat):
    """
    Replace in the matrix all the 0 to 1
    :param mat: input matrix containing 0
    :return:  mat with 1 instead of 0
    """
    mat_tmp = (mat != 0.0)
    res = mat * mat_tmp + ~mat_tmp
    return res



def InvPose(Pose):
    """
    Compute the inverse transform of Pose
    :param Pose: 4*4 Matrix of the camera pose
    :return: matrix containing the inverse transform of Pose
    y = R*x + T
    x = R^(-1)*y + R^(-1)*T
    """
    PoseInv = np.zeros(Pose.shape, Pose.dtype)
    # Inverse rotation part R^(-1)
    PoseInv[0:3, 0:3] = LA.inv(Pose[0:3, 0:3])
    # Inverse Translation part R^(-1)*T
    PoseInv[0:3, 3] = -np.dot(PoseInv[0:3, 0:3], Pose[0:3, 3])
    PoseInv[3, 3] = 1.0
    return PoseInv

def normalized_cross_prod(a, b):
    '''
    Compute the cross product of 2 vectors and normalized it
    :param a: first 3 elements vector
    :param b: second 3 elements vector
    :return: the normalized cross product between 2 vector
    '''
    res = np.zeros(3, dtype="float")
    if (LA.norm(a) == 0.0 or LA.norm(b) == 0.0):
        return res
    a = a / LA.norm(a)
    b = b / LA.norm(b)
    res[0] = a[1] * b[2] - a[2] * b[1]
    res[1] = -a[0] * b[2] + a[2] * b[0]
    res[2] = a[0] * b[1] - a[1] * b[0]
    if (LA.norm(res) > 0.0):
        res = res / LA.norm(res)
    return res


def division_by_norm(mat, norm):
    '''
    This fonction divide a n by m by p=3 matrix, point by point, by the norm made through the p dimension
    It ignores division that makes infinite values or overflow to replace it by the former mat values or by 0
    :param mat:
    :param norm:
    :return:
    '''
    for i in range(3):
        with np.errstate(divide='ignore', invalid='ignore'):
            mat[:, :, i] = np.true_divide(mat[:, :, i], norm)
            mat[:, :, i][mat[:, :, i] == np.inf] = 0
            mat[:, :, i] = np.nan_to_num(mat[:, :, i])
    return mat


def normalized_cross_prod_optimize(a, b):
    """
    Compute the cross product of list of 2 vectors and normalized it
    :param a: first 3 elements vector
    :param b: second 3 elements vector
    :return: the normalized cross product between 2 vector
    """
    # res = np.zeros(a.Size, dtype = "float")
    norm_mat_a = np.sqrt(np.sum(a * a, axis=2))
    norm_mat_b = np.sqrt(np.sum(b * b, axis=2))
    # changing every 0 to 1 in the matrix so that the division does not generate nan or infinite values
    norm_mat_a = in_mat_zero2one(norm_mat_a)
    norm_mat_b = in_mat_zero2one(norm_mat_b)
    # compute a/ norm_mat_a
    a = division_by_norm(a, norm_mat_a)
    b = division_by_norm(b, norm_mat_b)
    # compute cross product with matrix
    res = np.cross(a, b)
    # compute the norm of res using the same method for a and b
    norm_mat_res = np.sqrt(np.sum(res * res, axis=2))
    norm_mat_res = in_mat_zero2one(norm_mat_res)
    # norm division
    res = division_by_norm(res, norm_mat_res)
    return res