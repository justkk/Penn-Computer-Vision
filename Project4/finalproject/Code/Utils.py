
import numpy as np
import cv2

from finalproject.Code.FaceMaskUtility import *

def getFullFaceMask(shape, hull):

    mask = np.zeros(shape)
    mask = cv2.fillConvexPoly(mask, np.int32(hull), (255,255))
    return mask

def getPartialMask(points, shape):
    return get_face_mask(shape, points)

