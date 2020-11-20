
import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast
import matplotlib.pyplot as plt
import argparse

from finalproject.PR.api import PRN
from finalproject.PR.utils.render import render_texture
import cv2
from finalproject.PR.demo_texture import faceSwap



class PRNET():

    def __init__(self, path):
        self.path = path
        parser = argparse.ArgumentParser(description='Texture Editing by PRN')

        parser.add_argument('-i', '--image_path', default='TestImages/AFLW2000/image00081.jpg', type=str, \
                        help='path to input image')
        parser.add_argument('-r', '--ref_path', default='TestImages/trump.jpg', type=str,\
                        help='path to reference image(texture ref)')
        parser.add_argument('-o', '--output_path', default='TestImages/output.jpg', type=str,\
                        help='path to save output')
        parser.add_argument('--mode', default=1, type=int,\
                        help='ways to edit texture. 0 for modifying parts, 1 for changing whole')
        parser.add_argument('--gpu', default="-1", type=str,\
                        help='set gpu id, -1 for CPU')

    # ---- init PRN
        os.environ['CUDA_VISIBLE_DEVICES'] = parser.parse_args().gpu # GPU number, -1 for CPU
        self.prn = PRN(is_dlib = True, prefix=self.path)
        self.args = parser.parse_args()



    def faceSwap(self, source, target, sourceBbox, targetBbox):
        return faceSwap(self.prn, self.args, source, target, sourceBbox, targetBbox)



