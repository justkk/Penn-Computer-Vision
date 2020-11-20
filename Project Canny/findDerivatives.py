'''
  File name: findDerivatives.py
  Author:
  Date created:
'''

'''
  File clarification:
    Compute gradient information of the input grayscale image
    - Input I_gray: H x W matrix as image
    - Output Mag: H x W matrix represents the magnitude of derivatives
    - Output Magx: H x W matrix represents the magnitude of derivatives along x-axis
    - Output Magy: H x W matrix represents the magnitude of derivatives along y-axis
    - Output Ori: H x W matrix represents the orientation of derivatives
'''

from constants import *
from utils import *
from scipy import signal
from PIL import Image
import numpy as np

def findDerivatives(I_gray):
  # TODO: your code here

  gaussianKernel = GaussianPDF_2D(GAUSSIAN_MEAN, GAUSSIAN_VAR, GAUSSIAN_SIZE_ROWS, GAUSSIAN_SIZE_COLS)

  filteredImage =  signal.convolve2d(I_gray, gaussianKernel, 'same')

  I_x_d = signal.convolve2d(filteredImage, SOBEL_X, 'same')

  I_y_d = signal.convolve2d(filteredImage, SOBEL_Y, 'same')

  I_m_d = np.power(np.power(I_x_d, 2) + np.power(I_y_d, 2), 1/2)

  I_d_d = -1.0 * I_y_d / (I_x_d + DIVIDE_EPSILON)

  return I_m_d, I_x_d, I_y_d, np.arctan(I_d_d)

