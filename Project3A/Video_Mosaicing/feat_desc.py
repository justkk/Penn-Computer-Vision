'''
  File name: feat_desc.py
  Author:
  Date created:
'''

'''
  File clarification:
    Extracting Feature Descriptor for each feature point. You should use the subsampled image around each point feature, 
    just extract axis-aligned 8x8 patches. Note that it’s extremely important to sample these patches from the larger 40x40 
    window to have a nice big blurred descriptor. 
    - Input img: H × W matrix representing the gray scale input image.
    - Input x: N × 1 vector representing the column coordinates of corners.
    - Input y: N × 1 vector representing the row coordinates of corners.
    - Outpuy descs: 64 × N matrix, with column i being the 64 dimensional descriptor (8 × 8 grid linearized) computed at location (xi , yi) in img.
'''

from scipy.interpolate import griddata

import numpy as np

import itertools

from PIL import Image



def feat_desc(img, x, y):
  # Your Code Here

  return implOne(img, x, y)




# img = np.arange(1,21).reshape((5,4))

# orientation = np.zeros((5,4), dtype="float")

# orientation[2,2] = np.pi/6;

# print(img)

# print(orientation)

# print(np.round(implOneGenerateDALL(img, np.array([2]), np.array([2]),  orientation - np.pi/2)))
