'''
  File name: rmVerSeam.py
  Author:
  Date created:
'''

'''
  File clarification:
    Removes vertical seams. You should identify the pixel from My from which 
    you should begin backtracking in order to identify pixels for removal, and 
    remove those pixels from the input image. 
    
    - INPUT I: n × m × 3 matrix representing the input image.
    - INPUT Mx: n × m matrix representing the cumulative minimum energy map along vertical direction.
    - INPUT Tbx: n × m matrix representing the backtrack table along vertical direction.
    - OUTPUT Ix: n × (m - 1) × 3 matrix representing the image with the row removed.
    - OUTPUT E: the cost of seam removal.
'''

from constants import *

def rmVerSeam(I, Mx, Tbx):
  # Your Code Here 

  rows, cols = Mx.shape

  newImage = np.zeros((I.shape[0],I.shape[1]-1,I.shape[2]), dtype="float")

  minValueEnergy, valuesIndexes = helperUtils.findPathAndUpdateMatrix(Mx, Tbx)

  mapStore = helperUtils.findUpdateMatrix(Mx, valuesIndexes)

  channels = I.shape[2]

  for channel in range(channels):
    currentChannel = I[:,:,channel]
    newChannel = helperUtils.makeImageFromMapImage(currentChannel, mapStore)
    newImage[:,:,channel] = newChannel

  return newImage, minValueEnergy
