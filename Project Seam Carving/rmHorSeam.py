'''
  File name: rmHorSeam.py
  Author:
  Date created:
'''

'''
  File clarification:
    Removes horizontal seams. You should identify the pixel from My from which 
    you should begin backtracking in order to identify pixels for removal, and 
    remove those pixels from the input image. 
    
    - INPUT I: n × m × 3 matrix representing the input image.
    - INPUT My: n × m matrix representing the cumulative minimum energy map along horizontal direction.
    - INPUT Tby: n × m matrix representing the backtrack table along horizontal direction.
    - OUTPUT Iy: (n − 1) × m × 3 matrix representing the image with the row removed.
    - OUTPUT E: the cost of seam removal.
'''
from constants import *
from rmVerSeam import *

def rmHorSeam(I, My, Tby):
  # Your Code Here 

  Mx = np.transpose(My)

  Tbx = np.transpose(Tby)

  IUpdate = np.transpose(I, (1, 0, 2))

  Ix,E = rmVerSeam(IUpdate, Mx, Tbx)

  Iy = np.transpose(Ix, (1, 0, 2))

  return Iy, E
