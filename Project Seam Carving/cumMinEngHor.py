'''
  File name: cumMinEngHor.py
  Author:
  Date created:
'''

'''
  File clarification:
    Computes the cumulative minimum energy over the horizontal seam directions.
    
    - INPUT e: n × m matrix representing the energy map.
    - OUTPUT My: n × m matrix representing the cumulative minimum energy map along horizontal direction.
    - OUTPUT Tby: n × m matrix representing the backtrack table along horizontal direction.
'''


from constants import *


def cumMinEngHor(e):
  # Your Code Here

  eTranspose = np.transpose(e)

  Myt, Tbyt = helperUtils.findVerticalSeam(eTranspose)

  My =  np.transpose(Myt)

  Tby = np.transpose(Tbyt)

  return My, Tby