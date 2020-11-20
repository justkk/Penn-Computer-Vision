'''
  File name: cumMinEngVer.py
  Author:
  Date created:
'''

'''
  File clarification:
    Computes the cumulative minimum energy over the vertical seam directions.
    
    - INPUT e: n × m matrix representing the energy map.
    - OUTPUT Mx: n × m matrix representing the cumulative minimum energy map along vertical direction.
    - OUTPUT Tbx: n × m matrix representing the backtrack table along vertical direction.
'''

from constants import *

def cumMinEngVer(e):
  # Your Code Here 
  Mx, Tbx = helperUtils.findVerticalSeam(e)
  return Mx, Tbx