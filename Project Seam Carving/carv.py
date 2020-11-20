'''
  File name: carv.py
  Author:
  Date created:
'''

'''
  File clarification:
    Aimed to handle finding seams of minimum energy, and seam removal, the algorithm
    shall tackle resizing images when it may be required to remove more than one seam, 
    sequentially and potentially along different directions.
    
    - INPUT I: n × m × 3 matrix representing the input image.
    - INPUT nr: the numbers of rows to be removed from the image.
    - INPUT nc: the numbers of columns to be removed from the image.
    - OUTPUT Ic: (n − nr) × (m − nc) × 3 matrix representing the carved image.
    - OUTPUT T: (nr + 1) × (nc + 1) matrix representing the transport map.
'''
from genEngMap import *

from constants import *

from cumMinEngVer import *

from cumMinEngHor import *

from rmHorSeam import *

from rmVerSeam import *


def computeColRemoval(I):

  energyMap = genEngMap(I)

  energyMap = energyMap.astype("float")

  #energyMap = I[:,:,0]

  Mx, Tbx = cumMinEngVer(energyMap)

  newImg, energy = rmVerSeam(I, Mx, Tbx)

  return newImg, energy


def computeRowRemoval(I):

  energyMap = genEngMap(I)

  energyMap = energyMap.astype("float")

  #energyMap = I[:,:,0]

  My, Tby = cumMinEngHor(energyMap)

  newImg, energy = rmHorSeam(I, My, Tby)

  return newImg, energy




def wrapper(I, nr, nc):

  dpStorage = {}

  dpStorage["path"] = {}

  T = np.zeros((nr+1, nc+1), dtype="float")


  for i in range(nr+1):
    for j in range(nc+1):

      if i == 0 and j == 0:
        dpStorage[(0,0)] = (I, 0)
        dpStorage["path"][(i,j)] = (i,j)
        T[i][j] = 0
        continue

      elif i == 0:

        finalColRemoval = dpStorage[(i, j-1)]
        colValue, colEnergy = computeColRemoval(finalColRemoval[0])
        colEnergy =  finalColRemoval[1] + colEnergy
        dpStorage[(i,j)] = (colValue, colEnergy)
        dpStorage["path"][(i,j)] = (i,j-1)

      elif j == 0:
        finalRowRemoval = dpStorage[(i-1,j)]
        rowValue, rowEnergy = computeRowRemoval(finalRowRemoval[0])
        rowEnergy =  finalRowRemoval[1] + rowEnergy
        dpStorage[(i,j)] = (rowValue, rowEnergy)
        dpStorage["path"][(i,j)] = (i-1,j)

      else:

        finalRowRemoval = dpStorage[(i-1,j)]

        finalColRemoval = dpStorage[(i, j-1)]

        rowValue, rowEnergy = computeRowRemoval(finalRowRemoval[0])

        # print("### Row ", i,j)
        # print(finalRowRemoval[0][:,:,0])
        # print(rowValue[:,:,0])
        # print(rowEnergy)

        rowEnergy =  finalRowRemoval[1] + rowEnergy

        colValue, colEnergy = computeColRemoval(finalColRemoval[0])

        # print("### COL ",i,j)
        # print(finalColRemoval[0][:,:,0])
        # print(colValue[:,:,0])
        # print(colEnergy)

        colEnergy =  finalColRemoval[1] + colEnergy

        if rowEnergy > colEnergy: 
          dpStorage[(i,j)] = (rowValue, rowEnergy)
          dpStorage["path"][(i,j)] = (i-1,j)

        else:
          dpStorage[(i,j)] = (colValue, colEnergy)
          dpStorage["path"][(i,j)] = (i,j-1)

      #print((i,j), dpStorage[(i,j)][0][:,:,0])
      print((i,j))
      T[i][j] = dpStorage[(i,j)][1]

  print("Done ...")
  return dpStorage[(nr,nc)][0], T, dpStorage


def carv(I, nr, nc):
  # Your Code Here 
  
  Ic, T, dataStore = wrapper(I, nr, nc)

  return Ic, T