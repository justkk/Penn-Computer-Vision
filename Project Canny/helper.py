from constants import *
from scipy.spatial import distance
import queue
from PIL import Image
import itertools
import scipy.stats
from edgeLink import *

def getLocalThreshold(Image):


  colSplit = np.arange(0,Image.shape[1], STRIDE_COL).astype(int)
  rowSplit = np.arange(0, Image.shape[0], STRIDE_ROW ).astype(int)

  startPoints = itertools.product(list(rowSplit), list(colSplit))

  startPoints = [start for start in startPoints]

  magMax = np.ones(Image.shape, dtype=int)
  magMin = np.ones(Image.shape, dtype=int)

  mean = np.mean(Image.reshape((-1)))
  varience = np.var(Image.reshape((-1)))


  chunks = [Image[start[0]:start[0]+STRIDE_ROW, start[1]:start[1]+STRIDE_COL] for start in startPoints]

  thresh = [getThreshold(chunk, mean, varience) for chunk in chunks]


  threshMax = [(chunks[index] * 0 + 1)*thresh[index][0] for index in range(len(chunks))]

  threshMin = [(chunks[index] * 0 + 1)*thresh[index][1] for index in range(len(chunks))]



  for l in range(len(list(chunks))):
    start = startPoints[l]
    magMax[start[0]:start[0]+STRIDE_ROW, start[1]:start[1]+STRIDE_COL] = threshMax[l]
    magMin[start[0]:start[0]+STRIDE_ROW, start[1]:start[1]+STRIDE_COL] = threshMin[l]


  return magMax, magMin

def getThreshold(Image, mean, varience):


  Image = 2*Image

  histogram, binEdges = np.histogram(Image.reshape((-1)), bins=np.arange(0,int(np.amax(Image)+1),1))

  probability = histogram * 1.0 / (Image.shape[0]*Image.shape[1])
  cumpro = np.cumsum(probability[::-1])[::-1]

  subImageMean = np.mean(Image.reshape((-1)))
  subImageVarience = np.var(Image.reshape((-1)))

  Np = np.sum(histogram * (histogram - 1))/2

  lmin = -4 * np.log(Np)/np.log(0.125)

  thMeaningfulLength = int(2.0*np.log(Image.shape[0]*Image.shape[1])/np.log(8.0)+0.5)

  pMax= (1/np.exp((np.log(Np)/thMeaningfulLength))) ;

  pMin= 1/np.exp((np.log(Np)/np.sqrt(Image.shape[0]*Image.shape[1])));

  thGradientHigh = 256 
  thGradientLow = 0

  index = len(histogram)-1

  eq = scipy.stats.norm(mean, varience)


  while index >=0 :

    if cumpro[index]  < eq.pdf(subImageMean): 
      index -= 1;

    else :
      break;


  for index in np.arange(index-1, -1, -1):

    if cumpro[index] > pMax :
      thGradientHigh = index
      break;

  for index in np.arange(len(histogram)-1, -1, -1):
    if cumpro[index] > pMin:
      thGradientLow = index
      break;

  if thGradientHigh != 256:
    thGradientHigh = np.sqrt(thGradientHigh * 70) 


  return thGradientHigh*0.8, thGradientLow*0.8

def edgeLinkLocalThreshold(M, Mag, Ori):

  ht,lt = getLocalThreshold(Mag)

  HIGHER_THRESHOLD = ht / np.amax(Mag)
  LOWER_THRESHOLD =  lt / np.amax(Mag)

  edgeMask = edgeLinkVectorized(M, Mag, Ori, HIGHER_THRESHOLD, LOWER_THRESHOLD);

  edgeMap = edgeMask * 1;

  return edgeMap.astype("uint8")


def edgeLinkGlobalThreshold(M, Mag, Ori):
  # TODO: your code here

  ht, lt = getThreshold(Mag, np.mean(Mag.reshape((-1))), np.var(Mag.reshape((-1))) )

  HIGHER_THRESHOLD = ht / np.amax(Mag)
  LOWER_THRESHOLD =  lt / np.amax(Mag)

  edgeMask = edgeLinkVectorized(M, Mag, Ori, HIGHER_THRESHOLD, LOWER_THRESHOLD);

  edgeMap = edgeMask * 1;

  return edgeMap.astype("uint8")


def getPointsFittingLine(group):
  xData = [] 
  yData = []
  for candidates in group:
    xData.append(candidates[0])
    yData.append(candidates[1])

  z = np.polyfit(np.array(xData), np.array(yData), 1)

  newCandidates = [] 


  for data in xData:
    newY = z[0]*data + z[1]
    newCandidates.append([data, int(np.round(newY))])

  return newCandidates


def getLineGroup(edgeMap, Ori, infoPerc):

  visitMap = np.zeros(edgeMap.shape, dtype=float)

  lineMap = np.zeros(edgeMap.shape, dtype=float)

  candidates = zip(*np.where(edgeMap == 1))

  candidateQueue = queue.Queue()

  for candidate in candidates:
    candidateQueue.put(candidate)


  groups={};
  length = [];


  while not candidateQueue.empty():

    candidate = candidateQueue.get();
    
    if visitMap[candidate] == 1:
      continue;

    visitMap[candidate] = 1;

    orientation = Ori[candidate]

    bulgingQueue = queue.Queue();
    bulgingQueue.put(candidate);

    oriList = [Ori[candidate]];
    groupList = [candidate];

    while not bulgingQueue.empty():
      candidate = bulgingQueue.get();
      visitMap[candidate] = 1;
      topPixel, bottomPixel = getTopandBottomPixel(candidate, orientation, edgeMap.shape)
      orientationIndex = getDirectionIndex(orientation)
      topPixelOrientationIndex = getDirectionIndex(Ori[topPixel])
      bottomPixelOrientationIndex = getDirectionIndex(Ori[bottomPixel])

      # print("Stats")
      # print(candidate)
      # print(Ori[candidate]*180/np.pi)
      # print(orientationIndex)
      # print("topPixel")
      # print(topPixel)
      # print(Ori[topPixel]*180/np.pi)
      # print(topPixelOrientationIndex)
      # print("bottomPixel")
      # print(bottomPixel)
      # print(Ori[bottomPixel]*180/np.pi)
      # print(bottomPixelOrientationIndex)
      # print("######################")
      
      topOne = False
      bottomOne = False

      if topPixelOrientationIndex == orientationIndex and visitMap[topPixel] == 0 and edgeMap[topPixel] == 1:
        newList = oriList + [Ori[topPixel]]
        if np.abs(np.var(np.array(newList)) - np.var(np.array(oriList))) < np.pi/8 : 
          topOne = True

      if bottomPixelOrientationIndex == orientationIndex and visitMap[bottomPixel] == 0 and edgeMap[bottomPixel] == 1:
        newList = oriList + [Ori[bottomPixel]]
        if np.abs(np.var(np.array(newList)) - np.var(np.array(oriList))) < np.pi/8 : 
          bottomOne = True

      if topOne:
        oriList.append(Ori[topPixel])
        bulgingQueue.put(topPixel);
        groupList.append(topPixel)

      if bottomOne:
        oriList.append(Ori[bottomPixel])
        bulgingQueue.put(bottomPixel)
        groupList.append(bottomPixel)

    if len(groupList) not in groups:
      groups[len(groupList)] = [];

    groups[len(groupList)].append(groupList)

    length.append(len(groupList))

  information =  0;
  totalInformation  = len(length)

  dataKeys = list(groups.keys());
  dataKeys.sort(reverse = True)

  i=0;
  while i <  len(dataKeys):
    infoLength = len(groups[dataKeys[i]])
    if information + infoLength <= infoPerc * totalInformation:
      information = information + infoLength
      i += 1;
      continue
    break

  shape = lineMap.shape

  newDataKeys = dataKeys[:i]

  for key in newDataKeys:
    for group in groups[key]:
      newPoints = group #getPointsFittingLine(group)
      for point in newPoints:
        if point[0] < shape[0] and point[1] < shape[1] and point[1] >=0 and point[0] >=0:
          lineMap[point[0]][point[1]] = 1


  lineMap = lineMap;

  return lineMap

