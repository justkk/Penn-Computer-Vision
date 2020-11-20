import cv2

import numpy as np

import dlib

from imutils import face_utils

PATH = "shape_predictor_68_face_landmarks.dat"

class HistoryMarking():

    def __init__(self, pointSequence, imageSequence):
        self.pointSequence = []

    def addHistory(self, points):
        self.pointSequence.append(points)

def getFacePoints(image, useDlib):

  if useDlib == False:
      return None

  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(PATH)

  rects = detector(image, 1)
  print(rects)

  if(len(rects)==0):
    return None
  shape = predictor(image, rects[0])
  shape = face_utils.shape_to_np(shape)


  return shape



def getWeighedPoints(old_image,new_image,old_points,new_points):

  number = old_points.shape[0]
  #number = 68 + 56
  weighted_new = np.zeros([number,1,2])
  old_points = old_points.reshape([number,1, 2])

  lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

  klt_pts_new, st, err = cv2.calcOpticalFlowPyrLK(old_image, new_image, old_points.astype(np.float32), None, **lk_params)

  flag = 0

  if new_points is not None:
    if new_points.shape[0]==number:
      flag = 1
      new_points = new_points.reshape(number, 1, 2)

  print(klt_pts_new.shape)

  for pt_num in range(number):

      if (flag == 1 and st[pt_num] == 1):
          #print "path1"
          weighted_new[pt_num, :, :] = 1.0*new_points[pt_num, :, :] + 0.0*old_points[pt_num, :, :]

      elif (flag == 1 and st[pt_num] == 0):
          #print "path2"
          weighted_new[pt_num, :, :] = 0.2*old_points[pt_num, :, :] + 0.8*new_points[pt_num, :, :]

      elif (flag == 0 and st[pt_num] == 1):
          #print "path3"
          weighted_new[pt_num, :, :] = 0.2*old_points[pt_num, :, :] + 0.8*klt_pts_new[pt_num, :, :]
      else:
          #print "path4"
          weighted_new[pt_num, :, :] = old_points[pt_num, :, :]

  return weighted_new.reshape([number,2]).astype(int)


class Video_Land_Mark_Extractor():

    def __init__(self, openFace, path, params):
        self.openFace = openFace
        self.videoPath = path
        self.params = params
        self.frameInfromation = openFace.getFacialLandMarksInVideo(self.videoPath, self.params)


    def getLandMarkPoints(self, Image, frameIndex, params, videoPersonIndex = 0, photoPersonIndex = 0, ):

        pointsFromVideo = None
        pointsFromImage = None

        try:
            pointsFromImage = getFacePoints(Image, params["DLIB"])
            if pointsFromImage is None:
                pointsFromImage = self.openFace.getFacialLandMarks(Image, params)
        except:
            pointsFromImage = None

        try:
            pointsFromVideo = self.frameInfromation[frameIndex][videoPersonIndex]["pointArray"]

        except:
            pointsFromVideo = None

        if pointsFromVideo is None and pointsFromImage is None:
            return None
        elif pointsFromImage is None:
            return pointsFromVideo
        elif pointsFromVideo is None:
            return pointsFromImage

        return params["VIDEO_WEIGHT"] * pointsFromVideo + (1 - params["VIDEO_WEIGHT"]) * pointsFromImage



