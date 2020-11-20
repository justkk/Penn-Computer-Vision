import numpy as np
import dlib
import cv2
from imutils import face_utils
from scipy.spatial import Delaunay
from finalproject.Code.OPEN_FACE import Open_Face

import finalproject.Code.FaceMaskUtility as FaceMaskUtility

srcVideo = "/Users/nikhilt/PycharmProjects/FS/finalproject/CIS581Project4PartCDatasets/Hard/Joker.mp4"
targetVideo = "/Users/nikhilt/PycharmProjects/FS/finalproject/CIS581Project4PartCDatasets/Easy/MrRobot.mp4"
PATH = "shape_predictor_68_face_landmarks.dat"
instance = Open_Face(installationPath="/Users/nikhilt/Desktop/OpenFace", savePath="/Users/nikhilt/Desktop/playground/input",\
                     outputPath="/Users/nikhilt/Desktop/playground/output")

def getFacePoints(image):

  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(PATH)

  rects = detector(image, 1)
  print(rects)

  if(len(rects)==0):
    return None
  shape = predictor(image, rects[0])
  shape = face_utils.shape_to_np(shape)

  return shape

def getFacePoints2(inputImage):
  inputImage = np.asarray(inputImage)
  points = instance.getFacialLandMarks(inputImage)
  return points

def getFaceMask(points,shape):

  hullIndex = cv2.convexHull(points, returnPoints = False)
  hull =[]

  for i in range(0, len(hullIndex)):
      hull.append(points[int(hullIndex[i])])

  #print(hull)

  mask = np.zeros(shape)

  mask = cv2.fillConvexPoly(mask, np.int32(hull), (255,255))

  return mask,hull


def getFaceMask2(points, shape):
  hullIndex = cv2.convexHull(points, returnPoints = False)
  hull =[]
  for i in range(0, len(hullIndex)):
    hull.append(points[int(hullIndex[i])])

  mask = np.zeros(shape)

  mask = FaceMaskUtility.get_face_mask(mask, points)

  return mask,hull


def warpTriangle(img1, img2, tri1, tri2) :
    
    # Find bounding rectangle for each triangle
  r1 = cv2.boundingRect(tri1)
  r2 = cv2.boundingRect(tri2)
    
    # Offset points by left top corner of the respective rectangles
  tri1Cropped = []
  tri2Cropped = []
    
  for i in range(0, 3):
    tri1Cropped.append(((tri1[i][0] - r1[0]),(tri1[i][1] - r1[1])))
    tri2Cropped.append(((tri2[i][0] - r2[0]),(tri2[i][1] - r2[1])))

    # Crop input image
  img1Cropped = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    # Given a pair of triangles, find the affine transform.
  warpMat = cv2.getAffineTransform( np.float32(tri1Cropped), np.float32(tri2Cropped) )
    
    # Apply the Affine Transform just found to the src image
  img2Cropped = cv2.warpAffine( img1Cropped, warpMat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    # Get mask by filling triangle
  mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
  cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0);

  img2Cropped = img2Cropped * mask
    
    # Copy triangular region of the rectangular patch to the output image
  img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
    
  img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Cropped


def main():

  vidcap = cv2.VideoCapture(targetVideo)
  success,targetImage = vidcap.read()

  print(success)

  targetImage = cv2.resize(targetImage,(640, 480))

  target_gray = cv2.cvtColor(targetImage, cv2.COLOR_BGR2GRAY)

  target_face_points = getFacePoints(target_gray)

  # src_face = src_gray.copy()

  # src_face[src_mask == 0] =0

  # target_face = target_gray.copy()

  # target_face[target_mask ==0] =0

  #src_face_points_reshaped = src_face_points.reshape(68,2)        
  # for (x, y) in src_face_points:
  #     cv2.circle(src_gray, (np.int(x), np.int(y)), 1, (0, 0, 255), -1)

  # for (x, y) in target_face_points:
  #     cv2.circle(target_gray, (np.int(x), np.int(y)), 1, (0, 0, 255), -1)
  
  # cv2.imshow('src face with points', src_gray)
  # cv2.imshow('target face with points', target_gray)

  #print(src_face_points.shape)
  # src_pts_appended = np.append(src_face_points,np.array([[0,0],[0,src_gray.shape[0]-1],[src_gray.shape[1]-1,0], [src_gray.shape[1]-1,src_gray.shape[0]-1]]), axis=0)
  # target_pts_appended = np.append(target_face_points,np.array([[0,0],[0,target_gray.shape[0]-1],[target_gray.shape[1]-1,0], [target_gray.shape[1]-1,target_gray.shape[0]-1]]), axis=0)

  # src_tri = Delaunay(src_pts_appended)
  # target_tri =  Delaunay(target_pts_appended)

  # print(src_tri.simplices.shape)
  # print(src_tri.simplices.shape)

  # warped_img = srcImage.copy()
  # for tri in src_tri.simplices:
  #   warpTriangle(targetImage,warped_img,target_pts_appended[tri],src_pts_appended[tri])

  # cv2.imshow("warped",warped_img)

  # src_mask,hull = getFaceMask(src_face_points,src_gray.shape)
  # src_face = src_gray.copy()
  # warped_img[src_mask == 0] =0

  # cv2.imshow("warped",warped_img)

  # rect = cv2.boundingRect(np.float32([hull]))
    
  # center = ((rect[0]+int(rect[2]/2), rect[1]+int(rect[3]/2)))

  # output = cv2.seamlessClone(warped_img, srcImage, np.uint8(src_mask), center, cv2.NORMAL_CLONE)

  # cv2.imshow("output",output)

  # cv2.imshow("Source Gray",src_gray)

  # cv2.imshow("Source Mask",src_mask)

  # cv2.imshow("Source Face",src_face)

  # cv2.imshow("Target Gray",target_gray)

  # cv2.imshow("Target Mask",target_mask)

  # cv2.imshow("Target Face",target_face)


  # rows,cols = src_mask.shape
  # tri_src_points = np.float32([src_face_points[0],src_face_points[9],src_face_points[17]])
  # tri_tgt_points = np.float32([target_face_points[0],target_face_points[9],target_face_points[17]])

  # print(tri_src_points)

  # M = cv2.getAffineTransform(tri_tgt_points,tri_src_points)

  # tgt_img = cv2.warpAffine(targetImage,M,(cols,rows))

  # cv2.imshow("Source Image", tgt_img)

  # print(M)
  # print(tgt_img.shape)
  # print(src_gray)

  # center = tuple(np.array(src_face_points[34]))
  # print(center)

  # output = cv2.seamlessClone(tgt_img, srcImage, np.uint8(src_mask), center, cv2.NORMAL_CLONE)

  fourcc = cv2.VideoWriter_fourcc(*'MJPG')
  out = cv2.VideoWriter('outputeasy2.avi',fourcc, 20, (640, 480),True)
  cap = cv2.VideoCapture(srcVideo)
  while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # print("Frame count: %d" %count)
        #count +=1
        # Ic, T = carv(frame,nr,nc)
        # height, width, channels = frame.shape
        # print("Original frame size: %d %d %d" % (height, width, channels))
        #Ic = np.pad(Ic,((0,nr),(0,nc)),'constant')
        srcImage = cv2.resize(frame,(640, 480))
        src_gray = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
        src_face_points = getFacePoints(src_gray)
        if src_face_points is None:
          continue
        print(src_face_points.shape)
        src_pts_appended = np.append(src_face_points,np.array([[0,0],[0,src_gray.shape[0]-1],[src_gray.shape[1]-1,0], [src_gray.shape[1]-1,src_gray.shape[0]-1]]), axis=0)
        target_pts_appended = np.append(target_face_points,np.array([[0,0],[0,target_gray.shape[0]-1],[target_gray.shape[1]-1,0], [target_gray.shape[1]-1,target_gray.shape[0]-1]]), axis=0)

        src_tri = Delaunay(src_pts_appended)
        target_tri =  Delaunay(target_pts_appended)

        # plt.triplot(src_pts_appended[:,0], src_pts_appended[:,1], src_tri.simplices.copy())
        # plt.plot(src_pts_appended[:,0], src_pts_appended[:,1], 'o')
        # plt.show()

        print(src_tri.simplices.shape)
        print(src_tri.simplices.shape)

        warped_img = srcImage.copy()
        for tri in src_tri.simplices:
          warpTriangle(targetImage,warped_img,target_pts_appended[tri],src_pts_appended[tri])

        #cv2.imshow("warped",warped_img)

        src_mask,hull = getFaceMask(src_face_points,src_gray.shape)
        src_face = src_gray.copy()
        warped_img[src_mask == 0] =0

        #cv2.imshow("warped",warped_img)

        rect = cv2.boundingRect(np.float32([hull]))

        center = ((rect[0]+int(rect[2]/2), rect[1]+int(rect[3]/2)))

        output = cv2.seamlessClone(warped_img, srcImage, np.uint8(src_mask), center, cv2.NORMAL_CLONE)

        #cv2.imshow("output",output)
        #src_mask = getFaceMask(src_face_points,src_gray.shape)
        #src_face = src_gray.copy()
        #src_face[src_mask == 0] =0
        out.write(output)
        # height, width, channels = Ic.shape
        # print("Size after seam removal: %d %d %d" % (height, width, channels))
        # print("\n")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
  cap.release()
  out.release()
  cv2.destroyAllWindows()

  #src_bg = src_gray.copy()
  #src_bg[src_mask!=0]=0

  #cv2.imshow("SRC BG",src_bg)

  #print(src_bg.shape)
  #print(dst_mask.shape)
  #temp = src_bg + dst_mask

  #cv2.imshow("WARPED FACE",te

  cv2.waitKey(0)
  #cv2.destroyAllWindows()

  #for (x, y) in hull:
  #  cv2.circle(srcImage, (x, y), 1, (0, 0, 255), -1)

  #Image.fromarray(srcImage).show()
  #Image.fromarray(targetImage).show()

  #detector = dlib.get_frontal_face_detector()
  #predictor = dlib.shape_predictor(PATH)

  #src_gray = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
  #rects = detector(src_gray, 1)
  #rect = rects[0]
  #print(rects)

  #shape = predictor(src_gray, rect)
  #shape = face_utils.shape_to_np(shape)
  #print(shape)

  # (x, y, w, h) = face_utils.rect_to_bb(rect)
  # cv2.rectangle(srcImage, (x, y), (x + w, y + h), (0, 255, 0), 2)

  # hullIndex = cv2.convexHull(shape, returnPoints = False)
  # hull =[]

  # for i in range(0, len(hullIndex)):
  #     hull.append(shape[int(hullIndex[i])])

  # print(hull)

  #mask = np.zeros(src_gray.shape, dtype = src_gray.dtype)  
    
  #cv2.fillConvexPoly(mask, np.int32(hull), (255,255))

  #print(mask.shape)

if __name__ == "__main__":
  main()
