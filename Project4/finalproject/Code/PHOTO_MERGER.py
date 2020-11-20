
import cv2
import numpy as np
from scipy.spatial import Delaunay
import finalproject.Code.Utils as FS_Utils

class PHOTO_MERGER():

    @staticmethod
    def transformTriangles(Image1, Image2, points1, points2):

        region1 = list(cv2.boundingRect(points1))
        region2 = list(cv2.boundingRect(points2))


        points1Norm = [((points1[i][0] - region1[0]),(points1[i][1] - region1[1])) for i in range(0,3)]
        points2Norm = [((points2[i][0] - region2[0]),(points2[i][1] - region2[1])) for i in range(0,3)]

        Im2Mask = np.zeros((region2[1] + region2[3] - region2[1], region2[0] + region2[2] - region2[0], 3), dtype = np.float32)

        cv2.fillConvexPoly(Im2Mask, np.int32(points2Norm), (1.0, 1.0, 1.0), 16, 0)

        InverseMask = 1.0 - Im2Mask

        InverseMask = InverseMask.astype("uint8")


        ImageRegion1 = Image1[region1[1]:region1[1] + region1[3], region1[0]:region1[0] + region1[2]]
        affineTransformation = cv2.getAffineTransform(np.float32(points1Norm), np.float32(points2Norm))

        ImageRegion2 = cv2.warpAffine( ImageRegion1, affineTransformation, (region2[2], region2[3]), None, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        ImageRegion2 = ImageRegion2 * Im2Mask

        part = Image2[region2[1]:region2[1] + region2[3], region2[0]:region2[0] + region2[2], :]

        Image2[region2[1]:region2[1] + region2[3], region2[0]:region2[0] + region2[2], :] =\
            Image2[region2[1]:region2[1] + region2[3], region2[0]:region2[0] + region2[2], :] * InverseMask[0:part.shape[0],0:part.shape[1],:]



        Image2[region2[1]:region2[1] + region2[3], region2[0]:region2[0] + region2[2], :] =\
            Image2[region2[1]:region2[1] + region2[3], region2[0]:region2[0] + region2[2],:] + ImageRegion2[0:part.shape[0],0:part.shape[1],:]



    def getConvexHull(self, points):

        hullIndex = cv2.convexHull(points, returnPoints = False)
        hull =[]
        for i in range(0, len(hullIndex)):
            hull.append(points[int(hullIndex[i])])

        return hull


    def getFaceMask(self, points, shape, params):

        if params["FACE_MASK_TYPE"] == "FULL":
            return FS_Utils.getFullFaceMask(shape, self.getConvexHull(points))

        elif params["FACE_MASK_TYPE"] == "PARTIAL":
            return FS_Utils.getPartialMask(points, shape)

    def maskUpdate(self, mouthMask):
        mouthMask[mouthMask > 0 ] = 2
        mouthMask[mouthMask == 0] = 1
        mouthMask[mouthMask == 2] = 0
        return mouthMask


    def getMouthMask(self, points, shape, params):
        mask =  FS_Utils.getFullFaceMask(shape, self.getConvexHull(points[60:68]))
        return self.maskUpdate(mask)


    def getLefteyeMask(self, points, shape, params):
        mask =  FS_Utils.getFullFaceMask(shape, self.getConvexHull(points[35:41]))
        return self.maskUpdate(mask)

    def getRighteyeMask(self, points, shape, params):
        mask =  FS_Utils.getFullFaceMask(shape, self.getConvexHull(points[41:47]))
        return self.maskUpdate(mask)



    def blendImage(self, sourceImage, morphedImage,  triangles, src_points, sourceMask):


        dImage = sourceImage.copy()

        sourceMask[sourceMask != 255] = 0
        sourceMask[sourceMask == 255] = 1

        dMask = np.zeros((sourceMask.shape[0], sourceMask.shape[1], 3))
        dMask[:,:,0] = sourceMask
        dMask[:,:,1] = sourceMask
        dMask[:,:,2] = sourceMask

        sourceMask = dMask

        for tri in triangles:

            mask = np.zeros(sourceImage.shape)
            points = src_points[tri]

            mask = cv2.fillConvexPoly(mask, points, (255,255,255))
            mask[mask == 0] = 1.0

            center = (int(morphedImage.shape[1]/2), int(morphedImage.shape[0]/2))

            output = cv2.seamlessClone(morphedImage, sourceImage.copy(), np.uint8(mask[:,:,1]), center, cv2.MIXED_CLONE)

            mask[mask!=255] = 0
            mask[mask == 255] = 1
            dImage = dImage * (1 - mask) + output * mask

        output = dImage * sourceMask + sourceImage * (1 - sourceMask)
        return output


    '''
    ## source_image = Image1
    ## target_image = Image2
    '''

    def morphImages(self, Image1, Image2, Im1LandMarks, Im2LandMarks, params):

        src_gray = cv2.cvtColor(Image1, cv2.COLOR_BGR2GRAY)
        src_face_points = Im1LandMarks

        target_gray = cv2.cvtColor(Image2, cv2.COLOR_BGR2GRAY)
        target_face_points = Im2LandMarks

        src_pts_appended = np.append(src_face_points,np.array([[0,0],[0,src_gray.shape[0]-1],[src_gray.shape[1]-1,0], [src_gray.shape[1]-1,src_gray.shape[0]-1]]), axis=0)
        target_pts_appended = np.append(target_face_points,np.array([[0,0],[0,target_gray.shape[0]-1],[target_gray.shape[1]-1,0], [target_gray.shape[1]-1,target_gray.shape[0]-1]]), axis=0)

        #src_pts_appended = src_face_points
        #target_pts_appended = target_face_points

        src_tri = Delaunay(src_pts_appended)
        warped_img = Image1.copy()

        for tri in src_tri.simplices:
            PHOTO_MERGER.transformTriangles(Image2, warped_img, target_pts_appended[tri], src_pts_appended[tri])


        src_mask = self.getFaceMask(src_face_points,src_gray.shape, params)
        mouthMask = self.getMouthMask(src_face_points,src_gray.shape, params)
        leftMask = self.getLefteyeMask(src_face_points,src_gray.shape, params)
        rightMask = self.getRighteyeMask(src_face_points,src_gray.shape, params)

        src_mask = src_mask * mouthMask
        #src_mask = src_mask * leftMask
        #src_mask = src_mask * rightMask



        warped_img[src_mask == 0] = 0
        src_mask[src_mask == 0] = 1

        #cv2.imshow("warped",warped_img)

        center = (int(warped_img.shape[1]/2), int(warped_img.shape[0]/2))

        output = cv2.seamlessClone(warped_img, Image1.copy(), np.uint8(src_mask), center, cv2.NORMAL_CLONE)
        #output = self.blender.blendImage(warped_img,Image1.copy(), np.uint8(src_mask))

        #output = self.blendImage(Image1.copy(), warped_img.copy(), src_tri.simplices, src_pts_appended, src_mask)
        return output.astype("uint8")


