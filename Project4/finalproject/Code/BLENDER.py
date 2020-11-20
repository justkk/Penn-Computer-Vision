import os
import pickle
import argparse

from chainer import cuda, serializers
from PIL import Image

from skimage import img_as_float
from skimage.io import imread, imsave

from finalproject.GP.model import EncoderDecoder, DCGAN_G

from finalproject.GP.gp_gan import gp_gan
import numpy as np

class GAN_BLENDER():

    def __init__(self, modelPath = "/Users/nikhilt/PycharmProjects/FS/finalproject/Gp", useSupervised=True):
        self.modelPath = modelPath
        parser = argparse.ArgumentParser(description='Gaussian-Poisson GAN for high-resolution image blending')
        parser.add_argument('--nef', type=int, default=64, help='# of base filters in encoder')
        parser.add_argument('--ngf', type=int, default=64, help='# of base filters in decoder or G')
        parser.add_argument('--nc',  type=int, default=3,  help='# of output channels in decoder or G')
        parser.add_argument('--nBottleneck',  type=int, default=4000, help='# of output channels in encoder')
        parser.add_argument('--ndf', type=int, default=64, help='# of base filters in D')

        parser.add_argument('--image_size', type=int, default=64, help='The height / width of the input image to network')

        parser.add_argument('--color_weight', type=float, default=1, help='Color weight')
        parser.add_argument('--sigma', type=float, default=0.5, help='Sigma for gaussian smooth of Gaussian-Poisson Equation')
        parser.add_argument('--gradient_kernel', type=str, default='normal', help='Kernel type for calc gradient')
        parser.add_argument('--smooth_sigma', type=float, default=1, help='Sigma for gaussian smooth of Laplacian pyramid')

        parser.add_argument('--supervised', type=lambda x:x == 'True', default=True, help='Use unsupervised Blending GAN if False')
        parser.add_argument('--nz',  type=int, default=100, help='Size of the latent z vector')
        parser.add_argument('--n_iteration', type=int, default=1000, help='# of iterations for optimizing z')

        parser.add_argument('--gpu', type=int, default=0, help='GPU ID (negative value indicates CPU)')
        parser.add_argument('--g_path', default='models/blending_gan.npz', help='Path for pretrained Blending GAN model')
        parser.add_argument('--unsupervised_path', default='models/unsupervised_blending_gan.npz', help='Path for pretrained unsupervised Blending GAN model')
        parser.add_argument('--list_path', default='', help='File for input list in csv format: obj_path;bg_path;mask_path in each line')
        parser.add_argument('--result_folder', default='blending_result', help='Name for folder storing results')

        parser.add_argument('--src_image', default='', help='Path for source image')
        parser.add_argument('--dst_image', default='', help='Path for destination image')
        parser.add_argument('--mask_image', default='', help='Path for mask image')
        parser.add_argument('--blended_image', default='', help='Where to save blended image')

        args = parser.parse_args()

        args.g_path = os.path.join(self.modelPath, args.g_path)
        args.unsupervised_path = os.path.join(self.modelPath, args.unsupervised_path)
        args.supervised = useSupervised
        args.gpu = -1


        if args.supervised:
            G = EncoderDecoder(args.nef, args.ngf, args.nc, args.nBottleneck, image_size=args.image_size)
            print('Load pretrained Blending GAN model from {} ...'.format(args.g_path))
            serializers.load_npz(args.g_path, G)
        else:
            G = DCGAN_G(args.image_size, args.nc, args.ngf)
            print('Load pretrained unsupervised Blending GAN model from {} ...'.format(args.unsupervised_path))
            serializers.load_npz(args.unsupervised_path, G)


        self.model = G
        self.args = args





    def blendImage(self, objImage, targetImage, maskImage):

        #objImage, targetImage, maskImage = objImage[:,:,::-1], targetImage[:,:,::-1], maskImage
        args = self.args
        G = self.model

        maskImage[maskImage!=255] = 0.0
        maskImage[maskImage==255] = 1.0

        objImage = img_as_float(objImage)
        targetImage = img_as_float(targetImage)
        maskImage = maskImage.astype(objImage.dtype)

        blended_im = gp_gan(objImage, targetImage, maskImage, G, args.image_size, args.gpu, color_weight=args.color_weight, sigma=args.sigma,
                                gradient_kernel=args.gradient_kernel, smooth_sigma=args.smooth_sigma, supervised=args.supervised,
                                nz=args.nz, n_iteration=args.n_iteration)

        return blended_im




#
# obj = img_as_float(np.asarray(Image.open("/Users/nikhilt/PycharmProjects/FS/finalproject/GP/images/test_images/src.jpg")))
# mask = np.asarray(Image.open("/Users/nikhilt/PycharmProjects/FS/finalproject/GP/images/test_images/mask.png") ).astype(obj.dtype)
# targetImage = img_as_float(np.asarray(Image.open("/Users/nikhilt/PycharmProjects/FS/finalproject/GP/images/test_images/dst.jpg")))
#
#
# g = GAN_BLENDER(modelPath="/Users/nikhilt/PycharmProjects/FS/finalproject/GP/")
# image = g.blendImage(obj, targetImage, mask)
# Image.fromarray(image).show()


