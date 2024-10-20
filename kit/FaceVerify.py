#coding=utf-8
# cython:language_level=3
from kit import ToolKit
import numpy as np
import cv2;
from facelib.arcface import ArcFace

mIdentityImage=None;



def pad_image_for_arcface_input(current_img,target_size=(112,112)):
    factor_0 = target_size[0] / current_img.shape[0]
    factor_1 = target_size[1] / current_img.shape[1]
    factor = min(factor_0, factor_1)
    dsize = (int(current_img.shape[1] * factor), int(current_img.shape[0] * factor))
    current_img = cv2.resize(current_img, dsize)

    diff_0 = target_size[0] - current_img.shape[0]
    diff_1 = target_size[1] - current_img.shape[1]

    current_img = np.pad( current_img,
                        ((diff_0 // 2, diff_0 - diff_0 // 2),(diff_1 // 2, diff_1 - diff_1 // 2), (0, 0),),
                        "constant",
                    )
    if current_img.shape[0:2] != target_size:
                current_img = cv2.resize(current_img, target_size)

    return current_img
    #img_pixels = image.img_to_array(current_img)  # what this line doing? must?
    ##img_pixels = np.expand_dims(img_pixels, axis=0)
    #img_pixels /= 255 

    #return img_pixels

def setIdentityImage(image):
    pass

def VerifyImageFaceSimilarity(image1,image2):
    if image1 is None or image2 is None:
        print("错误，图片为空")
        return 100.0;
    dist=ArcFace.getImageFaceSimilarity(image1,image2)
    return dist;

def VerifyImageFile(path1,path2):
    pass