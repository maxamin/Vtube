#! /usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt
from core.DFLIMG.DFLJPG import DFLJPG
from PIL import Image
import os,cv2,random

try:
    import torch    # Install PyTorch first: https://pytorch.org/get-started/locally/
    from test.mls_deform_pt import (
        mls_affine_deformation as mls_affine_deformation_pt,
        mls_similarity_deformation as mls_similarity_deformation_pt,
        mls_rigid_deformation as mls_rigid_deformation_pt,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("device:",device)
except ImportError as e:
    print(e)
     
def getControlPoints(landmarks,narrow_face=0,jaw_thin=0,small_mouth=0,long_face=0):
    if landmarks is None:
        return None,None
    pi=landmarks.copy().astype(int)
    qi=landmarks.copy().astype(int)
     
    #---窄脸
    if narrow_face!=0:  
        pts=[1,15]
        for idx in pts:
            distance=landmarks[30][0]-pi[idx][0]
            nPixel=(narrow_face*distance).astype(int)
            qi[idx][0]+=nPixel 

    #---长短脸
    if long_face!=0:  
        pts=[8]
        for idx in pts:
            distance=pi[idx][1]-landmarks[30][1]
            nPixel=(long_face*distance).astype(int)
            qi[idx][1]+=nPixel 

    #---下巴
    if jaw_thin!=0:
        pts=[5,11]
        for idx in pts:
            distance=landmarks[33]-pi[idx]
            nPixel=(jaw_thin*distance).astype(int)
            qi[idx]+=nPixel 
    #小嘴
    if small_mouth!=0:
        pts=[48,54]
        for idx in pts:
            distance=landmarks[62]-pi[idx]
            nPixel=(small_mouth*distance).astype(int)
            qi[idx]+=nPixel 

    deform_idx=[]
    for idx in range(len(pi)):
        if pi[idx][0]!=qi[idx][0] or pi[idx][1]!=qi[idx][1]:
            deform_idx.append(idx)
    return pi[deform_idx],qi[deform_idx]



def demo():
    global device
    dir=r"F:\Ai_VideoImage\切脸\\"
    img_files=[file for file in os.listdir(dir) if ".jpg" in file] 
    for filename in img_files:
        filename=dir+filename
        print(filename)
        
        dfl=DFLJPG.load(filename)
        image=dfl.get_img()
        landmarks=dfl.get_landmarks()
        print("image shape:",image.shape)
        height, width, _ = image.shape

        gridX = torch.arange(width, dtype=torch.int16).to(device)
        gridY = torch.arange(height, dtype=torch.int16).to(device)
        vy, vx = torch.meshgrid(gridX, gridY)
        vy, vx = vy.transpose(0, 1), vx.transpose(0, 1)
    
        p,q=getControlPoints(landmarks,narrow_face=random.uniform(0.05,0.1),jaw_thin=random.uniform(0.1,0.15),
                               small_mouth=0,long_face=random.uniform(0.03,0.05))
        p=torch.from_numpy(p).to(device)
        q=torch.from_numpy(q).to(device)

        image = torch.from_numpy(image).to(device)
        time1=time.time()
        affine = mls_similarity_deformation_pt(vy, vx, p, q, alpha=1) 
        aug1 = torch.ones_like(image).to(device)
        aug1[vx.long(), vy.long()] = image[tuple(affine)]
        time2=time.time()
        print("pytorch mls_similarity_deformation_pt time:",time2-time1)

        
        cv2.imshow("img",image.cpu().numpy())
        cv2.waitKey(2000)
        time.sleep(2)


if __name__ == "__main__":
    #demo()
    height=12;width=15
    gridX = torch.arange(width, dtype=torch.int16).to(device)
    gridY = torch.arange(height, dtype=torch.int16).to(device)
    vy, vx = torch.meshgrid(gridX, gridY)
    vy, vx = vy.transpose(0, 1), vx.transpose(0, 1)
    vv=torch.cat((vx.reshape(1, height, width), vy.reshape(1, height, width)), dim=0)
    print("vv:",vv.shape,vv)
    print("vy:",vy.shape,vy)