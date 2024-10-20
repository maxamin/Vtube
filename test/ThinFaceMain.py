import os,shutil,cv2,numpy as np,time
from urllib.request import urlopen
import json
from core import cv2ex
from kit.FaceEngines import FaceEngines
from core.xlib.face import FRect,FLandmarks2D,ELandmarks2D
from facelib.facenet import facenet
import kit.ShareData as sd
import threading
from core.xlib.math.Affine2DMat import Affine2DMat
from test.MlsFaceDeform import MlsFaceDeform
from kit.FaceDeformLtw import LtwFaceDeform

def MarkThread():
    #videocap=cv2.VideoCapture(r"F:\AvVideo\白云飞\大脸.mp4");
    #videocap=cv2.VideoCapture(r"F:\AvVideo\白云飞\nn.mp4");
    videocap=cv2.VideoCapture(0)
    while(True):
        ok,frame_image_src=videocap.read()
        H,W,C=frame_image_src.shape
        small_img=cv2.resize(frame_image_src,(W//2,H//2))
        rects = FaceEngines.getS3fdFaceEngineOnnx().extract (small_img, threshold=0.3)[0]
        if len(rects)>=1:
            l,t,r,b=rects[0]
            ul,ut,ur,ub=l/W,t/H,r/W,b/H
            face_urect=FRect.from_ltrb((2*ul,2*ut,2*ur,2*ub))
            face_square_rect_area_img, face_uni_mat=face_urect.cut(frame_image_src,1.4,192) 
            verify_ok,v,_=facenet.get_verify_result_from_standard(face_square_rect_area_img,threshold=1.0)

            uni_lmrks,lmrks=FaceEngines.getFaceInsightEngineOnnx().extract_lmks_68(face_square_rect_area_img)
            face_ulmrks=FLandmarks2D.create(ELandmarks2D.L68,uni_lmrks)
            face_ulmmrks_src=face_ulmrks.transform(face_uni_mat,invert=True)
            #face_ulmmrks_src.draw(frame_image_src,(0,255,255),1)
            h,w,c=frame_image_src.shape
            ldmrkCoords=np.multiply(face_ulmmrks_src._ulmrks,[w,h]).astype(np.int32)
            #print(f"frame_image_src.shape:{frame_image_src.shape}")
            
   
            
            #cv2.imshow("raw_frame",frame_image_src ) 

            method="mls-"
            if method=="mls":
            #---- 最小二乘法算法瘦脸-------------
                time1=time.time()
                deform_face=frame_image_src.copy()
                deform_face=thin_face_mls(deform_face,ldmrkCoords)
                preview=np.concatenate((frame_image_src,deform_face),1)
                time2=time.time()
                print(f"最小二乘法算法瘦脸耗时:{time2-time1}",end='\r')
                cv2.imshow("preview",preview) 
                cv2.waitKey(100)
            else:
            #---- 采用局部平移变形算法瘦脸-------------
                time1=time.time()
                #LtsWarpThin=LtwFaceDeform(frame_image_src)
                deform_face=LtwFaceDeform.thin_face(frame_image_src,ldmrkCoords)
                preview=np.concatenate((frame_image_src,deform_face),1)
                time2=time.time()
                print(f"局部平移变形算法瘦脸耗时:{time2-time1}",end='\r')
                cv2.imshow("preview",preview) 
                cv2.waitKey(100)
        else:
            print("No Face");

th=threading.Thread(target=MarkThread)   
th.start()

def thin_face_mls(frame_image_src,ldmrkCoords):
    
    np_data_thin=ldmrkCoords.copy()
    center=np_data_thin[30]

    minX,minY=np.min(ldmrkCoords,axis=0)
    maxX,maxY=np.max(ldmrkCoords,axis=0)
    head_img=frame_image_src[minY:maxY,:,:]

    pi=np.concatenate((ldmrkCoords[17],ldmrkCoords[4],ldmrkCoords[5],ldmrkCoords[6],
                       ldmrkCoords[10],ldmrkCoords[11],ldmrkCoords[12]),axis=0).reshape(-1,2)
    #pi=np.subtract(pi,[0,minY])

    qi=pi.copy()
    #print("qi.shape:",qi.shape)
    qi[1][0]=qi[1][0]+20
    qi[2][0]=qi[2][0]+20
    qi[3][0]=qi[3][0]+20

    qi[4][0]=qi[4][0]-20
    qi[5][0]=qi[5][0]-20
    qi[6][0]=qi[6][0]-20
     
    
    ddd = MlsFaceDeform(frame_image_src, pi)
    preview = ddd.deformation(frame_image_src, qi)
    #frame_image_src[minY:maxY,:,:]=preview
    return preview
    #cv2.imshow("preview",frame_image_src) 
    #cv2.waitKey(10)


def localTranslationWarpFastWithStrength(srcImg, startX, startY, endX, endY, radius, strength):
    rr = float(radius * radius)
    copyImg = np.zeros(srcImg.shape, np.uint8)
    copyImg = srcImg.copy()
  
  
    maskImg = np.zeros(srcImg.shape[:2], np.uint8)
    cv2.circle(maskImg, (startX, startY), int(radius), (255, 255, 255), -1)
  
    w = 100/strength
  
    # 计算公式中的|m-c|^2
    lx2 = (endX - startX) * (endX - startX)
    ly2 = (endY - startY) * (endY - startY)
    H, W, C = srcImg.shape
  
    mapX = np.vstack([np.arange(W).astype(np.float32).reshape(1, -1)] * H)
    mapY = np.hstack([np.arange(H).astype(np.float32).reshape(-1, 1)] * W)
  
    #每个点到起始点的距离
    dx2 = (mapX - startX) * (mapX - startX)
    dy2 = (mapY - startY) * (mapY - startY)
    distance = dx2 + dy2
    d_v2s = np.sqrt(distance)

    ratio_x = (rr - dx2) / (rr - dx2 + w * lx2)
    ratio_y = (rr - dy2) / (rr - dy2 + w * ly2)
    ratio_x = ratio_x * ratio_x
    ratio_y = ratio_y * ratio_y
  
    UX = mapX - ratio_x * (endX - startX) * (1 - d_v2s/radius)
    UY = mapY - ratio_y * (endY - startY) * (1 - d_v2s/radius)
  
    np.copyto(UX, mapX, where=maskImg == 0)
    np.copyto(UY, mapY, where=maskImg == 0)
    UX = UX.astype(np.float32)
    UY = UY.astype(np.float32)
    copyImg = cv2.remap(srcImg, UX, UY, interpolation=cv2.INTER_LINEAR)
  
    return copyImg
  

def thin_by_ilw(frame_image_src,ulmmrks_src):
    h,w,c=frame_image_src.shape
    coords=ulmmrks_src._ulmrks
    np_ldmk_data=coords*(w,h)
    np_ldmk_data=np_ldmk_data.astype(np.int32)
    center=np_ldmk_data[30]

    minx,miny=np.min(np_ldmk_data,axis=0)
    maxx,maxy=np.max(np_ldmk_data,axis=0)
    radius=(maxx-center[0])
    thin_face_image=localTranslationWarpFastWithStrength(frame_image_src,maxx,maxy,center[0],center[1],radius,80) 
    thin_face_image=localTranslationWarpFastWithStrength(thin_face_image,minx,maxy,center[0],center[1],radius,80) 
  
     

    return thin_face_image



 