#coding=utf-8
# cython:language_level=3
import cv2,numpy as np,time
from core import cv2ex
import kit.TpsDeformTorch as TpsDeformTorch 


class  FaceBeautify:

    @staticmethod
    def White_Dermabration_Face(img,marks=None,white_factor=0,smooth_factor=0):
          
        res_img=img
        if white_factor>0: 
            white_mask=np.zeros_like(img)
            white_area_idx=np.concatenate([marks[0:17],marks[26:21:-1],marks[21:16:-1],marks[0:1]]).astype(np.int32)
            cv2.fillPoly(white_mask,[white_area_idx],color=(1, 1, 1))
            res_img=img+(white_factor/100)*white_mask
        if smooth_factor>0: 
            smooth_mask=np.zeros_like(img)
            smooth_mask_idx=np.concatenate([marks[0:17],marks[45:48],marks[35:34:-1],marks[54:61],marks[31:21],marks[39:42],marks[0:1]]).astype(np.int32)
            cv2.fillPoly(smooth_mask,[smooth_mask_idx],color=(1, 1, 1))
            dx=max(int(smooth_factor//10),8)
            fc=min(smooth_factor,50)
            fs=min(smooth_factor,75)
            blur_img = cv2.bilateralFilter(img, dx, fc, fs)
            onef=np.float(1.0)
            res_img=img*(onef-smooth_mask)+blur_img*smooth_mask
            #img = cv2.addWeighted(img, 0.2, blur_img, 0.8, 0)
        return res_img

    @staticmethod
    def DeformFaceImage_TpsDeformTorch(img,landmarks,malar_thin=0,jaw_thin=0,cheek_thin=0,small_mouth=0,long_face=0,
                                       eye_distance=0,low_cheek_thin=0,show_landmarks=False):
        
        pi,qi=FaceBeautify.getControlPoints(landmarks,malar_thin=malar_thin,jaw_thin=jaw_thin,cheek_thin=cheek_thin,low_cheek_thin=low_cheek_thin,
                               small_mouth=small_mouth,long_face=long_face,eye_distance=eye_distance)
        if len(pi)==0:
            new_img=img
        else: 
            h,w,c=img.shape
            minx,miny=np.min(landmarks,axis=0)*0.9
            maxx,maxy=np.max(landmarks,axis=0)*1.1
            pi=np.append(pi,[[minx,miny],[minx,maxy],[maxx,miny],[maxx,maxy]],axis=0)
            qi=np.append(qi,[[minx,miny],[minx,maxy],[maxx,miny],[maxx,maxy]],axis=0)
            pi=np.append(pi,[[20,20],[h-5,20],[0,w-20],[h-5,w-20]],axis=0)
            qi=np.append(qi,[[20,20],[h-5,20],[0,w-20],[h-5,w-20]],axis=0)
            
            time1=time.time()
                       
            mapxy=TpsDeformTorch.get_image_mapxy(img,qi,pi)
            new_img = cv2.remap(img, mapxy[:, :, 0], mapxy[:, :, 1], borderMode=cv2.BORDER_REPLICATE, interpolation=cv2.INTER_LINEAR)

            if show_landmarks:
                for x,y in pi:
                    cv2.circle(new_img,(int(x),int(y)),2,(0,255,0),-1)
                for x,y in qi:
                    cv2.circle(new_img,(int(x),int(y)),2,(0,0,255),-1)
            time2=time.time()
            #sd.FaceBeautyRemapTime=round((time2-time1)*1000)
        return new_img
     


 


    @staticmethod
    def DeformFaceImage_LtwTorch(img,landmarks,malar_thin=0,jaw_thin=0,cheek_thin=0,small_mouth=0,long_face=0):
        import kit.LtwDeformTorch as LtwDeformTorch

        pi,qi,r=FaceBeautify.getLtwPoints(landmarks,malar_thin=malar_thin,jaw_thin=jaw_thin,cheek_thin=cheek_thin,
                            small_mouth=small_mouth,long_face=long_face)
        mapxy=LtwDeformTorch.get_image_mapxy(img,pi,qi,r)
        new_img = cv2.remap(img, mapxy[:, :, 0], mapxy[:, :, 1], borderMode=cv2.BORDER_REPLICATE, interpolation=cv2.INTER_LINEAR)
        draw_point=False
        if draw_point:
            for x,y in pi:
                cv2.circle(new_img,(int(x),int(y)),2,(0,255,0),-1)
            for x,y in qi:
                cv2.circle(new_img,(int(x),int(y)),2,(0,0,255),-1)
        return new_img

    @staticmethod
    def white_face(img,ldmrks,white_weight=1.0):
       pass

    @staticmethod
    def getControlPoints(landmarks,malar_thin=0,jaw_thin=0,cheek_thin=0,low_cheek_thin=0,small_mouth=0,long_face=0,eye_distance=0):
        if landmarks is None:
            return None,None
        pi=landmarks.copy().astype(int)
        qi=landmarks.copy().astype(int)
     
        
        if malar_thin!=0 or cheek_thin!=0 or jaw_thin!=0 or small_mouth!=0 or eye_distance!=0 or low_cheek_thin!=0:  
            #---颧骨
            pts=[0,16,1,15]
            for idx in pts:
                distance=landmarks[27]-pi[idx]
                nPixel=(malar_thin*distance*0.5).astype(int)
                qi[idx]+=nPixel 
            
                #---面颊        
            pts=[2,14,3,13]
            for idx in pts:
                distance=landmarks[62]-pi[idx]
                nPixel=(cheek_thin*distance*0.5).astype(int)
                qi[idx]+=nPixel 

                #---下颌角        
            pts=[5,11,4,12]
            for idx in pts:
                distance=landmarks[30]-pi[idx]
                nPixel=(low_cheek_thin*distance*0.5).astype(int)
                qi[idx]+=nPixel 
     
            #---下巴        
            pts=[6,10]
            for idx in pts:
                distance=landmarks[66]-pi[idx]
                nPixel=(jaw_thin*distance).astype(int)
                qi[idx]+=nPixel 
                qi[8]+=[0,1]
            #小嘴
         
            pts=[48,54]
            for idx in pts:
                distance=landmarks[62]-pi[idx]
                nPixel=(small_mouth*distance).astype(int)
                qi[idx]+=nPixel
                
            #眼距
         
            pts=[39,42]
            for idx in pts:
                distance=(pi[idx]-landmarks[27])
                nPixel=(eye_distance*distance).astype(int)
                qi[idx]+=nPixel 
                 

        #---长短脸
        if long_face!=0:  
            pts=[8]
            for idx in pts:
                distance=pi[idx][1]-landmarks[57][1]
                nPixel=(long_face*distance).astype(int)
                qi[idx][1]+=nPixel 

        deform_idx=[]
        for idx in range(len(pi)):
            if pi[idx][0]!=qi[idx][0] or pi[idx][1]!=qi[idx][1]:
                deform_idx.append(idx)

        return pi[deform_idx],qi[deform_idx]
