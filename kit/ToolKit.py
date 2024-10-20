#coding=utf-8
# cython:language_level=3

import numpy as np
from core.xlib.face import FRect,FLandmarks2D
from core.xlib.face.ELandmarks2D import ELandmarks2D
from PyQt5.QtGui import QPixmap,QImage;
from PyQt5.QtCore import *;
from PyQt5.QtWidgets import QFileDialog,QMessageBox;
import cv2,numpy,time;
from core.DFLIMG import DFLIMG,DFLJPG
from facelib import FANExtractor,LandmarksProcessor,FaceType
import pickle,os; 
from core.leras import nn
from core.xlib.image import ImageProcessor
import kit.ShareData as sd;
from kit.FaceEngines import FaceEngines
from facelib.facenet.facenet import facenet
import threading

last_time_mark=0;
time_start_mark=0;
def setTimeStart():
    global last_time_mark,time_start_mark;
    last_time_mark=time.time();
    time_start_mark=time.time(); 

def getUseTimeSinceLast():
    global last_time_mark;
    time_now=time.time();
    time_use=time_now-last_time_mark;
    last_time_mark=time_now;
    return time_use;

def ShowWarningError(parent,title,content,mode="gui"):
    if(mode=="gui"):
        QMessageBox.information(None,title,content);
    else:
        print(title,":",content)
         
def PreLoadEngines(yolo=False,face_insght=False,xseg=False):    
    th=threading.Thread(target=LoadEngineModelThreadRun,args=(yolo,face_insght,xseg))
    th.daemo=True;
    th.start();

def LoadEngineModelThreadRun(yolo=False,face_insght=False,xseg=False):
    if yolo:
        YoloEngine=FaceEngines.getYoloEngineOnnx() 
    if face_insght:
        FaceInsightEngine=FaceEngines.getFaceInsightEngineOnnx()
    if xseg:
        FaceEngines.getXSegModelEngineTF(sd.xseg_model_dir)

def EnsureFaceDetectModelLoaded():
    
    #S3fdFaceEngine=FaceEngines.getS3fdFaceEngineOnnx()
    #CenterFaceEngine=FaceEngines.getCenterFaceEngineOnnx()
    FaceEngines.getXSegModelEngineTF(sd.xseg_model_dir)
    YoloEngine=FaceEngines.getYoloEngineOnnx() 
    FaceInsightEngine=FaceEngines.getFaceInsightEngineOnnx()  
    pass

def GetDetectAndAlignFace(raw_frame_image,image_size=256,filter_face_size=50,face_coverage=2.2,window_width=0,detector="yolo"):
    
    EnsureFaceDetectModelLoaded()
    extract_face_raw_images=[] ;   extract_face_aligned_images=[];    
    extract_face_landmarks_list=[]  ;  extract_face_angles_list=[]
    h,w,c=raw_frame_image.shape
    frame_image=cv2.resize(raw_frame_image,(w//2,h//2))

    if detector=='s3fd':
        face_rects = FaceEngines.getS3fdFaceEngineOnnx().extract (frame_image,fixed_window=window_width)[0]
    elif detector=='centerface':
        face_rects = FaceEngines.getCenterFaceEngineOnnx().extract (frame_image, fixed_window=window_width,threshold=0.5)[0]
    else:
        face_rects = FaceEngines.getYoloEngineOnnx().extract (frame_image, fixed_window=window_width,threshold=0.5)[0]

    for i,(l,t,r,b) in enumerate(face_rects):
        width=r-l;
        if width< filter_face_size/2:
            #face_rects.pop(i)
            continue;
        H,W,C=frame_image.shape
        l,t,r,b=l/W,t/H,r/W,b/H
        face_uniform_rect=FRect.from_ltrb((l,t,r,b))

        detect_rect_face, face_uni_mat=face_uniform_rect.cut(raw_frame_image,1.4,192)

        #--提取特征点 
        uni_lmrks,face_lmrks=FaceEngines.getFaceInsightEngineOnnx().extract_lmks_68(detect_rect_face) 
        #--LandmarksProcessor.draw_landmarks(detect_rect_face,face_lmrks)

        #----根据特征点 旋转对齐截取人脸
        face_ulmrks=FLandmarks2D.create(ELandmarks2D.L68,uni_lmrks)
        #face_ulmrks.draw(detect_rect_face,(0,255,0))
        face_ulmrks_inSrc = face_ulmrks.transform(face_uni_mat, invert=True)
        aligned_face_img, SrcImg_to_AlignFace_Uni_mat,_ = face_ulmrks_inSrc.cut(raw_frame_image,coverage=face_coverage,output_size=image_size)

        align_uni_lmrks,align_lmrks=FaceEngines.getFaceInsightEngineOnnx().extract_lmks_68(aligned_face_img)
        align_face_ulmrks=FLandmarks2D.create(ELandmarks2D.L68,align_uni_lmrks)
        #align_face_ulmrks.draw(aligned_face_img,(0,255,0))

        face_angles=LandmarksProcessor.estimate_pitch_yaw_roll_by_degree(face_lmrks)
        extract_face_raw_images.append(detect_rect_face)
        extract_face_aligned_images.append(aligned_face_img)
        extract_face_landmarks_list.append(align_lmrks)
        extract_face_angles_list.append(list(face_angles))
        #cv2.imshow("aligned_face_img",aligned_face_img)
        #cv2.imshow("detect_rect_face",detect_rect_face)
        #cv2.waitKey(10)

    return extract_face_raw_images,extract_face_aligned_images,extract_face_landmarks_list,extract_face_angles_list

def GetXsegMaskForFaceImage(dflimg):
    face_img=dflimg.get_img()
    xseg_res = FaceEngines.getXSegModelEngineTF(sd.xseg_model_dir).get_resolution()
    h,w,c = face_img.shape
    if w != xseg_res:
            face_img = cv2.resize( face_img, (xseg_res,xseg_res), interpolation=cv2.INTER_LANCZOS4 ) 
    try:
        if len(face_img.shape) == 2: face_img = face_img[...,None]
        mask = FaceEngines.getXSegModelEngineTF(sd.xseg_model_dir).extract(face_img)
        mask[mask < 0.5]=0
        mask[mask >= 0.5]=1
        dflimg.set_xseg_mask(mask)
        #dflimg.save()
    except Exception as Ex:
        print("!!! GetXsegMaskForFaceImage Error: %s" %  Ex)

def GetXsegMask(face_img):
    xseg_res = FaceEngines.getXSegModelEngineTF(sd.xseg_model_dir).get_resolution()
    h,w,c = face_img.shape
    if w != xseg_res:
            face_img = cv2.resize( face_img, (xseg_res,xseg_res), interpolation=cv2.INTER_LANCZOS4 ) 
    mask = FaceEngines.getXSegModelEngineTF(sd.xseg_model_dir).extract(face_img)
    mask[mask < 0.5]=0
    mask[mask >= 0.5]=1
    return mask

def SaveAsDflImage(face_image,output_filepath,face_type_str,face_image_landmarks,src_image_landmarks,face_angles=None,write_xseg_mask=True):
    SaveImageWithUniCode(face_image,output_filepath)
    dflimg = DFLJPG.load(output_filepath)
    dflimg.set_face_type(face_type_str)

    dict=dflimg.get_dict();
    if face_image_landmarks is not None:
        landmarks=face_image_landmarks.tolist()
        dflimg.set_landmarks(face_image_landmarks.tolist())
        if face_angles is None:
            pitch, yaw, roll=LandmarksProcessor.estimate_pitch_yaw_roll_by_degree(landmarks)
            #print("recalculate pitch, yaw, roll:",pitch, yaw, roll)
            dict["pitch"]=pitch
            dict["yaw"]=yaw
            dict["roll"]=roll
        if face_angles is not None:
            if(len(face_angles)==3):
                dict["pitch"]=face_angles[0]
                dict["yaw"]=face_angles[1]
                dict["roll"]=face_angles[2]
    if src_image_landmarks is not None:
       dflimg.set_source_landmarks(src_image_landmarks.tolist())
    #dflimg.set_image_to_face_mat(image_to_face_mat)
    if write_xseg_mask is True:
        GetXsegMaskForFaceImage(dflimg)
    dflimg.save()

#--Method: 向dfl图片写入 人脸角度
def WriteFaceAngleForDflImageFile(img_full_path,NoRecalc=False):
    succeed=False;tips=""
    if os.path.exists(img_full_path) is False:
        return False,"file path not exists"
    dflimg = DFLJPG.load(img_full_path);
    if dflimg is None:
        return False,"read dfl failed"
    if dflimg.has_data()==False:
        return False,"dfl has no data"
    dict=dflimg.get_dict()
    if ("pitch" in dict) and ("yaw" in dict) and NoRecalc:
        return False,"already has pitch/yaw angles"
    if "landmarks" not in dict:
        return False,"dfl has no landmarks"
    landmarks=dflimg.get_landmarks()
    pitch, yaw, roll=LandmarksProcessor.estimate_pitch_yaw_roll_by_degree(landmarks)
    dict["pitch"]=pitch
    dict["yaw"]=yaw
    dict["roll"]=roll
    dflimg.save()
    return True,f"angles pitch:{pitch},yaw:{yaw},roll:{roll} write succed"
    return current_img


#----函数：控件中显示图片
def ShowImageInLabel(img,label,bgr_cvt=True,scale=True):
        if(img is None):
            return;
        h,w,c=img.shape
        hh=(h//4)*4 ; ww=(w//4)*4;
        width,height = label.width(),label.height();
        if img.dtype==np.float32:
            img=img*255;
        img_src = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.uint8)
        
        try:
            qImage = QImage(img_src, img_src.shape[1], img_src.shape[0], QImage.Format_RGB888)
            qImage=qImage.copy(0,0,ww,hh)
            qPixmap = QPixmap.fromImage(qImage).scaled(width,height, Qt.KeepAspectRatio);
            label.setPixmap(qPixmap)
        except :
            pass               
 
            
        

def LoadImageWithChineseCode(file_path):
    cv_img = cv2.imdecode(numpy.fromfile(file_path,dtype=numpy.uint8),-1)
    return cv_img

def SaveImageWithUniCode(img,file_path):
    cv2.imencode('.jpg', img )[1].tofile(file_path)

mCamVideoCapture=None;
mVideoFile=None;
def LoadVideoFile(video_file):
    global mCamVideoCapture,mVideoFile;
    mCamVideoCapture=cv2.VideoCapture(video_file);
    #mCamVideoCapture.set(cv2.CAP_PROP_CUDA_DECODER,True)
    if mCamVideoCapture is None:
        print("load video file failed:",video_file)
        return
    mVideoFile=video_file


def GetVideoFrameImage(video_file,frame_num):
    global mCamVideoCapture,mVideoFile;
    start=time.time()
    if(video_file!=mVideoFile):
        LoadVideoFile(video_file);
    frame_count=int(mCamVideoCapture.get(7))
    curr_frame=int(mCamVideoCapture.get(1))
    frame_gap=frame_num-curr_frame;
    if frame_gap>1 :
        for i in range(frame_gap-1):
            mCamVideoCapture.grab()
    else:
        mCamVideoCapture.set(1,frame_num)       
    set_time_end=time.time()
    ok,frame_image=mCamVideoCapture.read()
    read_time_end=time.time()
    set_time_use=round(set_time_end-start,3)
    read_time_use=round(read_time_end-set_time_end,3)
    whole_time_use=round((read_time_end-start)*1000)    
    #print(f"[P]读取总用时{whole_time_use},{set_time_use}+{read_time_use},帧画面大小{frame_image.shape[1]}x{frame_image.shape[0]}")
    sd.VideoFrameReadTime=whole_time_use;
    sd.VideoCurrFrameIndex=frame_num;
    sd.ExtractFrameImage=frame_image
    sd.VideoFrameCount=frame_count;
    return ok,frame_image,frame_count;

def GetVideoFrameImageByRead(video_file,frame_num):
    global mCamVideoCapture,mVideoFile;
    if(video_file!=mVideoFile):
        LoadVideoFile(video_file);
    frame_count=int(mCamVideoCapture.get(7))
    curr_frame=int(mCamVideoCapture.get(1))
    mCamVideoCapture.set(1,frame_num)
    ok,frame_image=mCamVideoCapture.read()
    sd.VideoCurrFrameIndex=frame_num;
    sd.ExtractFrameImage=frame_image
    sd.VideoFrameCount=frame_count;
    return ok,frame_image,frame_count;

def GetVideoFrameCount(video_file):
    global mCamVideoCapture,mVideoFile;
    if(video_file!=mVideoFile):
        LoadVideoFile(video_file);
    frame_count=int(mCamVideoCapture.get(7))
    return frame_count


def SavedConfigData(name,value):
    cfg_file=f"live.cfg";
    dict={}
    if os.path.exists(cfg_file)== True:
        #print("cfg exists,read exist config dict");  
        f_read=open(cfg_file,"rb")
        dict=pickle.load(f_read)
        f_read.close()
    dict[name]=value; 
    f_write=open(cfg_file,"wb")
    bin_data=pickle.dump(dict,f_write)
    f_write.close()

def SavedConfigDataDict(options,cfg_file="live.cfg"):    
    dict={}
    if os.path.exists(cfg_file)== True:
        #print("cfg exists,read exist config dict");  
        f_read=open(cfg_file,"rb")
        dict=pickle.load(f_read)
        f_read.close()
    for k in options.keys():
        dict[k]=options[k];  
    f_write=open(cfg_file,"wb")
    bin_data=pickle.dump(dict,f_write)
    f_write.close()

def GetSavedConfigData(key_name=None,cfg_file="live.cfg"):
    if(os.path.exists(cfg_file)==False):
        print(f"{cfg_file} not exists");
        return None;
    f=open(cfg_file,"rb")
    cfg_dict=pickle.load(f)
    f.close()
    if cfg_dict is None:
        print("load cfg file ,content is None");
        return None;
    if key_name is None:
        #print(cfg_dict)
        return cfg_dict
    else:
        return cfg_dict.get(key_name,None)

def LiveDetectAndAlignFace():
        
        detect_start_time_tick=time.time()
        #----提取面部范围(centerface)---
        if sd.mRawFrameImageFloat is None: 
           time.sleep(0.02);
           return
        frame_image_src_int=sd.mRawFrameImageInt
        frame_image_src_float=sd.mRawFrameImageFloat;
        H,W,C=frame_image_src_int.shape
        

        rects = FaceEngines.getYoloEngineOnnx().extract (frame_image_src_int, threshold=sd.DetectThreshold,fixed_window=sd.DetectWindowSize)[0]
        #rects = FaceEngines.getYoloEngineOnnx().extract (frame_image_src_int, threshold=sd.DetectThreshold)[0]
        if  len(rects)==0:
            sd.ExistTargetFace=False;
            #return;
        if len(rects)>=1:
            for (l,t,r,b) in rects:
                ul,ut,ur,ub=l/W,t/H,r/W,b/H
                face_urect=FRect.from_ltrb((ul,ut,ur,ub))
                face_rect_img, face_uni_mat=face_urect.cut(frame_image_src_int,1.4,192) 
                verify_ok,v=facenet.getVerifyResult(face_rect_img,threshold=sd.VerifyThreshold)
                if verify_ok is False:
                    sd.ExistTargetFace=False;
                    continue;
                if verify_ok is True:
                    sd.ExistTargetFace=True;
                    sd.DetectFaceRect=(int(l),int(t),int(r),int(b))
                    break


        if sd.ExistTargetFace==False:            
            #self.mSwapEndUpdateSignal.emit()
            return;

        detect_end_time_tick=time.time()
        sd.DetectFaceTime=(detect_end_time_tick-detect_start_time_tick)*1000


        #---- FaceMesh 从正方形脸提取106-转68特征点
        uni_lmrks,lmrks=FaceEngines.getFaceInsightEngineOnnx().extract_lmks_68(face_rect_img)
        sd.face_ulmrks=FLandmarks2D.create(ELandmarks2D.L68,uni_lmrks)
        fh,fw,fc=face_rect_img.shape
        sd.face_lmks_coords=np.multiply(sd.face_ulmrks._ulmrks,[fw,fh]).astype(np.int32)

        #=== 获取在原帧图中的特征点坐标（绝对坐标值）
        sd.face_ulmrks_inSrc = sd.face_ulmrks.transform(face_uni_mat, invert=True)
        ih,iw,c=frame_image_src_float.shape
        sd.ldmrkCoordsSrc=np.multiply(sd.face_ulmrks_inSrc._ulmrks,[iw,ih]).astype(np.int32)
        
        #---- 特征点标记图片绘制
        #sd.mMarkFaceImage=face_rect_img.copy();
        #sd.face_ulmrks.draw(sd.mMarkFaceImage,(0,255,255),1)
        #face_urect.draw(sd.mMarkFaceImage ,(0,0,255),2)
        #cv2.imshow("marker",sd.mMarkFaceImage)
                
        sd.ExistTargetFace=True
        

def OpenXsegEditor(folder):
    pythonw_path=os.path.dirname(os.getcwd())+"/python/pythonw.exe"
    script_path=f"{os.getcwd()}/start.py"
    open_xseg_param=f"{script_path} -app xseg_editor -path {folder}"
    if os.path.exists(pythonw_path) is False:
        QMessageBox.information(None,"Error","没找到pythonw文件");
        return;
    import win32api
    win32api.ShellExecute(0,'open',pythonw_path,open_xseg_param,'',True) 
            



if __name__=="__main__":
    pass