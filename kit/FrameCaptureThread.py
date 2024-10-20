#coding=utf-8
# cython:language_level=3
import threading,cv2,time,queue
from PyQt5.QtWidgets import QFileDialog,QMessageBox;
from PyQt5.QtCore import QThread, pyqtSignal
import kit.ShareData as sd
import numpy as np;
from core.xlib.image import color_transfer as lib_ct
from core.xlib.image.ImageProcessor import ImageProcessor  
import kit.ToolKit as ToolKit
import kit.screen_cap as cap
from kit.FaceEngines import FaceEngines
from facelib.facenet.facenet import facenet
from core.xlib.face import FRect,FLandmarks2D,ELandmarks2D
from kit.MattingRVM import MattingRvmTorch

class FrameCaptureThread(QThread):
    """帧源获取线程"""

    mFrameCaptureSignal=pyqtSignal();
    MatEngine=MattingRvmTorch()
    mp_face_detection,face_detection=None,None
    mEventQueue=queue.Queue()

    def __init__(self):
        super(FrameCaptureThread, self).__init__()

        self.mCamVideoCapture=None;self.mFileVideoCapture=None;self.mThreadRun=True;   
        self.SrcType="camera";self.mVideoFilename=None;
        self.PlayToggle=True;self.mLastPlayTime=0.0; 
        self.ForcePlayFrameEnable=False; self.ForcePlayFrame=0;

        

    def run(self):
        while(self.mThreadRun):
            
            while(self.mEventQueue.empty() is False):
                #print("fetch queue")
                input=self.mEventQueue.get()
                op = input['op']
                if op=="set_cam_source":
                    #print("处理队列，更换摄像头捕捉设置")
                    self.setMediaSource(type="camera",input=input)
           
            frame_cap_start_time=time.time()
            #print(frame_cap_start_time,end='\r')

            if self.SrcType=="camera":
                if self.mCamVideoCapture is None:
                    continue
                try:
                    ok,frame_image=self.mCamVideoCapture.read()
                    if ok:
                        if sd.FrameRotateMode==1:
                            frame_image = cv2.rotate(frame_image,cv2.ROTATE_90_CLOCKWISE) 
                        elif sd.FrameRotateMode==2:
                            frame_image=cv2.rotate(frame_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        elif sd.FrameRotateMode==3:
                            frame_image=cv2.rotate(frame_image, cv2.ROTATE_180)
                        if sd.CameraFlipHorizontal:
                            frame_image=cv2.flip(frame_image,1)
                        self.mRawFrameImageInt=frame_image;
                        
                    else:
                        self.mRawFrameImageInt=None;
                        self.mRawFrameImageFloat=None;
                        #print("camera capture",frame_cap_start_time,end='\r')
                        #sd.mRawFrameImageFloat=frame_image
                except Exception as ex:
                    print("[P]摄像头画面捕捉发生错误: %s" % ex)

                    
            
            if self.SrcType=="video":
                if self.mFileVideoCapture is None:
                    self.mFileVideoCapture=cv2.VideoCapture(self.mVideoFilename);
                    if(self.mFileVideoCapture.isOpened() == False):
                        print("视频加载文件失败")
                if self.mFileVideoCapture.isOpened() == False:
                    return;

                if self.PlayToggle==True:
                    self.ReadVideoFileFrame();
             

            if self.SrcType=="capture":
                try:
                    frame_image=cap.capture_one_frame(sd.CaptureRect,sd.CaptureClipMode); 
                    self.mRawFrameImageInt=frame_image;
                except Exception as ex:
                    print("[P]截屏发生异常: %s" % ex)
                    pass

            if self.SrcType=="image":
                pass;
             
            
               #---- 原始画面捕捉结束，开始背景替换
            if sd.ReplaceBgAI:
                #self.mRawFrameImageFloat=self.MatEngine.GetAiMattingMerge(self.mRawFrameImageFloat)
                self.mRawFrameImageInt=self.MatEngine.GetAiMattingMerge(self.mRawFrameImageInt)
            if sd.ReplaceBgGreen:
                self.mRawFrameImageInt=self.MatEngine.GetGreenScreenFilter(self.mRawFrameImageInt)
            
            #---- 数据类型转换
            if self.mRawFrameImageInt is not None:
                self.mRawFrameImageFloat=ImageProcessor(self.mRawFrameImageInt).to_ufloat32().get_image('HWC') 
                sd.mRawFrameImageInt=self.mRawFrameImageInt;
                sd.mRawFrameImageFloat=self.mRawFrameImageFloat;
            else:
               sd.mRawFrameImageInt=self.mRawFrameImageInt=None;
               sd.mRawFrameImageFloat= self.mRawFrameImageFloat=None;

            if sd.ReplacePreview and (sd.mRawFrameImageFloat is not None) :
                cv2.imshow("matting",sd.mRawFrameImageFloat)
                cv2.waitKey(3)



            #---计算当前帧捕获+处理耗时
            now=time.time()
            time_use_seconds=now-frame_cap_start_time;
            sd.FrameCapUseTime=time_use_seconds*1000
            sd.FrameCaptureEndTimeTick=now
            #sd.DetectFaceInfo["cap_timetick"]=now

           
          
            if time_use_seconds<0.04:
                time.sleep(max(0.04-time_use_seconds,0.001))
            self.mFrameCaptureSignal.emit() 

    def ForcePlayVideoToFrame(self,frame_num):
        self.ForcePlayFrameEnable=True
        self.ForcePlayFrame=frame_num;
        #print(f"强制切换到帧{frame_num}")

    def ReadVideoFileFrame(self):
            if self.mFileVideoCapture is None:
                print("mFileVideoCapture is None")
                return;
     
            NowTime=time.time();
            deltaTime=NowTime-self.mLastPlayTime;
            deltaTime=0.0 if deltaTime>1 else deltaTime;
            fps=self.mFileVideoCapture.get(5)
            frame_count=self.mFileVideoCapture.get(7)
            last_frame=self.mFileVideoCapture.get(1);
            
            

            step_frame=int(deltaTime*fps);
            now_frame=last_frame+step_frame;
            #print(f"[P]now_frame:{now_frame},{frame_count}")
            if now_frame>=frame_count-1:
               self.mFileVideoCapture.set(1,0);
               #print(f"[P]now_frame:{now_frame},{frame_count},到达末尾")
            if step_frame>1 :
                for i in range(step_frame-1):
                    self.mFileVideoCapture.grab()
                success,frame_image=self.mFileVideoCapture.read()
            elif step_frame==1:
                success,frame_image=self.mFileVideoCapture.read()
            else:
                self.mFileVideoCapture.set(1,now_frame)

            if self.ForcePlayFrameEnable is True:
                self.mFileVideoCapture.set(1,self.ForcePlayFrame);
                self.ForcePlayFrameEnable=False;

            success,frame_image=self.mFileVideoCapture.read()
             
            self.mLastPlayTime=NowTime;
          
            frame_num=self.mFileVideoCapture.get(1);
            frame_num=round(frame_num)
            frame_count=round(frame_count)
            print(f"[V]{frame_num}/{frame_count}")
            #self.ui.spinFrameNum.setValue(frame_num);

            
            if success:
                h,w,c=frame_image.shape
                if sd.ResizeVideoFrame is True:
                    r=sd.ResizeVideoWidth/w
                    frame_image=cv2.resize(frame_image,(0,0),fx=r,fy=r)
                self.mRawFrameImageInt=frame_image;
                 

    def setMediaSource(self,type="camera",deviceID=0,videoFile="",input=None):
        #print(f"开始改变视频源为{type}")

        time_start=time.time()
        self.SrcType=type;
        if(self.SrcType=="video"):
            self.mVideoFilename=videoFile;
            self.mFileVideoCapture=cv2.VideoCapture(self.mVideoFilename);

        if(self.SrcType=="camera"):
            self.CamIdx=deviceID;
            if input is not None:
                CamIdx=input.get("cam_idx",0)
                width=input.get("width",640)
                height=input.get("height",480)
                rotate=input.get("rotate",0)
                drive=input.get("drive",0)
            if self.mCamVideoCapture is not None:
                self.mCamVideoCapture.release() 
            
            if drive==0:
                self.mCamVideoCapture=cv2.VideoCapture(CamIdx,cv2.CAP_MSMF) 
            else:
                self.mCamVideoCapture=cv2.VideoCapture(CamIdx,cv2.CAP_DSHOW)

            self.CamIdx=CamIdx;  
            self.mCamVideoCapture.set(5,24)
            self.mCamVideoCapture.set(3,int(width))
            self.mCamVideoCapture.set(4,int(height)) 

            if self.mCamVideoCapture.isOpened()==False:
                try:
                    self.mCamVideoCapture=cv2.VideoCapture(CamIdx,cv2.CAP_DSHOW)
                    self.mCamVideoCapture.open()
                except :
                    self.mCamVideoCapture=None;
                    print(f"该序号为{CamIdx}的摄像头在{width}x{height}分辨率下调用失败")
                  
            time_end=time.time()
            time_use=round((time_end-time_start),3)
            print(f"[P]切换摄像头为{type}_{self.CamIdx},{width}x{height}px,耗时{time_use}秒")
        
    
    def setVideoFileName(self,filename):
        self.mVideoFilename=filename;
        self.mFileVideoCapture=cv2.VideoCapture(self.mVideoFilename);

   


    def setCamDeviceID(self,CamIdx):
        self.CamIdx=CamIdx;

   
    
   