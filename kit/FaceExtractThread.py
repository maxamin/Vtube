#coding=utf-8
# cython:language_level=3
import threading,cv2,time,queue,os
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

class FaceExtractThread(QThread):
    mFrameProcessEndSignal=pyqtSignal();     
    mEventQueue=queue.Queue()

    def __init__(self,**kwargs):
        super(FaceExtractThread, self).__init__()
        self.Step=kwargs.get("step",5)
        self.VideoFilePath=kwargs.get("video_file",5);
        self.mFileVideoCapture=None;
        self.mThreadRun=True;   
        if os.path.exists(self.VideoFilePath) is False:
            print(f"错误，视频文件({self.VideoFilePath})不存在")
            self.mThreadRun=True;   
         

    def run(self):
        print(f"开始提取视频文件: {self.VideoFilePath}")
        while(self.mThreadRun):            
            while(self.mEventQueue.empty() is False):
                #print("fetch queue")
                input=self.mEventQueue.get()
                op = input['op']
                if op=="stop":
                    print("[P]停止后台自动提取线程")
                    return
           
            frame_start_time=time.time()
            #print(frame_cap_start_time,end='\r')

            ok,frame_img,frame_count=ToolKit.GetVideoFrameImage(self.VideoFilePath,sd.VideoCurrFrameIndex)
          
            if sd.VideoCurrFrameIndex>= (frame_count-1):
                print("到达视频结尾帧，人脸提取进程结束")
                return
            sd.VideoCurrFrameIndex=sd.VideoCurrFrameIndex+ self.Step;

            now=time.time()
            time_use_seconds=now-frame_start_time;
            time_use_seconds=round(time_use_seconds,3)
            if time_use_seconds<0.04:
                time.sleep(max(0.04-time_use_seconds,0.001))
            print(f"[P]处理帧{sd.VideoCurrFrameIndex}/{sd.VideoFrameCount},耗时{time_use_seconds}秒")
            self.mFrameProcessEndSignal.emit() 

              
   


   