#coding=utf-8
# cython:language_level=3
from PyQt5.QtWidgets import QFileDialog,QMessageBox,QApplication,QInputDialog;
from PyQt5.QtCore import QThread,QObject,pyqtSignal;
from PyQt5.QtGui import QPixmap,QImage,QTextCursor;

from pathlib import Path
import os,time,threading,pickle,shutil,cv2,numpy,sys,datetime,numpy as np;
from core.leras import nn
from core import  pathex,cv2ex
from facelib import LandmarksProcessor,FaceType
from core.DFLIMG import DFLIMG,DFLJPG
from kit import ToolKit
import kit.ShareData as sd
import kit.screen_cap as cap
from facelib.facenet.facenet import facenet
import keyboard,webbrowser
from kit.ServerRequestThread import ServerRequestThread

DefaultFolder="F:/"
class Signal(QObject):
    console_update = pyqtSignal(str)
    def write(self, text):
        self.console_update.emit(str(text))
        QApplication.processEvents()
    def flush(self):
        pass;


class LabUiHandler():

    SrcType="capture";
    mAuthCheckThread=None;

    def __init__(self,ui):
        self.ui=ui;

        sys.stdout = Signal()
        sys.stdout.console_update.connect(self.UpdateConsoleText)
        sd.SoftName=f"{sd.SoftName}_Lab";
        self.ui.setWindowTitle(f"{sd.SoftName} (version:{sd.Version})");
         

        self.UpdateWindowListForUi()
        self.LoadSavedConfig();

        th=threading.Thread(target=self.LoadModelThreadRun,args=(3,))
        th.daemo=True;
        th.start();

        keyboard.hook(self.global_key_hook_proc)

        self.mAuthCheckThread=ServerRequestThread(app=sd.SoftName,delay=5.0,mode="auth_check",interval=8)
        self.mAuthCheckThread.mAlertSignal.connect(self.AuthCheckAlert)
        self.mAuthCheckThread.start();

    def UpdateConsoleText(self,text): 
        if '\r' in text or "\n" in text:
           return;
        if text.startswith("[ui]"):
            self.UpdateUiFromData()
            return
        now=datetime.datetime.now().strftime('%H:%M:%S')
        if text.startswith('[M]') :
            ToolKit.ShowWarningError(None,"提示",text);
            return
         
        self.ui.statusbar.showMessage(f"[{now}] {text}",200000)


    def AuthCheckAlert(self):
        if sd.ShowAlert:
            ToolKit.ShowWarningError(None,"提示",sd.AlertMsg); 
        if sd.ExitApp:
            exit(0);
        if sd.OpenWeb:
            webbrowser.open(sd.WebUrl, new=0, autoraise=True);

    def global_key_hook_proc(self,evt):
        if self.ui.tabWidget.currentIndex()>0:
            return;
        if evt.event_type == 'down':
            if evt.scan_code == 57:  #space-auto save
                #self.btnBatchExtractFaceClick()
                pass;
            if evt.scan_code >= 59 and evt.scan_code <= 63:  #F1-F5
                self.btnExtractFrameFaceClick()
            if evt.scan_code == 2:  #num_1---extrace 
                self.btnExtractFrameFaceClick()
            if evt.scan_code == 3:  #num_2---save
                self.btnSaveFaceAndStepClick()
            if evt.scan_code == 5:  #num_4---clear filter
                self.btnClearIdentityFilterFaceClick()
            if evt.scan_code == 1:  #Esc--unhook
                 #keyboard.unhook_all()
                 pass
             
        
    def LoadModelThreadRun(self,s):
        ToolKit.EnsureFaceDetectModelLoaded()



    def onSignalReceive(self,v):
        self.ui.spinBoxCurrentFrame.setValue(v);
         
    
    def LoadSavedConfig(self):
        cfg_options=ToolKit.GetSavedConfigData(cfg_file="lab.cfg")
        if cfg_options is None:
            #print("cfg_options is none")
            return;
        self.ui.spinMinFaceSize.setValue(cfg_options.get("MinFaceSize",80))
        self.ui.spinBoxExtractFaceSize.setValue(cfg_options.get("ExtractFaceSize",256))
 
        self.ui.LineEditVideoFile.setText(cfg_options.get("VideoFile",""))
        self.ui.LineEditImageFolder.setText(cfg_options.get("ImageFolder",""))
        self.ui.LineEditFaceSaveFolder.setText(cfg_options.get("FaceSaveFolder",""))
        self.ui.LineEditAdjustFolder.setText(cfg_options.get("AdjustFolder",""))
        self.ui.LineEditPreviewFolder.setText(cfg_options.get("PreviewFolder",""))
         
        self.ui.LineEditOccludeSaveFolder.setText(cfg_options.get("OccludeSaveFolder",""))

        #self.ui.LineEditTrainDstFaceFolder.setText(cfg_options.get("DstFaceFolder",""))
        #self.ui.LineEditTestSampleDir.setText(cfg_options.get("TrainTestFolder",""))
        

        
    def SaveConfigData(self,txt=None):        
        options={}
        options["MinFaceSize"]=self.ui.spinMinFaceSize.value();
        options["ExtractFaceSize"]=self.ui.spinBoxExtractFaceSize.value();
 
        options["VideoFile"]=self.ui.LineEditVideoFile.text();
        options["ImageFolder"]=self.ui.LineEditImageFolder.text();
        options["FaceSaveFolder"]=self.ui.LineEditFaceSaveFolder.text();
        options["AdjustFolder"]=self.ui.LineEditAdjustFolder.text();
        options["PreviewFolder"]=self.ui.LineEditPreviewFolder.text();
         
        options["OccludeSaveFolder"]=self.ui.LineEditOccludeSaveFolder.text(); 
        

        ToolKit.SavedConfigDataDict(options,"lab.cfg") 
        #print("Finish Save Config Data")

    def UpdateWindowListForUi(self):
        sd.hWnd_info_list=cap.get_all_windows()
        cap.set_cap_desktop()
        self.ui.comboWindowList.clear();
        for hwnd_info in sd.hWnd_info_list: 
            self.ui.comboWindowList.addItem(f"{hwnd_info[2]}")

    def SetScreenCaptureWindow(self):
        idx=self.ui.comboWindowList.currentIndex();
        hwnd,classname,title=sd.hWnd_info_list[idx]
        #print("选择了屏幕捕获窗口：",classname,title,hwnd)
        succeed=cap.cap_init(classname,title,hwnd)
        if succeed==False:
            ToolKit.ShowWarningError(None,"错误","查找窗口失败") 

    def btnRefreshWindowSizeClick(self):
        cap.calc_window_size();
        self.ui.spinCaptureLeft.setValue(cap.left)
        self.ui.spinCaptureTop.setValue(cap.top)
        self.ui.spinCaptureWidth.setValue(cap.width)
        self.ui.spinCaptureHeight.setValue(cap.height)

    def SetScreenCaptureRect(self):
        left=self.ui.spinCaptureLeft.value();
        top=self.ui.spinCaptureTop.value();
        width=self.ui.spinCaptureWidth.value();
        height=self.ui.spinCaptureHeight.value();
        sd.CaptureRect=(left,top,width,height)
        sd.CaptureClipMode=self.ui.comboLabClipMode.currentIndex();

    def CaptureCurrentScreenFrame(self):
        self.current_frame_image=cap.capture_one_frame(sd.CaptureRect,sd.CaptureClipMode);
        self.ui.FrameImageMarkLabel.clear()
        ToolKit.ShowImageInLabel(self.current_frame_image,self.ui.FrameImageMarkLabel,bgr_cvt=True,scale=True) 
    
    def CaptureCurrentCameraFrame(self):
        if hasattr(self,"mCameraVideoCapture") is False:
            idx=self.ui.comboSrcCameraID.currentIndex();
            self.mCameraVideoCapture=cv2.VideoCapture(idx)
            self.onChangeCameraResolution()
            print("Create cv2.VideoCapture")
        ok,self.current_frame_image=self.mCameraVideoCapture.read()
        h,w,c=self.current_frame_image.shape
        cv2.putText(self.current_frame_image, f"{w}*{h}px",(0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ToolKit.ShowImageInLabel(self.current_frame_image,self.ui.FrameImageMarkLabel,bgr_cvt=True,scale=True) 
        pass

    def onChangeCameraResolution(self):
        resolution=self.ui.comboSrcCameraResolution.currentText();
        cameraID=self.ui.comboSrcCameraID.currentIndex();

        reso=resolution.replace("px","").split("*");
        if hasattr(self,"mCameraVideoCapture"):
            width,height=int(reso[0]),int(reso[1])
            #if(self.mCameraVideoCapture.isOpened()==False): 
            self.mCameraVideoCapture.open(cameraID)
            #print(f"open device camera {cameraID}")
            self.mCameraVideoCapture.set(3,width)
            self.mCameraVideoCapture.set(4,height)
            print(f"Set Camera Resolution:{width}x{height}")
        else:
            QMessageBox.warning(None,"错误","尚未启动摄像头");



    def OnBackendInitFinish(self):
        pass 

    #---------选择视频文件----------
    def onBrowseVideoFileClick(self):
        file,ok=QFileDialog.getOpenFileName(None,"选择视频文件",DefaultFolder);
        if ok:
            self.ui.LineEditVideoFile.setText(file);
            sd.VideoFileName=file;
            print(f"[P]打开视频文件{file}",end='\r');
            dir=os.path.dirname(file)
            name=os.path.basename(file)
            dir,ext=os.path.splitext(file)

            ok,frame_image,frame_num=ToolKit.GetVideoFrameImage(file,1) 
            if ok:
                self.ui.spinFrameCount.setValue(frame_num)
                ToolKit.ShowImageInLabel(frame_image,self.ui.FrameImageMarkLabel,bgr_cvt=True) 
            else:
                ToolKit.ShowWarningError(None,"错误","读取视频错误");
                return

    def btnOpenVideoFolderClick(self):
        extract_video=self.ui.LineEditVideoFile.text();
        dir=os.path.dirname(extract_video)
        if os.path.exists(dir):
            os.startfile(dir)
        else:
            ToolKit.ShowWarningError(None,"错误","目录不存在");

    #---------选择保存的图片文件夹----------
    def btnBrowseImageFolderClick(self):
        folder=QFileDialog.getExistingDirectory(None,"选择文件夹",DefaultFolder);
        if(len(folder)>2):
            self.ui.LineEditImageFolder.setText(folder);
            self.LoadImageListFromFolder(folder);

    def LoadImageListFromFolder(self,folder=None):
        if(folder is None):
            folder=self.ui.LineEditImageFolder.text();
        if os.path.exists(folder)==False:
            QMessageBox.warning(None,"错误","图片文件夹不存在");
            return;
        sd.ImageList.clear();
        sd.CurrImageIdx=0;
        sd.ImageCount=0;
        files=[file for file in os.listdir(folder) if ".jpg" in file ]
        for name in files:
            ext=os.path.splitext(name)[1].lower();
            if(ext.find("jpg")==-1):
                continue;
            full_path=os.path.join(folder, name);
            sd.ImageList.append((full_path,name))
                #print(f"add {full_path}---{name} to ImageList")

        sd.ImageCount=len(sd.ImageList);
        print(f"[P]load image folder {folder}, image number {sd.ImageCount}",end='\r')
    #---------文件浏览器打开图片文件夹----------
    def btnOpenImageFolderClick(self):
        folder=self.ui.LineEditImageFolder.text();
        print(f"[P]打开文件夹：{folder}",end='\r');
        if(os.path.exists(folder)==False):
            ToolKit.ShowWarningError(None,"warning","文件夹不存在");
            return;
        try:           
            os.startfile(folder)
        except:
            ToolKit.ShowWarningError(None,"warning","打开失败");

    def btnPreImageClick(self):
        if sd.CurrImageIdx>1:
            sd.CurrImageIdx=sd.CurrImageIdx-1;
        self.LoadImageFileAsFrame();

    def btnNextImageClick(self):
        if len(sd.ImageList)==0:
            self.LoadImageListFromFolder()
        if sd.CurrImageIdx<=sd.ImageCount-1:
            self.LoadImageFileAsFrame();
            sd.CurrImageIdx=sd.CurrImageIdx+1;
        else:
            print("[M]已经到最后一张");
            self.AutoRunContinue=False;
        
    def btnFirstImageClick(self): 
        sd.CurrImageIdx=0;
        self.LoadImageFileAsFrame();

    def btnLastImageClick(self):
        sd.CurrImageIdx=sd.ImageCount-1;
        self.LoadImageFileAsFrame();



    def LoadImageFileAsFrame(self):
        self.ui.label_image_idx.setText(f"{sd.CurrImageIdx+1}/{sd.ImageCount}")
        if len(sd.ImageList)==0:
            ToolKit.ShowWarningError(None,"","图片文件列表为空")
            self.current_frame_image=None;
            return;
        FileToLoad=sd.ImageList[sd.CurrImageIdx];
        FullPath=FileToLoad[0];
        FileName=FileToLoad[1]
        self.ui.label_curr_image.setText(FileName)
        
        self.current_frame_image=cv2ex.cv2_imread(FullPath)
        ToolKit.ShowImageInLabel(self.current_frame_image,self.ui.FrameImageMarkLabel)
        

    #---------浏览人脸文件夹----------
    def btnBrowseFaceFolderClick(self):
        folder=QFileDialog.getExistingDirectory(None,"选择人脸文件夹",os.path.dirname(self.ui.LineEditFaceSaveFolder.text()));
        if(len(folder)>2):
            self.ui.LineEditFaceSaveFolder.setText(folder);
    #------打开人脸文件夹
    def btnOpenFaceFolderClick(self):
        folder=self.ui.LineEditFaceSaveFolder.text();  
        try:           
            os.startfile(folder)
        except:
            ToolKit.ShowWarningError(None,"错误","打开人脸文件夹失败,请检查文件夹是否存在");



 

        
    def goToStartVideoFrameClick(self):
        self.stepVideoFrameNum(-99999999);

    def goToPreVideoFrameClick(self):
        self.stepVideoFrameNum(-1) 

    def goToNextVideoFrameClick(self):
        self.stepVideoFrameNum(1) 
        #---判断批量提取视频的时候，是否到结尾了
        step_frame=self.ui.spinBoxFrameStepNum.value()
        curr_frame=self.ui.spinBoxCurrentFrame.value()
        total_frame=self.ui.spinFrameCount.value()
       
        if (step_frame+curr_frame)>=total_frame:
            self.btnStopExtractFaceClick();
            #print("[M]到达视频结尾");
            return 

    def goToLastVideoFrameClick(self):
        self.stepVideoFrameNum(99999999);

    def stepVideoFrameNum(self,steps,scaleStep=True,src="video",preview=True):
        extract_video=self.ui.LineEditVideoFile.text();
        FrameStepNum=self.ui.spinBoxFrameStepNum.value();
        currFrameNum=self.ui.spinBoxCurrentFrame.value();
        if scaleStep:
            steps=steps*FrameStepNum;
        newFrame=max(currFrameNum+steps,1)
        sd.VideoFrameCount=VideoFrameCount=ToolKit.GetVideoFrameCount(extract_video)
        newFrame=min(VideoFrameCount-1,newFrame)
        self.ui.spinBoxCurrentFrame.setValue(newFrame); 
        self.ui.spinFrameCount.setValue(VideoFrameCount)
        if(preview==False):
            return;
        currFrameNum=self.ui.spinBoxCurrentFrame.value();
        video_name=os.path.basename(extract_video);
        video_name=video_name.split('.')[0]
        ok,frame_image,frame_count=ToolKit.GetVideoFrameImage(extract_video,currFrameNum) 
        if ok:
            #marker_frame_image=frame_image.copy();
            #cv2.putText(marker_frame_image, f"No:{currFrameNum}/{frame_count}",(0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ToolKit.ShowImageInLabel(frame_image,self.ui.FrameImageMarkLabel,bgr_cvt=True) 
            self.current_frame_image=frame_image;
        else:
            print("[M]没读取到帧图片");
            return




    def PreviewCurrentFrame(self):
        extract_video=self.ui.LineEditVideoFile.text();
        frame_num=self.ui.spinBoxCurrentFrame.value();
        video_name=os.path.basename(extract_video);
        video_name=video_name.split('.')[0]
        ok,frame_image,frame_count=ToolKit.GetVideoFrameImage(extract_video,frame_num) 
        if ok:
            marker_frame_image=frame_image.copy();
            cv2.putText(marker_frame_image, f"No:{frame_num}/{frame_count}",(0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ToolKit.ShowImageInLabel(marker_frame_image,self.ui.FrameImageMarkLabel,bgr_cvt=True) 
            self.current_frame_image=frame_image;
        else:
            print("[M]没读取到帧图片");
            return



    #----- 设置身份筛选图片
    StandardIdentityFaceImg=None;
    def btnBrowseIdentityFilterFaceImageClick(self):
        file,ok=QFileDialog.getOpenFileName(None,"选择人脸文件","",'image files (*.jpg *.png )');
        if ok:
           frame_image= cv2ex.cv2_imread(file);
           face_imgs,align_imgs,_,_=ToolKit.GetDetectAndAlignFace(frame_image,256)
           if len(face_imgs)==0:
               ToolKit.ShowWarningError(None,"设置失败","未能从图片中提取到人脸")
               return;
           #self.StandardIdentityFaceImg=align_imgs[0];
           facenet.add_ref_face_img(align_imgs[0]);
           self.StandardIdentityFaceImg=facenet.getRefImagesMerge();
           ToolKit.ShowImageInLabel(self.StandardIdentityFaceImg,self.ui.label_identity_face_img,bgr_cvt=True,scale=True)

    #----- 清除身份筛选图片
    def btnClearIdentityFilterFaceClick(self):
        facenet.clear_standard_face();
        self.ui.label_identity_face_img.clear() 
     
    
    #----- 添加身份筛选图片
    def btnAddRefImageClick(self):
        if self.current_frame_image is None :
            ToolKit.ShowWarningError(None,"错误","当前帧画面为空，无法提取人脸")
            return; 
        if len(self.extract_aligned_images)==0:
            ToolKit.ShowWarningError(None,"失败","当前提取的人脸列表为空")
            return;
        if self.ui.radioButtonThird.isChecked():
            idx= 2;
        if self.ui.radioButtonSecond.isChecked():
            idx= 1;
        else:
            idx= 0;
        if idx>=len(self.extract_aligned_images):
            ToolKit.ShowWarningError(None,"错误","序号错误，序号超过了提取的人脸总数")
            return; 
        ref_img=self.extract_aligned_images[idx]
        facenet.add_ref_face_img(ref_img)
        ToolKit.ShowImageInLabel(facenet.getRefImagesMerge(),self.ui.label_identity_face_img,bgr_cvt=True,scale=True)
       
    #----- 删除身份筛选图片
    def btnDeleteLastRefClick(self):
        facenet.del_last_face_img()
        ToolKit.ShowImageInLabel(facenet.getRefImagesMerge(),self.ui.label_identity_face_img,bgr_cvt=True,scale=True)



    current_frame_image=None;
    extract_raw_images=[]
    extract_aligned_images=[];
    extract_landmarks=[];
    extract_angles_list=[];
    extract_src_landmarks_list=[];
    mCamVideoCapture=None; 
    
    #--- 提取人脸
    def ExtractCurrentFrameFaces(self,face_type_str="full_face",image_size=256):

        AutoStep=self.ui.checkBoxAutoNext.isChecked();
        FrameSrc=self.ui.tabWidgetFrameSrc.currentIndex();
        if AutoStep and FrameSrc==0:
            self.goToNextVideoFrameClick();
        if AutoStep and FrameSrc==1:
            self.btnNextImageClick();
        if FrameSrc==2:
            self.CaptureCurrentScreenFrame()
        if FrameSrc==3:
            self.CaptureCurrentCameraFrame()

        face_type_str=self.ui.comboBoxFaceType.currentText();
        image_size=self.ui.spinBoxExtractFaceSize.value();
        filter_face_size=self.ui.spinMinFaceSize.value();
        face_coverage=self.ui.spinFaceCoverage.value();
        detector=self.ui.comboDetectEngine.currentText()
        window_width=self.ui.spinBoxDetectWindow.value()
        frame_image=self.current_frame_image;
        detect_time_start_tick=time.time();

        if(frame_image is None):
            return;

        self.extract_raw_images,self.extract_aligned_images,self.extract_landmarks, self.extract_angles_list  \
        =ToolKit.GetDetectAndAlignFace(frame_image,image_size,filter_face_size,face_coverage=face_coverage,window_width=window_width,detector=detector)
        time_use=(time.time()-detect_time_start_tick)*1000;
        self.ui.groupBox_Detect.setTitle(f"检测耗时:{round(time_use)}ms")
        if(len(self.extract_aligned_images)>0):   #平铺显示提取到的人脸
            self.detect_align_img_preview=numpy.hstack(self.extract_aligned_images)
        else:
            self.detect_align_img_preview=None;
        
               
    def FilterExtractedFaces(self):
        id_verify_threshold=self.ui.spinExtractVerifyThreshold.value();
        del_index_list=[]
        for i in range(0,len(self.extract_aligned_images)):
            verify_face=self.extract_aligned_images[i] 
            ok,dist=facenet.getVerifyResult(verify_face,id_verify_threshold)
            if ok is False:
                del_index_list.append(i)
        del_index_list.reverse()
        for i in del_index_list:
            self.extract_aligned_images.pop(i)
            self.extract_angles_list.pop(i)
            self.extract_landmarks.pop(i) 

        if(len(self.extract_aligned_images)>0):    
            self.filtered_align_img_preview=numpy.hstack(self.extract_aligned_images)
        else:
            self.filtered_align_img_preview=None;

    #-----点击提取当前帧的人脸
    def btnExtractFrameFaceClick(self):              
        self.ExtractCurrentFrameFaces() 
        if self.detect_align_img_preview is not None: 
            ToolKit.ShowImageInLabel(self.detect_align_img_preview,self.ui.ExtractedAlignedFacesLabel,bgr_cvt=True);  
        else:
            self.ui.ExtractedAlignedFacesLabel.clear(); 

    #--点击筛选人脸按钮
    def btnFilterExtractedFacesClick(self): 
        self.FilterExtractedFaces()
        if self.filtered_align_img_preview is not None: 
            ToolKit.ShowImageInLabel(self.filtered_align_img_preview,self.ui.ExtractedFilterAlignedFacesLabel,bgr_cvt=True); 
        else:
            self.ui.ExtractedFilterAlignedFacesLabel.clear(); 
        self.ui.label_filter_face_num.setText(f"筛选到人脸:{len(self.extract_aligned_images)}")



   
         

    def SaveFilterFaces(self,save_dir):
        if (len(self.extract_aligned_images)==0):
            frame_num=self.ui.spinBoxCurrentFrame.value();
            step_whole_time_use=time.time()-self.frame_start_timetick;
            step_whole_time_use=round(step_whole_time_use*1000)
            print(f"[P]当前帧没有提取到人脸，无需保存，耗时{step_whole_time_use}ms")
            return 

        self.extract_filter_end_timetick=time.time()
        
        AutoStep=self.ui.checkBoxAutoNext.isChecked();
        FrameSrc=self.ui.tabWidgetFrameSrc.currentIndex();
        SavePrefix=self.ui.LineEditSavePrefix.text();
        if(len(SavePrefix)>0):
            SavePrefix=SavePrefix+"_";
        if  FrameSrc==0:
            frame_num=self.ui.spinBoxCurrentFrame.value();
            save_filename=f"{SavePrefix}f{frame_num}"  
        if FrameSrc==1:
            save_filename=f"{SavePrefix}p{sd.ExtractNum}"  
        if FrameSrc==2:
            save_filename=f"{SavePrefix}c{sd.ExtractNum}"
        if FrameSrc==3:
            save_filename=f"{SavePrefix}cm{sd.ExtractNum}"
        sd.ExtractNum=sd.ExtractNum+1;
        #self.ui.spinExtractNum.setValue(sd.ExtractNum)
        #-- 输出文件的文件名
 
        idx=0;
        for face_image,face_landmark,face_angles in zip(self.extract_aligned_images,self.extract_landmarks,self.extract_angles_list): 
            output_final_filepath=f"{save_dir}/{save_filename}_{idx}.jpg"  
            if(os.path.exists(output_final_filepath)):
                output_final_filepath=f"{save_dir}/{save_filename}_{idx}_n.jpg"  
            ToolKit.SaveAsDflImage(face_image,output_final_filepath,"whole_face",face_landmark,None,face_angles)
            #print("保存地址：",output_final_filepath)
            idx=idx+1;
  
        frame_end_timetick=time.time()
        step_whole_time_use=round(frame_end_timetick*1000-self.frame_start_timetick*1000);
        step_save_time_use=round(frame_end_timetick*1000-self.extract_filter_end_timetick*1000) 
        extract_time_use=round(self.extract_filter_end_timetick*1000-self.frame_start_timetick*1000)
        print(f"[P]单帧提取耗时{step_whole_time_use}ms,【读帧{sd.VideoFrameReadTime}ms,检测筛选{extract_time_use}ms,遮罩计算+保存{step_save_time_use}ms】,保存为{output_final_filepath}")

 
    #---- 提取单帧的按钮处理-----
    def btnSaveFaceAndStepClick(self):
        self.frame_start_timetick=time.time();
        face_type_str=self.ui.comboBoxFaceType.currentText();
        save_dir=self.ui.LineEditFaceSaveFolder.text(); 
        if(os.path.exists(save_dir)==False):
            self.btnStopExtractFaceClick()
            print("[M]人脸输出路径不存在");
            self.AutoRunContinue=False;
            return 
          
        self.btnExtractFrameFaceClick(); 
        self.btnFilterExtractedFacesClick()
        self.SaveFilterFaces(save_dir);
        
         

    #---------------(多线程)批量提取--UI主线程-----------------
    def onThreadBatchExtractFaceClick(self):
        #检查是否符合条件
        idx=self.ui.tabWidgetFrameSrc.currentIndex()
        if idx==0: #视频
            frame_count=self.ui.spinFrameCount.value()
            if frame_count==0:
                print("[M]尚未读取视频信息，可以点击下一帧按钮先读取视频信息")
                return

        self.AutoRunContinue=True;
        save_dir=self.ui.LineEditFaceSaveFolder.text(); 
        th=threading.Thread(target=self.BatchExtractSaveThreadRun,args=(save_dir,))
        th.daemo=True;
        th.start(); 

        self.ui.btnBatchExtractThread.setEnabled(False)


        #----批量提取线程执行函数---
    def BatchExtractSaveThreadRun(self,save_dir):
        while(self.AutoRunContinue ):
            self.frame_start_timetick=time.time()
            self.ExtractCurrentFrameFaces()
            self.FilterExtractedFaces()
            self.SaveFilterFaces(save_dir) 
            time.sleep(0.012)
            print("[ui]")
        print("自动提取停止")
        self.btnStopExtractFaceClick()


    def btnStopExtractFaceClick(self):
        self.AutoRunContinue=False;
        self.ui.btnBatchExtractThread.setEnabled(True)
   
    def UpdateUiFromData(self):        
        if hasattr(self,"detect_align_img_preview"):
            if self.detect_align_img_preview is not None:
                ToolKit.ShowImageInLabel(self.detect_align_img_preview,self.ui.ExtractedAlignedFacesLabel,bgr_cvt=True);  
            else:
                self.ui.ExtractedAlignedFacesLabel.clear()
        if hasattr(self,"filtered_align_img_preview"):
            if self.filtered_align_img_preview is not None:
                ToolKit.ShowImageInLabel(self.filtered_align_img_preview,self.ui.ExtractedFilterAlignedFacesLabel,bgr_cvt=True); 
            else:
                self.ui.ExtractedFilterAlignedFacesLabel.clear()

    def btnVerifyChooseFaceClick(self):
        face_file=self.ui.LineEditVerifyFace.text();
        face_folder=self.ui.LineEditRecognizeDirectory.text();
        if(os.path.exists(face_file)==False ):
            ToolKit.ShowWarningError(None,"错误","选取的人脸文件不存在"+face_file);
            return;
        if(os.path.exists(face_folder)==False):
            ToolKit.ShowWarningError(None,"错误","选取的人脸文件夹不存在"+face_folder);
            return;
        thres=self.ui.spinVerifyThreshold.value();
        model_name=self.ui.comboVerifyModel.currentText();
        
        th=threading.Thread(target=self.VerifyChooseFaceThreadRun,args=(face_folder,face_file,thres))
        th.daemo=True;
        th.start();
