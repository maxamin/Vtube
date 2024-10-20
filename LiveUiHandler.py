#coding=utf-8
# cython:language_level=3
import pickle,time,threading,cv2,os,numpy as np,sys,pickle,datetime,shutil;
import webbrowser;
from core.leras import nn
from PyQt5.QtWidgets import QFileDialog,QMessageBox,QApplication,QInputDialog,QListView,QDialog,QFormLayout;
from PyQt5.QtCore import QTimer,QThread,QObject,pyqtSignal;
from PyQt5.QtGui import QPixmap,QImage,QTextCursor,QIcon;
from PyQt5.QtCore import QStringListModel,QAbstractListModel,QModelIndex,QSize,Qt

from core.xlib.image.ImageProcessor import ImageProcessor  
from kit.FrameCaptureThread import FrameCaptureThread
from kit.WholeFaceSwapThread import WholeFaceSwapThread
from kit.AudioRecorder import AudioRecorder
from core.DFLIMG.DFLJPG import DFLJPG 
import core.pathex as pathex
from core import cv2ex
import kit.ShareData as sd
import kit.ToolKit as ToolKit
import kit.screen_cap as cap
from kit.LiveSwapModel import  LiveSwapModel
import kit.Auth as auth
from facelib.facenet.facenet import facenet
from kit.ServerRequestThread import ServerRequestThread
#----class head

class Signal(QObject):
    console_update = pyqtSignal(str)
    def write(self, text):
        self.console_update.emit(str(text))
        QApplication.processEvents()
    def flush(self):
        pass;

class LiveUiHandler:

      #-------------- class member 
    mServerRequestThread=None; mAuthCheckThread=None; mUpdateImageTimer=None;mAudioRecorder=None;
    mDetectAlignThread=None; mSwapFaceThread=None;mFrameCaptureThread=None;mMergePreviewThread=None;
    mWholeFaceSwapThread=None;

    def UpdateConsoleText(self,text): 
        now=datetime.datetime.now().strftime('%H:%M:%S')
        if text.startswith('[T]') or text.startswith('[P]'):
            self.print_endline=False;
            self.ui.statusbar.showMessage(f"[{now}] {text}");
            return
        if text.startswith('[V]'):
            self.print_endline=False;
            frame_str=text.replace("[V]","") 
            curr_frame,totalframe=frame_str.split("/")
            curr_frame=int(curr_frame)
            totalframe=int(totalframe)
            self.ui.sliderVideoPlayFrame.setMaximum(totalframe)
            self.ui.sliderVideoPlayFrame.setValue(curr_frame)
            self.ui.spinVideoPlayFrame.setMaximum(totalframe)
            self.ui.spinVideoPlayFrame.setValue(curr_frame)
            return
        if '\r' in text:
            return
        cursor = self.ui.textEditConsole.textCursor()  
        cursor.movePosition(QTextCursor.End)
        if '\n' in text:
            if  self.print_endline==True:
                cursor.insertText(text)  
        else:
            cursor.insertText(f"[{now}] {text}")  
            self.print_endline=True

    def __init__(self,ui):
        self.ui=ui;
        self.media_source="camera"
        self.model=None;
        sd.ShowWarn=True;
        sd.ShowAd=True;
        
        
        sys.stdout = Signal()
        sys.stdout.console_update.connect(self.UpdateConsoleText)     
        self.mLastPlayTime=0.0;
        self.mPlayFrameNum,self.mFrameCount=1,10;
        self.PlayToggle,self.PreviewWindowToggle=True,False;
        nn.initialize_main_env()  
        #self.ui.setWindowTitle("DeepFaceAudio");
        sd.SoftName=f"{sd.SoftName}_Live";
        self.ui.setWindowTitle(f"{sd.SoftName} (version:{sd.Version})");
        self.ui.setWindowIcon(QIcon(sd.LiveLogoImage))
         

    def Init_All_System(self):
        self.ReadLicenceFile();
        self.LoadSavedConfig(); 
        self.UpdateWindowListForUi()
        self.UpdateListForModelFiles()
        self.ListAllCameras()

        self.mAudioRecorder=AudioRecorder()
        self.mAuthCheckThread=ServerRequestThread(app=sd.SoftName,delay=5.0)
        self.mFrameCaptureThread=FrameCaptureThread()
         
        self.mAuthCheckThread.mAlertSignal.connect(self.AuthCheckAlert)
        self.mAuthCheckThread.start();

        self.mFrameCaptureThread.mFrameCaptureSignal.connect(self.UpdateFrameSourcePreview) 
        self.mFrameCaptureThread.start()
        self.mWholeFaceSwapThread=WholeFaceSwapThread();
        self.mWholeFaceSwapThread.mDetectAlignEndSignal.connect(self.UpdateDetectAlignPreview)
        self.mWholeFaceSwapThread.mSwapEndUpdateSignal.connect(self.UpdateSwapOutputPreview)
        self.mWholeFaceSwapThread.mMergeEndSignal.connect(self.UpdateMergeOutputPreview)
        self.mWholeFaceSwapThread.start()

          
        self.ApplyCameraCaptureSetting();
        ui_model_file_path=self.ui.lineEditModelFilePath.text();
        ui_model_file_path=sd.swap_model_dir+"/"+ui_model_file_path
        #if os.path.exists(ui_model_file_path) and os.path.isfile(ui_model_file_path):
        #    self.mWholeFaceSwapThread.SetModelFile(ui_model_file_path)

        self.ApplyConfigDataChange(Save=False)
        

    def ListAllCameras(self):
        from PyCameraList.camera_device import list_video_devices
        cameras=list_video_devices()
        self.ui.comboLiveCameraID.clear()
        for camera in cameras:
            camera_name=f"{camera[0]}:{camera[1]}"
            self.ui.comboLiveCameraID.addItem(camera_name)
       

    def ReadLicenceFile(self):
        try:
            path=os.path.abspath(os.getcwd())
            dir=os.path.dirname(path); 
            licence_paths=[file for file in os.listdir(dir) if  file.endswith(".lic")]
            if len(licence_paths)==0:
                return
            licence_path=dir+"\\"+licence_paths[0] 

            #--读取licence文件
            import configparser
            config=configparser.ConfigParser()
            config.read(licence_path)
            sn=config.get("lic","sn")
            auth_version=config.get("lic","auth_version")
            expire_live=config.get("lic","expire_live")
            expire_train=config.get("lic","expire_train")
            expire_extract=config.get("lic","expire_extract")
            token=config.get("lic","token")
            merge=f"{sn}{expire_live}{expire_train}{expire_extract}vtkasd";
            token_verify=auth.get_md5_hash_str(merge); 

            #---比对时间
            #print(f"sd.SysDate:{sd.SysDate}|expire_live:{expire_live}")
            if(expire_live<sd.SysDate):
                print(f"许可文件过期（Licence Expire)")
                return
            

            #--比对机器码
            my_sn=auth.getMachineSn(False)
            if(my_sn!=sn):
                print("许可文件机器码错误");
                return

            if token.upper()==token_verify.upper() :
                print("已经读取许可文件，授权正常")
                sd.ShowWarn=False
        except Exception as ex:
            print(ex);
        

    def AuthCheckAlert(self):
        if sd.ShowAlert:
            ToolKit.ShowWarningError(None,"提示",sd.AlertMsg); 
        if sd.ExitApp:
            exit(0);
        if sd.OpenWeb:
            webbrowser.open(sd.WebUrl, new=0, autoraise=True);

    def btnResetBeatyClick(self):

        self.ui.spinSharpen.setValue(0);
        self.ui.spinFaceWhite.setValue(0);
        self.ui.spinFaceSmooth.setValue(0); 

        self.ui.spinMalarThin.setValue(0);
        self.ui.spinCheekThin.setValue(0);
        self.ui.spinJawThin.setValue(0);
        self.ui.spinFaceLong.setValue(0);
        self.ui.spinSmallMouth.setValue(0);
        self.ui.spinEyeDistance.setValue(0);
        self.ui.spinLowCheekWidth.setValue(0);
        


    def LoadSavedConfig(self):
        cfg_options=ToolKit.GetSavedConfigData()
        if cfg_options is None:
            return
        self.ui.comboLiveCameraID.setCurrentIndex(cfg_options.get("cam_id",0)) 
        self.ui.comboCameraRotate.setCurrentIndex(cfg_options.get("cam_rotate",0))
        self.ui.comboCamResolution.setCurrentIndex(cfg_options.get("cam_resolution",0))

        self.ui.spinCapRectLeft.setValue(cfg_options.get("cap_rect_left",0));
        self.ui.spinCapRectTop.setValue(cfg_options.get("cap_rect_top",0));
        self.ui.spinCapRectWidth.setValue(cfg_options.get("cap_rect_width",600));
        self.ui.spinCapRectHeight.setValue(cfg_options.get("cap_rect_height",800));
        self.ui.comboClipMode.setCurrentIndex(cfg_options.get("cap_clip_mode",0));

        self.ui.spinFaceDetectThreshold.setValue(cfg_options.get("detect_threshold",0.5))
        self.ui.spinSharpen.setValue(cfg_options.get("sharp_value",20))

        #self.ui.spinInnerBlur.setValue(cfg_options.get("inner_blur",15))
        self.ui.spinMaskDilate.setValue(cfg_options.get("out_blur",15))

        self.ui.spinOutputWidthLimit.setValue(cfg_options.get("width_limit",1200))
        self.ui.spinOutputHeightLimit.setValue(cfg_options.get("height_limit",900))

        self.ui.lineEditRecordDir.setText(cfg_options.get("record_dir",""))
        self.ui.lineEditLiveVideoFile.setText(cfg_options.get("video_file",""))
        self.ui.lineEditLiveImageFolder.setText(cfg_options.get("image_dir",""))
         
        self.ui.lineEditModelFilePath.setText(cfg_options.get("model_file",""))
        self.ui.lineEditModelFilePath.setToolTip(cfg_options.get("model_file",""))

        self.ui.checkMaskFromSrc.setChecked(cfg_options.get("mask_src",False))
        self.ui.checkMaskXSeg.setChecked(cfg_options.get("mask_xseg",True))

        #self.ui.spinGammaR.setValue(cfg_options.get("gamma_r",1.0))
        #self.ui.spinGammaG.setValue(cfg_options.get("gamma_g",1.0))
        #self.ui.spinGammaB.setValue(cfg_options.get("gamma_b",1.0))
         
        #self.ui.spinFaceOffsetX.setValue(cfg_options.get("offset_x",0.0))
        #self.ui.spinFaceOffsetY.setValue(cfg_options.get("offset_y",0.0))
        self.ui.spinFaceScaleX.setValue(cfg_options.get("face_scale_x",1.0))
        self.ui.spinFaceScaleY.setValue(cfg_options.get("face_scale_y",1.0))

        live_model_folder=cfg_options.get("model_folder","-")
        if os.path.exists(live_model_folder):
            sd.swap_model_dir=live_model_folder 

        xseg_model_folder=cfg_options.get("xseg_model_folder",sd.xseg_model_dir)
        if os.path.exists(xseg_model_folder):
            sd.xseg_model_dir=xseg_model_folder
        self.ui.lineEditXsegModelFolder.setText(sd.xseg_model_dir)
    

    def setOutputTopMost(self):
        try:
            import win32gui, win32con
            hwnd = win32gui.FindWindow(None, sd.OutputWindowName)
            if hwnd==0:
                return;
            left, top, right, bot = win32gui.GetWindowRect(hwnd)
            if self.ui.checkTopMost.isChecked(): 
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, left, top, right-left, bot-top, win32con.SWP_SHOWWINDOW) 
                print("[P]输出窗口设为置顶")
            else: 
                win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, left, top, right-left, bot-top, win32con.SWP_SHOWWINDOW) 
                print("[P]输出窗口取消置顶")
        except :
            pass

    def setReplacePreview(self):
        sd.ReplacePreview=self.ui.checkReplacePreview.isChecked()
        if sd.ReplacePreview is False:
            import win32gui, win32con
            hwnd = win32gui.FindWindow(None, "matting")
            if hwnd==0:
                return;
            win32gui.PostMessage(hwnd,win32con.WM_CLOSE,0,0)
        

    def OnConfigDataChange(self):
        self.ApplyConfigDataChange(True)

    def ApplyConfigDataChange(self,Save=True):

        sd.ReplaceBgNo=self.ui.radioBgReplaceNo.isChecked();
        sd.ReplaceBgGreen=self.ui.radioBgReplaceGreen.isChecked();
        sd.ReplaceBgAI=self.ui.radioBgReplaceAI.isChecked();
        sd.BgDownSample=self.ui.spinDownSample.value()
        
        sd.FrameRotateMode=self.ui.comboCameraRotate.currentIndex();
        sd.CameraFlipHorizontal=self.ui.checkBoxFlipCameraH.isChecked();
        sd.CamResolution=self.ui.comboCamResolution.currentText().split("*")

        sd.ResizeVideoFrame=self.ui.checkResizeFrame.isChecked();
        sd.ResizeVideoWidth=self.ui.spinFrameResizeWidth.value()
         
        left=self.ui.spinCapRectLeft.value();
        top=self.ui.spinCapRectTop.value();
        width=self.ui.spinCapRectWidth.value();
        height=self.ui.spinCapRectHeight.value();
        sd.CaptureRect=(left,top,width,height)
        sd.CaptureClipMode=self.ui.comboClipMode.currentIndex();

        sd.AlignResolution=self.ui.spinFaceReso.value();
        #sd.FastFaceDetect=self.ui.checkFastDetect.isChecked();
        sd.DetectWindowSize=self.ui.spinDetectWindowSize.value();
        sd.DetectThreshold=self.ui.spinFaceDetectThreshold.value();

        sd.MorphFactor=self.ui.spinMorphFactor.value()
        sd.VerifyThreshold=self.ui.spinFaceVerifyThreshold.value();
        
        sd.BeautyMode=self.ui.comboMergeMode.currentText();

        sd.ColorTransferMode=self.ui.comboColorTransMode.currentText();
        
        sd.NoFaceMode=self.ui.comboNoFaceMode.currentText(); 
        sd.FaceScaleX=self.ui.spinFaceScaleX.value();
        sd.FaceScaleY=self.ui.spinFaceScaleY.value();

        sd.PreviewMask=self.ui.checkPreviewMask.isChecked();
        sd.PreviewFrame=self.ui.checkPreviewFrame.isChecked();
        sd.PreviewAlignFace=self.ui.checkPreviewAlignFace.isChecked();
        sd.PreviewSwap=self.ui.checkPreviewSwap.isChecked();

        sd.MergeSuperResoEngine=self.ui.comboSuperResoEngine.currentText();
        sd.ShowMergeFace=self.ui.checkShowFaceMerge.isChecked();
        sd.MaskFromSrc=self.ui.checkMaskFromSrc.isChecked(); 
        sd.MaskDstXseg=self.ui.checkMaskXSeg.isChecked();
        sd.ShowAiMark=self.ui.checkShowAiMark.isChecked();
        sd.face_mask_erode=self.ui.spinMaskErode.value();
        sd.face_mask_dilate=self.ui.spinMaskDilate.value();
        sd.OutputType=self.ui.comboOutputType.currentIndex();
        sd.OutputWidthLimit=self.ui.spinOutputWidthLimit.value();
        sd.OutputHeightLimit=self.ui.spinOutputHeightLimit.value();
        sd.OutputTopMost=1 if self.ui.checkTopMost.isChecked() else 0;
        sd.OutputRotateMode=self.ui.comboOutputRotate.currentIndex();
        if self.ui.radioOutSizeOrigin.isChecked(): sd.OutputResizeMode=0
        if self.ui.radioOutFixWidth.isChecked(): sd.OutputResizeMode=1 
        if self.ui.radioOutFixHeight.isChecked():sd.OutputResizeMode=2 ;
        sd.RecordTempSave=self.ui.checkRecordTempSave.isChecked();
        
        
        
        if Save is True:
            options={}
            options["cam_id"]=self.ui.comboLiveCameraID.currentIndex();
            options["cam_rotate"]=self.ui.comboCameraRotate.currentIndex();
            options["cam_resolution"]=self.ui.comboCamResolution.currentIndex();

            options["cap_rect_left"]=self.ui.spinCapRectLeft.value();
            options["cap_rect_top"]=self.ui.spinCapRectTop.value();
            options["cap_rect_width"]=self.ui.spinCapRectWidth.value();
            options["cap_rect_height"]=self.ui.spinCapRectHeight.value();
            options["cap_clip_mode"]=self.ui.comboClipMode.currentIndex();

            options["detect_threshold"]=self.ui.spinFaceDetectThreshold.value();
            options["sharp_value"]=self.ui.spinSharpen.value();

            options["out_blur"]=self.ui.spinMaskDilate.value();
            options["width_limit"]=self.ui.spinOutputWidthLimit.value();
            options["height_limit"]=self.ui.spinOutputHeightLimit.value();

            options["video_file"]=self.ui.lineEditLiveVideoFile.text();
            options["image_dir"]=self.ui.lineEditLiveImageFolder.text();
            options["record_dir"]=self.ui.lineEditRecordDir.text();

            options["mask_src"]=self.ui.checkMaskFromSrc.isChecked();
            options["mask_xseg"]=self.ui.checkMaskXSeg.isChecked();
            options["model_file"]=self.ui.lineEditModelFilePath.text();
            options["model_folder"]=sd.swap_model_dir;
            options["xseg_model_folder"]= self.ui.lineEditXsegModelFolder.text()
            options["face_scale_x"]=self.ui.spinFaceScaleX.value();
            options["face_scale_y"]=self.ui.spinFaceScaleY.value();


             

            ToolKit.SavedConfigDataDict(options) 
            now=datetime.datetime.now().strftime('%H:%M:%S')
            print(f"[P]保存参数设置") 

    def onClickCameraSetting(self):
        self.vcap=self.mFrameCaptureThread.mCamVideoCapture
        if self.vcap is not None and self.vcap.isOpened():
            self.vcap.set(cv2.CAP_PROP_SETTINGS, 0)

    def ApplyCameraCaptureSetting(self):
        if(self.mFrameCaptureThread is None):
            return;
        camIdx=self.ui.comboLiveCameraID.currentIndex();
        input={};
        width,height=self.ui.comboCamResolution.currentText().split("*")
        input["width"]=int(width);
        input["cam_idx"]=camIdx;
        input["height"]=int(height);
        input["rotate"]=width;
        input["drive"]=self.ui.comboCamDrive.currentIndex();
        input["op"]="set_cam_source"
        self.mFrameCaptureThread.mEventQueue.put(input)
 
    def OnSlideFaceParam(self):
        self.ui.spinMalarThin.setValue(self.ui.sliderMalarThin.value())
        self.ui.spinCheekThin.setValue(self.ui.sliderCheekThin.value())
        self.ui.spinJawThin.setValue(self.ui.sliderJawThin.value())
        self.ui.spinFaceLong.setValue(self.ui.sliderFaceLong.value())
        self.ui.spinSmallMouth.setValue(self.ui.sliderSmallMouth.value())
        self.ui.spinEyeDistance.setValue(self.ui.sliderEyeDistance.value())
        self.ui.spinLowCheekWidth.setValue(self.ui.sliderLowCheekWidth.value())

        self.ui.spinFaceWhite.setValue(self.ui.sliderFaceWhite.value())
        self.ui.spinSharpen.setValue(self.ui.sliderSharpen.value())
        self.ui.spinFaceSmooth.setValue(self.ui.sliderFaceSmooth.value())

    def OnChangeFaceBeautyConfig(self):
        #sd.BeautyShapeEnable=self.ui.group_BeautyShape.isChecked()
        sd.WhiteValue=self.ui.spinFaceWhite.value()
        sd.SharpenValue=self.ui.spinSharpen.value(); 
        sd.SmoothValue=self.ui.spinFaceSmooth.value();        

        sd.MalarThinValue=float(self.ui.spinMalarThin.value())/100.0                  
        sd.JawThinValue=float(self.ui.spinJawThin.value())/100.0
        sd.CheekThinValue=float(self.ui.spinCheekThin.value())/100.0
        sd.FaceLongValue=float(self.ui.spinFaceLong.value())/100.0
        sd.MouthSmallValue=float(self.ui.spinSmallMouth.value())/100.0
        sd.EyeDistanceValue=float(self.ui.spinEyeDistance.value())/100.0
        sd.LowCheekThinValue=float(self.ui.spinLowCheekWidth.value())/100.0  

        self.ui.sliderFaceWhite.setValue(self.ui.spinFaceWhite.value())
        self.ui.sliderFaceSmooth.setValue(self.ui.spinFaceSmooth.value())
        self.ui.sliderSharpen.setValue(self.ui.spinSharpen.value())
         
        self.ui.sliderMalarThin.setValue(self.ui.spinMalarThin.value())
        self.ui.sliderCheekThin.setValue(self.ui.spinCheekThin.value())
        self.ui.sliderJawThin.setValue(self.ui.spinJawThin.value())
        self.ui.sliderFaceLong.setValue(self.ui.spinFaceLong.value())
        self.ui.sliderSmallMouth.setValue(self.ui.spinSmallMouth.value())
        self.ui.sliderEyeDistance.setValue(self.ui.spinEyeDistance.value())
        self.ui.sliderLowCheekWidth.setValue(self.ui.spinLowCheekWidth.value())


    def StartPlayVideo(self):
        video_file=self.ui.lineEditLiveVideoFile.text();
        if os.path.exists(video_file)==False:
            ToolKit.ShowWarningError(None,"错误","[P]视频文件不存在")
            return
        if(self.mFrameCaptureThread is None):
            ToolKit.ShowWarningError(None,"错误","[P]播放线程尚未启动，请先开启")
            return
        self.mFrameCaptureThread.mLastPlayTime=time.time();
        self.mFrameCaptureThread.PlayToggle=True;

        self.mFrameCaptureThread.setMediaSource("video",0,self.ui.lineEditLiveVideoFile.text());

    def PausePlayVideo(self):
        self.mFrameCaptureThread.PlayToggle=not self.mFrameCaptureThread.PlayToggle;

    def StopPlayVideo(self):
        pass;


    



    def PreviewToggleClick(self):
        self.ApplyConfigDataChange();
        self.mSwapFaceThread.ShowPreview=not self.mSwapFaceThread.ShowPreview;

    def PlayJusmToHead(self):
        if self.mFrameCaptureThread.mFileVideoCapture is None:
            return;
        #self.mFrameCaptureThread.mFileVideoCapture.set(1,0)
        self.mFrameCaptureThread.ForcePlayVideoToFrame(0)

    def PlayJusmToEnd(self):
        if self.mFrameCaptureThread.mFileVideoCapture is None:
            return;
        frame_count=self.mFrameCaptureThread.mFileVideoCapture.get(7);
        self.mFrameCaptureThread.ForcePlayVideoToFrame(frame_count-5)
        #self.mFrameCaptureThread.mFileVideoCapture.set(1,frame_count-5);

    def MoveVideoPlayFrame(self):
        if self.mFrameCaptureThread.mFileVideoCapture is None:
            return;
        self.mFrameCaptureThread.PlayToggle=False;
        frame=self.ui.sliderVideoPlayFrame.value()
        #self.mFrameCaptureThread.mFileVideoCapture.set(1,int(frame));
        self.mFrameCaptureThread.ForcePlayVideoToFrame(frame)
 
    def ShowVideoPlayInfo(self):
        pass;


 
    def OnbtnImageParseClick(self):
        #mCamVideoCapture=cv2.VideoCapture(lineEditLiveImageFile.text())
        #img=cv2.imread(self.ui.lineEditLiveImageFile.text())
        import numpy as np;
        img=cv2.imdecode(np.fromfile(self.ui.lineEditLiveImageFile.text(),dtype=np.uint8),cv2.IMREAD_COLOR)
        if img is None:
            print("加载图片失败");
            return;
        suc,detect_face,align_face=ToolKit.getAlignedFace(img);
        self.ShowImageInLabel(detect_face,self.ui.FaceMarkLabel,True)
       

    def btnCloseLiveSwapClick(self):
        self.mUpdateImageTimer.stop(); 
        self.mFrameCaptureThread.stop();
        self.mSwapFaceThread.stop();
        cv2.destroyAllWindows();

   
    def OnOpenImageFolderClick(self):
        folder=self.ui.lineEditLiveImageFolder.text();  
        try:           
            os.startfile(folder)
        except:
            QMessageBox.information(None,"错误","打开图片文件夹失败,请检查文件夹是否存在");

    def OnBrowseImageFolderClick(self):
        folder=QFileDialog.getExistingDirectory(None,"选择图片文件夹",self.ui.lineEditLiveImageFolder.text());
        if(len(folder)>2):
            self.ui.lineEditLiveImageFolder.setText(folder);
            self.LoadImageList()
           
    def LoadImageList(self):
        folder=self.ui.lineEditLiveImageFolder.text();
        if os.path.exists(folder):
            sd.ImageList.clear();
            sd.CurrImageIdx=0;
            sd.ImageList=[file for file in os.listdir(folder) if ".jpg" in file ]
            sd.ImageCount=len(sd.ImageList);
            print(f"Load {sd.ImageCount} images from dir {folder}")
            
    def pushButtonPreImageClick(self):
        self.mFrameCaptureThread.SrcType="image"
        if len(sd.ImageList)==0:
            self.LoadImageList()
            return;
        if sd.CurrImageIdx>1:
            sd.CurrImageIdx=sd.CurrImageIdx-1;
        self.LoadImageFileAsFrame();

    def pushButtonNextImageClick(self):
        self.mFrameCaptureThread.SrcType="image"
        if len(sd.ImageList)==0:
            self.LoadImageList()
            #return;
        if(len(sd.ImageList)==0): return;
        sd.CurrImageIdx=(sd.CurrImageIdx+1)%len(sd.ImageList);
       
        self.LoadImageFileAsFrame();
        pass

    def LoadImageFileAsFrame(self):
        
        image_count=len(sd.ImageList)
        if image_count==0:
            ToolKit.ShowWarningError(None,"","图片文件列表为空")
            self.current_frame_image=None;
            return;
        if sd.CurrImageIdx>=image_count:
            sd.CurrImageIdx=0;
            return;
        folder=self.ui.lineEditLiveImageFolder.text();
        FileName=sd.ImageList[sd.CurrImageIdx];
        FullPath=f"{folder}/{FileName}";
        
        if os.path.exists(FullPath) is False:
            return;
        try:
            #dfl=DFLJPG.load(FullPath)
            #frame_image=dfl.get_img()
            frame_image=cv2ex.cv2_imread(FullPath)
            if frame_image is None:
                ToolKit.ShowWarningError(None,"错误","读取图片失败")
                return;
            h,w,c=frame_image.shape
            if(h>1080 or w>1080):
                frame_image=cv2.resize(frame_image,(w//2,h//2))
            sd.mRawFrameImageInt=frame_image;
            sd.mRawFrameImageFloat=ImageProcessor(frame_image).to_ufloat32().get_image('HWC') 
        
            self.ui.lineEditLiveImageFile.setText(FileName)
            self.ui.label_image_index.setText(f"{sd.CurrImageIdx}/{sd.ImageCount}")
            self.ui.label_image_reso.setText(f"{w}x{h} px")
        except :
            return
       
        
        

    def btnBrowseMediaFileClick(self):
        file,ok=QFileDialog.getOpenFileName(None,"选择视频文件",self.ui.lineEditLiveVideoFile.text(),'video files (*.mp4 *.wmv *.avi)');
        if ok:
            self.ui.lineEditLiveVideoFile.setText(file);
            self.mFileVideoCapture=cv2.VideoCapture(file)
            self.mPlayPosTime=0.0;
            self.mPlayFrameNum=0;

            self.mPlayPosTime=self.mFileVideoCapture.get(0) 	#视频文件的当前位置（播放）以毫秒为单位
            self.mPlayFrameNum=self.mFileVideoCapture.get(1) 	#基于以0开始的被捕获或解码的帧索引
            self.mFileVideoCapture.get(2) 	#视频文件的相对位置（播放）：0=电影开始，1=影片的结尾。
            self.mVideoWidth =self.mFileVideoCapture.get(3) 	#在视频流的帧的宽度
            self.mVideoHeight=self.mFileVideoCapture.get(4) 	#在视频流的帧的高度
            self.mVideoFPS=self.mFileVideoCapture.get(5) 	#帧速率
            self.mFileVideoCapture.get(6) 	#编解码的4字-字符代码
            self.mVideoFrameCount=self.mFileVideoCapture.get(7) 	#视频文件中的帧数
            print(f"[P]播放帧:{self.mPlayFrameNum}/{self.mVideoFrameCount},FPS:{self.mVideoFPS}")

    def btnBrowseBgClick(self):
        file,ok=QFileDialog.getOpenFileName(None,"选择背景文件","",'image files (*.jpg)');
        if ok:
            BacgkgroundImage=cv2ex.cv2_imread(file)
            ToolKit.ShowImageInLabel(BacgkgroundImage,self.ui.AllDetectFacesLabel)
            self.mFrameCaptureThread.MatEngine.SetBackgroundImage(BacgkgroundImage)


    def OnBrowseXsegFolderClick(self):
        xseg_folder=QFileDialog.getExistingDirectory(None,"选择分割模型文件夹",self.ui.lineEditXsegModelFolder.text());
        if(len(xseg_folder)>2):
            self.ui.lineEditXsegModelFolder.setText(xseg_folder);
            xseg_model_file=f"{xseg_folder}/XSeg_256.npy"
            if os.path.exists(xseg_model_file)==False:
                QMessageBox.information(None,"错误",f"{xseg_model_file}遮罩模型文件不存在");
            else:
                self.ApplyConfigDataChange(True)
                QMessageBox.information(None,"提示","已经保存新的遮罩模型路径，将在下次程序启动时生效");
            #sd.xseg_model_dir=os.path.abspath(xseg_folder)
            #print(sd.xseg_model_dir)
            


    def btnReloadXsegModelClick(self):
        self.mMergePreviewThread.reload_xseg_model(self.ui.lineEditXsegModelFolder.text())

    def ChangeInferDevice(self):
        #self.mSwapFaceThread.SetDetectEngine(self.ui.comboInferDevices.currentText())
        pass

    def UpdateListForModelFiles(self):  
        self.UpdateModelListFromFolder();
        return;

    def btnSubscribeModelClick(self):
        self.mServerRequestThread=ServerRequestThread(app="live",delay=0.001,mode="create_model")
        self.mServerRequestThread.mAlertSignal.connect(self.AuthCheckAlert)
        self.mServerRequestThread.start();

    def UpdateModelListFromFolder(self):
        
        #print(f"model folder {sd.swap_model_dir}")
        if(os.path.exists(sd.swap_model_dir) is False):
            print("Model Folder do not exists ");
            return; 
        #files=pathex.get_file_paths_str(sd.swap_model_dir)
        files=[os.path.basename(file) for file in os.listdir(sd.swap_model_dir) if  file.endswith(".dfm") or file.endswith(".vtfm") ]
        #print(files)

        slm=QStringListModel()
        slm.setStringList(files)
        self.ui.tableViewModelLib.setModel(slm) 
    
    def OnSetSwapModelFolderClick(self):
        folder=QFileDialog.getExistingDirectory(None,"选择文件夹","");
        if(len(folder)>2):
            sd.swap_model_dir=folder;
            self.UpdateListForModelFiles();
            self.ApplyConfigDataChange(Save=True)

    

    def btnOpenModelFolderClick(self):
        folder=os.path.abspath(sd.swap_model_dir)	
        try:           
            os.startfile(folder)
        except:
            QMessageBox.information(None,"错误","打开模型文件夹失败,请检查文件夹是否存在");

    def OnChangeLiveModelFileClick(self):  
        if self.mWholeFaceSwapThread is None:
            return;
        model_name = self.ui.tableViewModelLib.selectionModel().selectedIndexes()[0].data()
        #model_name=self.ui.comboLiveModelList.currentText()
        model_file_fullpath=sd.swap_model_dir+"/"+model_name;
        if os.path.isfile(model_file_fullpath) is False:
            return;
        
        #---检查模型文件是否存在
        if(os.path.exists(model_file_fullpath)==False):
            QMessageBox.information(None,"错误",f"模型文件{model_file_fullpath}不存在，请检查");
            return;
        #--检查模型使用权限
        isValid,Mode,RequestCode,RightToken=LiveSwapModel.check_model_authorized(model_file_fullpath)
        if isValid==False:
            if "machine" in Mode:
                print(f"请求码{RequestCode}，复制后到启动器授权")
                text, ok=QInputDialog.getText(None, f'授权码验证(可在启动器请求授权码）', f'请求码{RequestCode},请输入授权码：') 
                if ok is False:
                    return
                if ok:
                    if text!=RightToken:
                        QMessageBox.information(None,"错误",f"授权码错误");
                        return
                    else:
                        LiveSwapModel.write_model_metadata(model_file_fullpath,"auth_token",text)
            if "pwd" in Mode:
                pwd, ok=QInputDialog.getText(None, f'模型密码验证', f'请输入模型密码：')
                if ok is False:
                    return
                input_pwd_token=auth.get_md5_hash_str(pwd+"ppp")
                if input_pwd_token!=RightToken:
                    QMessageBox.information(None,"错误",f"模型密码错误");
                    return          

        self.ui.lineEditModelFilePath.setText(model_name)
        self.ui.lineEditModelFilePath.setToolTip(model_file_fullpath)
        #self.mSwapFaceThread.SetModelFile(model_file_fullpath)
        self.mWholeFaceSwapThread.SetModelFile(model_file_fullpath)
        self.ApplyConfigDataChange()

    def OnBrowseImageFileClick(self):
        file,ok=QFileDialog.getOpenFileName(None,"选择图片文件","");
        if ok:
            self.ui.lineEditLiveImageFile.setText(file);

    def OnBtnOpenLiveMediaFolderClick(self):
        video_file=self.ui.lineEditLiveVideoFile.text();  
        folder=os.path.dirname(video_file)	
        try:           
            os.startfile(folder)
        except:
            QMessageBox.information(None,"错误","打开模型文件夹失败,请检查文件夹是否存在");
    
    def UpdateWindowListForUi(self):
        
        sd.hWnd_info_list=cap.get_all_windows()
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
        self.mFrameCaptureThread.SrcType="capture"
        #cap.show_test_cap_window()

    def OnChangeCaptureRect(self):
        left=self.ui.spinCapRectLeft.value();
        top=self.ui.spinCapRectTop.value();
        width=self.ui.spinCapRectWidth.value();
        height=self.ui.spinCapRectHeight.value();
        self.mFrameCaptureThread.CaptureRect=(left,top,width,height)
    #----- 设置身份筛选图片
    #-----------------------------------
    StandardIdentityFaceImg=None;
    def btnBrowseIdentityFilterFaceImageClick(self):
        file,ok=QFileDialog.getOpenFileName(None,"选择人脸文件","",'image files (*.jpg *.png )');
        if ok:
           frame_image= cv2ex.cv2_imread(file);
           face_imgs,align_imgs,_,_=ToolKit.GetDetectAndAlignFace(frame_image,256,detector="yolo")
           if len(face_imgs)==0:
               ToolKit.ShowWarningError(None,"设置失败","未能从图片中提取到人脸")
               return;
           self.StandardIdentityFaceImg=align_imgs[0];
           facenet.set_standard_face_img(self.StandardIdentityFaceImg);
           ToolKit.ShowImageInLabel(self.StandardIdentityFaceImg,self.ui.SwapTargetFaceLabel,bgr_cvt=True,scale=True)

    #-----提取选择要替换的人脸
    def btnAddFilterFaceClick(self):
        if sd.mRawFrameImageInt is None:
            ToolKit.ShowWarningError(None,"","原画面不存在")
            return 

        from PyQt5.QtWidgets import QSpinBox,QDialogButtonBox,QDialog,QLabel,QPushButton,QGroupBox,QComboBox
        createDialog=QDialog()
        createDialog.setWindowFlags( Qt.WindowCloseButtonHint)
        createDialog.setWindowTitle("添加参考人脸")
        formLayout=QFormLayout(createDialog)
        FacesLabel=QLabel()
        combo=QComboBox()
        combo.addItems(["第1张","第2张","第3张","第4张"])
        buttonBox=QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(createDialog.accept)
        buttonBox.rejected.connect(createDialog.reject)

        FacesLabel.setMinimumSize(700,256)
        formLayout.addRow("人脸列表",FacesLabel)
        formLayout.addRow("选择对象",combo)   
        formLayout.addRow(buttonBox)

        sd.DetectRawFaces,sd.DetectAlignFaces,_,_=ToolKit.GetDetectAndAlignFace(sd.mRawFrameImageInt);
        if sd.DetectAlignFaces is not None:
            if len(sd.DetectAlignFaces)>0:
                all_align_img=np.hstack(sd.DetectAlignFaces)
                ToolKit.ShowImageInLabel(all_align_img,FacesLabel) 

        if (createDialog.exec() == QDialog.Accepted):
            if sd.DetectAlignFaces is None :
                ToolKit.ShowWarningError(None,"错误","提取目标人脸为空，请先手动提取人脸列表")
                return;
            if len(sd.DetectAlignFaces)==0:
                ToolKit.ShowWarningError(None,"错误","提取目标人脸数为0")
                return;
            face_num=len(sd.DetectAlignFaces)
            choose_idx=combo.currentIndex();

            if(choose_idx>face_num):
                ToolKit.ShowWarningError(None,"错误","超过提取人脸总数")
                return;
            target_face=sd.DetectAlignFaces[choose_idx-1].copy()
            facenet.add_ref_face_img(target_face);
            ToolKit.ShowImageInLabel(facenet.getRefImagesMerge(),self.ui.SwapTargetFaceLabel,bgr_cvt=True,scale=True)

    def btnClearIdentityFilterFaceClick(self):
        self.ui.SwapTargetFaceLabel.clear()
        facenet.clear_standard_face()
      
    def btnAddLiveRefFaceClick(self):
        if sd.DetectAlignFaces is None :
            ToolKit.ShowWarningError(None,"错误","提取目标人脸为空，请先手动提取人脸列表")
            return;
        if len(sd.DetectAlignFaces)==0:
            ToolKit.ShowWarningError(None,"错误","提取目标人脸数为0")
            return;

        face_num=len(sd.DetectAlignFaces)
        choose_idx=1;
        if self.ui.radioChoose2.isChecked(): choose_idx=2;
        if self.ui.radioChoose3.isChecked(): choose_idx=3;

        if(choose_idx>face_num):
            ToolKit.ShowWarningError(None,"错误","超过提取人脸总数")
            return;

        target_face=sd.DetectAlignFaces[choose_idx-1].copy()
        facenet.add_ref_face_img(target_face);
        ToolKit.ShowImageInLabel(facenet.getRefImagesMerge(),self.ui.SwapTargetFaceLabel,bgr_cvt=True,scale=True)
    
    def btnLivePopRefClick(self):
        facenet.del_last_face_img()
        RefImagesMerge=facenet.getRefImagesMerge()
        if RefImagesMerge is not None:
            ToolKit.ShowImageInLabel(facenet.getRefImagesMerge(),self.ui.SwapTargetFaceLabel,bgr_cvt=True,scale=True)      
        else:
            self.ui.SwapTargetFaceLabel.clear()

    def OnBrowseRecordDirClick(self):
        folder=QFileDialog.getExistingDirectory(None,"选择視頻录制保存文件夹","");
        if(len(folder)>2):
            self.ui.lineEditRecordDir.setText(folder);
            self.ApplyConfigDataChange()

    def OnOpenRecordDirClick(self):
        folder=self.ui.lineEditRecordDir.text();  
        try:           
            os.startfile(folder)
        except:
            QMessageBox.information(None,"错误","打开視頻录制保存文件夹失败,请检查文件夹是否存在");


    def OnRecordStartStopClick(self):
        btnText=self.ui.btnStartWriteVideo.text()
        if "停止" not in btnText:
            self.StartRecordVideo()
        else:
            self.StopRecordVideo();



    def StartRecordVideo(self):
        folder=self.ui.lineEditRecordDir.text();  
        fps=float(self.ui.comboWriteFPS.currentText())
        encoder=self.ui.comboWriteEncoder.currentText();
        if os.path.exists(folder) is False:
            QMessageBox.information(None,"错误","視頻录制保存文件夹路徑不存在");
            return

        import datetime
        file_name=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if encoder=="XVID":
            ext="avi"
        elif encoder=="MJPG":
            ext="mp4"
        elif encoder=="FLV1":
            ext="flv"
        else:
            ext="avi"
        video_full_path=f"{folder}/{file_name}.{ext}"
        audio_full_path=f"{folder}/{file_name}.wav"
        merge_out_path=f"{folder}/{file_name}_merge.mp4"
        if sd.RecordTempSave is False:
            audio_full_path=f"{folder}/temp.wav"
            video_full_path=f"{folder}/temp.{ext}"
        sd.last_record_video_file=video_full_path;
        sd.last_record_audio_file=audio_full_path;
        fourcc = cv2.VideoWriter_fourcc(*encoder)

        th=threading.Thread(target=self.RecordThreadRun,args=(video_full_path,fourcc,fps))
        th.daemo=True;
        th.start();
        self.ui.btnStartWriteVideo.setText("■停止录制")

        rec_mode=self.ui.comboAudioRecMode.currentText()
        self.mAudioRecorder.start_audio_record(audio_full_path,rec_mode)


    WriteVideoFlag=True;
    def RecordThreadRun(self,video_full_path,fourcc,fps=24.0):
        if sd.mOutputImage is None:
                return
        
        h,w,c=sd.mOutputImage.shape;
        frame_should_gap_time=1.0/fps;
        
        #print(f"start recording video to {video_full_path}")
        out = cv2.VideoWriter(video_full_path,fourcc, fps, (w,h),True) 
        self.WriteVideoFlag=True;
        frame_num=0;
        last_frame_save_timetick=10.0;
        while(self.WriteVideoFlag==True):
            save_image=sd.mOutputImage.copy();
            save_image=ImageProcessor(save_image).to_uint8().get_image('HWC') 
            if save_image is None:
                continue
            #cv2.imshow("record",save_image)
            time_since_last_frame_save=time.time()-last_frame_save_timetick;
            if time_since_last_frame_save<frame_should_gap_time:
                time.sleep(frame_should_gap_time-time_since_last_frame_save);
            out.write(save_image) 
            last_frame_save_timetick=time.time()
            frame_num=frame_num+1;
            #cv2.waitKey(int(1000/fps))
            #print(f"write output  frame ,type={save_image.dtype},frame {frame_num}")
   
        #print("\nend recording video")
        out.release()


    def StopRecordVideo(self):
        self.WriteVideoFlag=False;
        self.ui.btnStartWriteVideo.setText("开始录制")
        self.mAudioRecorder.stopRecord()

        folder=self.ui.lineEditRecordDir.text(); 
        import datetime
        file_name=datetime.datetime.now().strftime("%m%d%H%M%S")
        merge_out_path=f"{folder}/{file_name}.mp4"

        th=threading.Thread(target=self.MuxRecordVideoAndAudioThread,args=(sd.last_record_video_file,sd.last_record_audio_file,merge_out_path))
        th.daemo=True;
        th.start();

    def MuxRecordVideoAndAudioThread(self,video_file,audio_file="",merge_out_path=""):
        if os.path.exists(video_file) is False:
            print(f"video file {video_file} not exist, mux failed")
            return;
        from kit import VideoAudioKit
        VideoAudioKit.mux_video_audio(video_file,audio_file,merge_out_path)

    def btnSaveOutputImageClick(self):
        folder=self.ui.lineEditRecordDir.text(); 
        if os.path.exists(folder) is False:
            ToolKit.ShowWarningError(None,"错误","保存路径错误、文件夹不存在")
            return
        if sd.mOutputImage is None:
            ToolKit.ShowWarningError(None,"错误","输出图片为空")
            return
        import datetime
        file_name=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        save_path=f"{folder}/S_{file_name}.jpg"
        print(f"[P]保存图片：{save_path}")
        from core import cv2ex
        img=sd.mOutputImage.copy()
        cv2ex.cv2_imwrite(save_path,img*255)
        #ToolKit.ShowWarningError(None,"成功","图片已经保存到目录")
       
    #----快捷键进入隐私模式
    def btnTogglePrivateModeClick(self):
        if hasattr(self,"PrivateMode") is False:
            self.PrivateMode=False;
        self.PrivateMode=not self.PrivateMode;
        if self.PrivateMode is True:
            self.ui.checkPreviewFrame.setChecked(False)
            self.ui.checkPreviewAlignFace.setChecked(False)
            self.ui.AllDetectFacesLabel.clear();
            self.ui.SwapTargetFaceLabel.clear();
            slm=QStringListModel()
            slm.setStringList(["**********","***********"])
            self.ui.tableViewModelLib.setModel(slm) 
        if self.PrivateMode is False:
            self.ui.checkPreviewFrame.setChecked(True)
            self.ui.checkPreviewAlignFace.setChecked(True)
            self.UpdateListForModelFiles()

        
    def UpdateFrameSourcePreview(self):
        if sd.mRawFrameImageInt is None:
            self.ui.FrameImageLabel.clear();
            self.ui.FrameImageLabel.setText(f"No Frame Image")
            return;
        h,w,c=sd.mRawFrameImageInt.shape;
        self.ui.LabelCapTime.setText(f"帧源采集:{round(sd.FrameCapUseTime)}ms")
        self.ui.lineEditFrameInfo.setText(f"{w}x{h} px")
        self.ui.lineEditBeautyInfo.setText(f"美颜调型:{sd.FaceBeautyTime} ms")

        if  sd.PreviewFrame:
            ToolKit.ShowImageInLabel(sd.mRawFrameImageInt,self.ui.FrameImageLabel,True)
        else:
            self.ui.FrameImageLabel.clear()
            self.ui.FrameImageLabel.setText("Frame Preview Close")

    def UpdateDetectAlignPreview(self):
        if sd.ExistTargetFace is False:
            self.ui.FaceAlignPicLabel.clear();
            self.ui.FaceAlignPicLabel.setText("No Face Detect");
            self.ui.FaceSwapPicLabel.clear();
            self.ui.LabelDetectTime.setText(f"检测:-- ms")
            #self.ui.LabelAlignTime.setText(f"标记:-- ms")
            self.ui.LabelSwapTime.setText(f"交换:-- ms")
        if sd.ExistTargetFace is True:
            if sd.mAlignFaceImage is not None :
                if sd.PreviewAlignFace:
                    ToolKit.ShowImageInLabel(sd.mAlignFaceImage,self.ui.FaceAlignPicLabel,True)
                else:
                    self.ui.FaceAlignPicLabel.clear()
                    self.ui.FaceAlignPicLabel.setText("Align Preview Close")
                self.ui.LabelDetectTime.setText(f"检测标记:{round(sd.DetectFaceTime)}+{round(sd.AlignFaceTime)}ms")
                #self.ui.LabelAlignTime.setText(f"标记:{round(sd.AlignFaceTime)} ms")
             
    def UpdateSwapOutputPreview(self):
        if sd.ExistTargetFace is False:
            self.ui.FaceSwapPicLabel.clear();
            self.ui.LabelSwapTime.setText(f"交换:-- ms")

        if sd.ExistTargetFace is True:
            if sd.mSwapOutImageForPreview is not None:
                if sd.PreviewSwap:
                    ToolKit.ShowImageInLabel(sd.mSwapOutImageForPreview,self.ui.FaceSwapPicLabel,True)
                else:
                    self.ui.FaceSwapPicLabel.clear()
                    self.ui.FaceSwapPicLabel.setText("Swap Preview Close")
            #if sd.mMergeImageInt is not None:
            #    ToolKit.ShowImageInLabel(sd.mMergeImageInt,self.ui.FaceMergePicLabel,True) 
            
            self.ui.LabelSwapTime.setText(f"风格转换:{round(sd.SwaFaceTime)}ms")
        
        
   
    def UpdateMergeOutputPreview(self):

        if sd.mOutputImage is None:
            print("[P]sd.mOutputImage is None")
            return
        self.ui.LabelMergeTime.setText(f"图像合成:{round(sd.MergeFaceTime)} ms") 
        #self.ui.lineEditMergeDetail.setText(f"遮罩:{round(sd.MaskXsegTime)} ms") 
        
        if sd.mMergeFaceImage is not None:
            ToolKit.ShowImageInLabel(sd.mMergeFaceImage,self.ui.FaceMergePicLabel,True)
        else:
            self.ui.FaceMergePicLabel.clear();

        time_use=(time.time()-sd.last_output_time_tick)*1000
        sd.OutputGapTime=round(time_use);
        sd.last_output_time_tick=time.time() 

        self.ui.groupBox_OutputPreview.setTitle(f"输出间隔:{sd.OutputGapTime} ms")
        #self.ui.lineEditMergeDetail.setText(f"遮罩计算:{sd.MaskXsegTime} ms")

        if sd.OutputRotateMode==1:
            sd.mOutputImage=cv2.rotate(sd.mOutputImage,cv2.ROTATE_90_CLOCKWISE) 
        elif sd.OutputRotateMode==2:
            sd.mOutputImage=cv2.rotate(sd.mOutputImage,cv2.ROTATE_90_COUNTERCLOCKWISE) 
        elif sd.OutputRotateMode==3:
            sd.mOutputImage=cv2.rotate(sd.mOutputImage,cv2.ROTATE_180) 
 
        #---输出窗口预览图
        height,width=sd.mOutputImage.shape[0:2];
        
        if sd.OutputResizeMode==1:
            new_width=sd.OutputWidthLimit;
            r=new_width/width;
            sd.mOutputResizeImage=cv2.resize(sd.mOutputImage,(0,0),fx=r,fy=r)
            cv2.imshow(sd.OutputWindowName,sd.mOutputResizeImage)
            
        elif sd.OutputResizeMode==2:
            new_height=sd.OutputHeightLimit;
            r=new_height/height;
            sd.mOutputResizeImage=cv2.resize(sd.mOutputImage,(0,0),fx=r,fy=r)
            cv2.imshow(sd.OutputWindowName,sd.mOutputResizeImage)
        elif sd.OutputResizeMode==0:
            cv2.imshow(sd.OutputWindowName,sd.mOutputImage)
        