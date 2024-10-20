#coding=utf-8
# cython:language_level=3
import threading,cv2,time,os
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog,QMessageBox;
 
from core.xlib.image import ImageProcessor
from core.xlib.image import color_transfer as lib_ct
import kit.ShareData as sd
import kit.ToolKit as ToolKit
from  kit.LiveSwapModel import LiveSwapModel
from core.xlib.face import FRect,FLandmarks2D,ELandmarks2D
from kit.FaceEngines import FaceEngines
#from facelib.facenet import facenet

class SwapFaceThread(QThread):

    mSwapEndUpdateSignal=pyqtSignal();
    ModelFilename=""; ChangeDetectEngineFlag=False; DetectEngineName="Yolo";
    DetectModelSelect=0;  ChangeSwapModelFlag=False;    SwapNetModel=None; SwapImageMode=1; 

    def __init__(self): 
        super(SwapFaceThread, self).__init__()
        self.last_run_timetick=0;
         

    def run(self):
         
        while(True):
             
             #---更换Swap模型
            if self.SwapNetModel is None:
                if os.path.exists(self.ModelFilename):
                    self.SwapNetModel=LiveSwapModel(self.ModelFilename,"cuda")
                else:
                    time.sleep(0.05)
                    continue

            if self.ChangeSwapModelFlag==True:
                if self.SwapNetModel is not None:
                    import gc
                    del self.SwapNetModel
                    gc.collect()
                if os.path.exists(self.ModelFilename):
                    self.SwapNetModel=LiveSwapModel(self.ModelFilename,"cuda")
                    self.ChangeSwapModelFlag=False;
                     
            now=time.time()
            over_time=0.04-(now-self.last_run_timetick)
            self.last_run_timetick=time.time()
            if over_time>0:
                time.sleep(over_time)
            #----更换人脸
            self.SwapFace();
         

    
         

    def SwapFace(self):
        

        swap_start_time=time.time();
        if sd.mFrameImageFloat is None:
            time.sleep(0.02)
            return

        if sd.face_ulmrks_inSrc is None :
            #ToolKit.ShowWarningError(None,"错误","检测人脸队列为空")
            print(f"[P]检测人脸为空:{time.time()}",end='\r')
            time.sleep(0.02)
            return

          
        

          
        face_align_img=sd.DetectFaceInfo.get("face_align_img",None);
        frame_image_src_float=sd.DetectFaceInfo.get("frame_image_src_float",None)
        aligned_to_source_uni_mat=sd.DetectFaceInfo.get("aligned_to_source_uni_mat",None)

        face_swap_input_img=face_align_img;
        if face_swap_input_img is None:
            #self.mSwapEndUpdateSignal.emit()
            time.sleep(0.02)
            return
 
        if self.SwapNetModel is None:
            #print("[P]换脸模型未加载",end='\r')
            time.sleep(0.05);
            return;


        #----- 换脸模型转换人脸
        face_swap_out_img, src_celeb_mask_img, src_swap_mask_img = self.SwapNetModel.convert(face_swap_input_img,factor=1.0)
        face_swap_out_img, src_celeb_mask_img, src_swap_mask_img = face_swap_out_img[0], src_celeb_mask_img[0], src_swap_mask_img[0]
        
        face_swap_out_ip=ImageProcessor(face_swap_out_img).ch(3).to_ufloat32();

        if sd.PostSharpenValue !=0:
            face_swap_out_ip.gaussian_sharpen(sigma=1.0, power=sd.PostSharpenValue)

        if(sd.PostGammaR!=1.0 or sd.PostGammaG!=1.0 or sd.PostGammaB!=1.0):
            face_swap_out_ip.gamma(sd.PostGammaR,sd.PostGammaG,sd.PostGammaB)
        if(sd.PostBright!=1.0):
            face_swap_out_ip.bright(sd.PostBright)
        #face_swap_out_img=face_swap_out_ip
        face_swap_out_img=face_swap_out_ip.get_image('HWC') ;
        
         
        
        sd.FaceSwapInfo["face_swap_out_img"]=face_swap_out_img
        sd.FaceSwapInfo["src_swap_mask_img"]=src_swap_mask_img
        sd.FaceSwapInfo["frame_image_src_float"]=frame_image_src_float
        sd.FaceSwapInfo["face_align_img"]=face_align_img
        sd.FaceSwapInfo["aligned_to_source_uni_mat"]=aligned_to_source_uni_mat

        swap_end_time_tick=time.time();
        sd.SwaFaceTime=(swap_end_time_tick-swap_start_time)*1000;

        #sd.mSwapOutImageForPreview=face_swap_out_img
        self.mSwapEndUpdateSignal.emit()
        pass;

    def SetModelFile(self,filename):
            self.ChangeSwapModelFlag=True;
            self.ModelFilename=filename;
            if(os.path.exists(filename) is False):
                filename=sd.swap_model_dir+"/"+filename;
            if os.path.exists(filename):
                self.ModelFilename=filename;
            print(f"[P]设置模型文件:{self.ModelFilename}",end='\r');
            #self.SwapNetModel=LiveSwapModel(self.ModelFilename,get_available_devices()[0])

    def SetDetectEngine(self,engine='Yolo'):
        self.ChangeDetectEngineFlag=True;
        self.DetectEngineName=engine;
        print(f"[P]变更人脸检测引擎:{engine}");