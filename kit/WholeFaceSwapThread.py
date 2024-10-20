#coding=utf-8
# cython:language_level=3

import threading,cv2,time,queue, traceback,os, random 

from PyQt5.QtWidgets import QFileDialog,QMessageBox;
from PyQt5.QtCore import QThread, pyqtSignal
import kit.ShareData as sd
from kit.FaceEngines import FaceEngines
from  kit.LiveSwapModel import LiveSwapModel
import numpy as np;
import numexpr as cne
from core.leras import nn
from core.xlib.image import color_transfer as lib_ct
from core.xlib.image.ImageProcessor import ImageProcessor  
import kit.ToolKit as ToolKit
import kit.screen_cap as cap
from kit.FaceBeautify import FaceBeautify
import facelib.SuperReso.gpen as gpen
import facelib.SuperReso.CodeFormerRun as CodeFormer

class WholeFaceSwapThread(QThread):
    mMergeEndSignal=pyqtSignal();
    mDetectAlignEndSignal=pyqtSignal();
    mSwapEndUpdateSignal=pyqtSignal();
    mEventQueue=queue.Queue()
    mLastMergeTimeTick=1.0;
    ChangeSwapModelFlag=False;    SwapNetModel=None;ModelFilename="";
    main_ctrl_event=threading.Event()
    seg_ctrl_event=threading.Event()

    def __init__(self):
        super(WholeFaceSwapThread, self).__init__()
 

    def RealtimeSegThreadExec(self): 
        print("启动实时遮罩计算线程")
        while(True):
            if sd.MaskDstXseg is False:
                self.main_ctrl_event.set()
                time.sleep(0.05)
                continue
            #self.seg_ctrl_event.wait()
            xseg_start_time=time.time()
            face_swap_input_img=sd.DetectFaceInfo.get("face_align_img",None);
            if face_swap_input_img is None:
                time.sleep(0.05)
                continue;
            sd.mFaceSegMask=FaceEngines.getXSegModelEngineTF(sd.xseg_model_dir).extract(face_swap_input_img)
            self.main_ctrl_event.set()

    def run(self):

        while(True):
            self.Detect_Align_Beautify_Face()


    def Detect_Align_Beautify_Face(self):
        ##---- 画面源没有         
        if sd.mRawFrameImageFloat is None :
            time.sleep(0.05)
            return;

        ##---- 检测人脸有没有  
        ToolKit.LiveDetectAndAlignFace();


        ##---- 处理没有检测到人脸的时候
        if sd.ExistTargetFace is False:
            #print("[P]没有检测到目标人脸",time.time(),end='\r')
            if "冻结" in sd.NoFaceMode:
                return;
            if "原画面" in sd.NoFaceMode:
                sd.mMergeFrameImage=sd.mRawFrameImageFloat;
                self.CombineImagesForOutput()
                self.mMergeEndSignal.emit()
                time.sleep(0.01)
                return;

        sd.BeautyShapeEnable=False;

        #------- 对全帧画面进行瘦脸
        if sd.BeautyMode=="frame" :
            time_beauty_start=time.time() 
            beauty_frame_image=FaceBeautify.DeformFaceImage_TpsDeformTorch(sd.mRawFrameImageFloat,sd.ldmrkCoordsSrc,malar_thin=sd.MalarThinValue,jaw_thin=sd.JawThinValue,
                                cheek_thin=sd.CheekThinValue,small_mouth=sd.MouthSmallValue,long_face=sd.FaceLongValue,
                                eye_distance=sd.EyeDistanceValue,low_cheek_thin=sd.LowCheekThinValue) 
            
            sd.mFrameImageFloat=beauty_frame_image;
            time_beauty_end=time.time()
            sd.FaceBeautyTime=round((time_beauty_end-time_beauty_start)*1000)
        else:
            sd.mFrameImageFloat=sd.mRawFrameImageFloat;
            
        
             
        #----- 没画面就退出
        if sd.mFrameImageFloat is None or sd.face_ulmrks_inSrc is None:
            print("[P]没有帧画面")
            return

        #-------重新从美形后的Src画面提取对齐人脸
        align_start_time=time.time();
        frame_image_src_float=sd.mFrameImageFloat        
        face_align_img, src_to_align_uni_mat,src_align_mat = sd.face_ulmrks_inSrc.cut(frame_image_src_float, coverage=sd.AlignCoverage, 
                        output_size=sd.AlignResolution,exclude_moving_parts=True)
        aligned_to_source_uni_mat = src_to_align_uni_mat.invert() 
        align_rect=sd.face_ulmrks_inSrc.get_FRect(coverage=sd.AlignCoverage)
        #cv2.rectangle(sd.mFrameImageFloat,(),(),(0,0,1))
        align_end_time=time.time()
        sd.AlignFaceTime=round((align_end_time-align_start_time)*1000)

        align_uni_lmrks,align_lmrks=FaceEngines.getFaceInsightEngineOnnx().extract_lmks_68(face_align_img.copy())

        #----- 对齐后的人脸特征点再次提取
        if sd.BeautyMode=="align" :
            time_beauty_start=time.time() 
            face_align_img=FaceBeautify.DeformFaceImage_TpsDeformTorch(face_align_img,align_lmrks,malar_thin=sd.MalarThinValue,jaw_thin=sd.JawThinValue,
                                    cheek_thin=sd.CheekThinValue,small_mouth=sd.MouthSmallValue,long_face=sd.FaceLongValue,
                                    eye_distance=sd.EyeDistanceValue,low_cheek_thin=sd.LowCheekThinValue) 
            time_beauty_end=time.time()
            sd.FaceBeautyTime=round((time_beauty_end-time_beauty_start)*1000)
        
        #--- 对提取的align人脸进行 美白、磨皮、锐化
        face_align_img=FaceBeautify.White_Dermabration_Face(face_align_img,align_lmrks,sd.WhiteValue,sd.SmoothValue) 
        fai_ip = ImageProcessor(face_align_img)
        if sd.SharpenValue !=0:
            fai_ip.gaussian_sharpen(sigma=1.0, power=float(sd.SharpenValue*0.05))   
        face_align_img=fai_ip.get_image("HWC")
        

        sd.mAlignFaceImage=face_align_img;
       
        

        #sd.FrameWriteLock.lock()
        sd.DetectFaceInfo["frame_image_src_float"]=frame_image_src_float
        sd.DetectFaceInfo["face_align_img"]=face_align_img
        sd.DetectFaceInfo["aligned_to_source_uni_mat"]=aligned_to_source_uni_mat

        #---- 实时遮罩线程计算遮罩开始
        self.seg_ctrl_event.set()

        self.mDetectAlignEndSignal.emit() 
        self.SwapFace()
        self.MergeFace()


    def SwapFace(self):
        if self.SwapNetModel is None:
                if os.path.exists(self.ModelFilename):
                    self.SwapNetModel=LiveSwapModel(self.ModelFilename,"cuda")
                else:
                    return

        if self.ChangeSwapModelFlag==True:
            if self.SwapNetModel is not None:
                import gc
                del self.SwapNetModel
                gc.collect()
            if os.path.exists(self.ModelFilename):
                self.SwapNetModel=LiveSwapModel(self.ModelFilename,"cuda")
                self.ChangeSwapModelFlag=False;
                      
        swap_start_time=time.time();
        face_align_img=sd.DetectFaceInfo.get("face_align_img",None);
        frame_image_src_float=sd.DetectFaceInfo.get("frame_image_src_float",None)
        aligned_to_source_uni_mat=sd.DetectFaceInfo.get("aligned_to_source_uni_mat",None)
        face_swap_input_img=face_align_img;
        if face_swap_input_img is None:
            #self.mSwapEndUpdateSignal.emit()
            time.sleep(0.02)
            return
        #----- 换脸模型转换人脸
        
        face_swap_out_img, src_celeb_mask_img, src_swap_mask_img = self.SwapNetModel.convert(face_swap_input_img,morph_factor=sd.MorphFactor)
        face_swap_out_img, src_celeb_mask_img, src_swap_mask_img = face_swap_out_img[0], src_celeb_mask_img[0], src_swap_mask_img[0]
        
        face_swap_out_ip=ImageProcessor(face_swap_out_img).ch(3).to_ufloat32();

        #----- 换后人脸超分辨
        if sd.MergeSuperResoEngine=="cf": 
            face_swap_out_img=CodeFormer.process(face_swap_out_img) 
        elif sd.MergeSuperResoEngine=="gpen":
            face_swap_out_img=gpen.process(face_swap_out_img)
        else: 
            face_swap_out_img=face_swap_out_ip.get_image('HWC') ;   
             
  
        
        sd.FaceSwapInfo["face_swap_out_img"]=face_swap_out_img
        sd.FaceSwapInfo["src_swap_mask_img"]=src_swap_mask_img
        sd.FaceSwapInfo["frame_image_src_float"]=frame_image_src_float
        sd.FaceSwapInfo["face_align_img"]=face_align_img
        sd.FaceSwapInfo["aligned_to_source_uni_mat"]=aligned_to_source_uni_mat

        swap_end_time_tick=time.time();
        sd.SwaFaceTime=(swap_end_time_tick-swap_start_time)*1000;

        self.mSwapEndUpdateSignal.emit()
        


    def MergeFace(self):
         #---- 从换脸进程获取数据
            
            FaceSwapInfo=sd.FaceSwapInfo
            if FaceSwapInfo is None :
                print("[P]没有人脸信息",end='\r')
                time.sleep(0.05)
                return
                        
            frame_image_src_float=FaceSwapInfo.get("frame_image_src_float",None)
            face_swap_input_img=FaceSwapInfo.get("face_align_img",None)
            aligned_to_source_uni_mat=FaceSwapInfo.get("aligned_to_source_uni_mat",None)
            face_swap_out_img=FaceSwapInfo.get("face_swap_out_img",None)
            src_swap_mask_img=FaceSwapInfo.get("src_swap_mask_img",None)
            if (frame_image_src_float is None) or (face_swap_out_img is None):
                time.sleep(0.04)
                return;

            #--避免Merge线程运行过快，消耗CPU资源
            now=time.time();
            merge_over_time=0.04-(now-self.mLastMergeTimeTick-0.04);
            if merge_over_time>0:
                time.sleep(merge_over_time)

            merge_start_time_tick=time.time()
            

            #--- 计算人脸遮罩
            face_height, face_width = face_swap_input_img.shape[0:2]
             
            if sd.MaskFromSrc is True:
                sd.mFaceSegMask= ImageProcessor(src_swap_mask_img).ch(3).to_ufloat32().get_image('HWC') 
            if sd.MaskDstXseg is True:
                sd.mFaceSegMask=FaceEngines.getXSegModelEngineTF(sd.xseg_model_dir).extract(face_swap_input_img)
            if sd.mFaceSegMask is None:
                sd.mFaceSegMask = np.ones(shape=(face_height, face_width,3), dtype=np.float32)

            #-- 颜色转移(10ms)
            sd.mFaceSegMask = ImageProcessor(sd.mFaceSegMask).to_ufloat32().get_image('HWC')
            if sd.ColorTransferMode=='rct':
                face_swap_input_img = ImageProcessor(face_swap_input_img).to_ufloat32().get_image('HWC')
                face_swap_out_img = lib_ct.rct(face_swap_out_img, face_swap_input_img, target_mask=sd.mFaceSegMask, source_mask=sd.mFaceSegMask)
            if sd.ColorTransferMode=='sot':
                face_swap_input_img = ImageProcessor(face_swap_input_img).to_ufloat32().get_image('HWC')
                face_swap_out_img = lib_ct.sot(face_swap_out_img, face_swap_input_img, mask=sd.mFaceSegMask)
        
            #---  遮罩边缘模糊(最少10ms，外扩增加1像素多1ms)
            if(sd.face_mask_erode>0 or sd.face_mask_dilate>0):
                sd.mFaceSegMask = ImageProcessor(sd.mFaceSegMask).erode_blur(sd.face_mask_erode, sd.face_mask_dilate, fade_to_border=True).get_image('HWC')
                
            
            #--换脸图遮罩预览（不耗时)
            if sd.PreviewMask :
                face_swap_img=face_swap_out_img.copy();
                mask_debug=ImageProcessor(sd.mFaceSegMask).to_ufloat32().ch(3).get_image('HWC')
                mask_debug[:,:,2]=0; 
                if face_swap_img.shape[0]!=mask_debug.shape[0]:
                    mask_debug=cv2.resize(mask_debug,(face_swap_img.shape[0],face_swap_img.shape[0]));
                sd.mSwapOutImageForPreview = ( face_swap_img * (1-mask_debug) + face_swap_img * mask_debug/ 2 )
            else: 
                sd.mSwapOutImageForPreview =face_swap_out_img;

            #--- 阶段耗时统计（遮罩）
            mask_done_end_time_tick=time.time(); 
            sd.MaskXsegTime=round(mask_done_end_time_tick-merge_start_time_tick,3)*1000
            sd.mDstSegMark=sd.mFaceSegMask;       

             #---人脸融合图计算(把遮罩和换完的融合在一起)
            sd.mMergeFaceImage= face_swap_input_img*(1-sd.mFaceSegMask)+face_swap_out_img*sd.mFaceSegMask;
                 
            #-- 计算align-Src的转换矩阵
            face_height, face_width = face_swap_out_img.shape[:2]
            frame_height, frame_width = frame_image_src_float.shape[:2]
            aligned_to_source_std_mat=aligned_to_source_uni_mat.to_exact_mat (face_width, face_height, frame_width, frame_height)

            

            #--- 人脸位置调整：平移和缩放调整(不耗时)
            if sd.FaceScaleX!=1.0 or sd.FaceScaleY!=1.0:
                aligned_to_source_uni_mat=aligned_to_source_uni_mat.source_scaled_around_center(sd.FaceScaleX,sd.FaceScaleY)
                aligned_to_source_std_mat=aligned_to_source_uni_mat.to_exact_mat (face_width, face_height, frame_width, frame_height)
            
          
        
            mask_to_frame_end_time_tick=time.time()
            
            

            #cv2.imshow("out_merged_frame",out_merged_frame)
            #cv2.waitKey(5)

            #---计算融合输出帧图像
            if sd.BeautyMode=="align" :
                out_merged_frame = ImageProcessor(sd.mMergeFaceImage).warp_affine(aligned_to_source_std_mat, frame_width, frame_height).get_image('HWC')
                origin_frame_image = ImageProcessor(frame_image_src_float).to_ufloat32().get_image('HWC')
                mh,mw,mc=sd.mMergeFaceImage.shape
                moh,mow,moc=sd.mAlignFullMask.shape
                if mh!=moh:
                    sd.mAlignFullMask=np.ones((mh,mw,mc),dtype=float) 
                    sd.mAlignFullMask[0,:,:]=0
                    sd.mAlignFullMask[:,0,:]=0
                    sd.mAlignFullMask[:,-1,:]=0
                    sd.mAlignFullMask[-1,:,:]=0
                
                align_total_mask_frame=ImageProcessor(sd.mAlignFullMask).warp_affine(aligned_to_source_std_mat, frame_width, frame_height, interpolation=ImageProcessor.Interpolation.LINEAR).get_image('HWC')
                one_f = np.float32(1.0)
                dict={"origin_frame_image":origin_frame_image,"one_f":one_f,"align_total_mask_frame":align_total_mask_frame,
                      "out_merged_frame":out_merged_frame}
                sd.mMergeFrameImage = cne.evaluate('origin_frame_image*(one_f-align_total_mask_frame) + out_merged_frame*align_total_mask_frame',dict,global_dict=dict)

            #---计算融合输出帧图像(720*480px-5ms)(640*480px-4ms)(1280*720px-9ms)(用mask去计算)
            if sd.BeautyMode=="frame" :
                 #----把mask遮罩映射到原图
                frame_height, frame_width = frame_image_src_float.shape[0:2]
                face_mask_to_frame = ImageProcessor(sd.mFaceSegMask).warp_affine(aligned_to_source_std_mat, frame_width, frame_height).get_image('HWC')
                sd.mDstSegMarOnFrame=face_mask_to_frame;
                origin_frame_image = ImageProcessor(frame_image_src_float).to_ufloat32().get_image('HWC')
                face_swapped_to_frame_img = ImageProcessor(face_swap_out_img).warp_affine(aligned_to_source_std_mat, frame_width, frame_height, interpolation=ImageProcessor.Interpolation.LINEAR).get_image('HWC')
                one_f = np.float32(1.0)
                dict={"origin_frame_image":origin_frame_image,"one_f":one_f,"face_mask_to_frame":face_mask_to_frame,
                      "face_swapped_to_frame_img":face_swapped_to_frame_img}
                sd.mMergeFrameImage = cne.evaluate('origin_frame_image*(one_f-face_mask_to_frame) + face_swapped_to_frame_img*face_mask_to_frame',dict,global_dict=dict)
             
                 
            #---- 显示AI换脸警示
            if sd.ShowWarn:
                sd.WarnPosX=abs(sd.WarnPosX+sd.WarnOffX)
                sd.WarnPosY=abs(sd.WarnPosY+sd.WarnOffY)
                height,width,c=sd.mMergeFrameImage.shape
                right_margin=max(100,width-150);
                bottom_margin=max(100,height-20)
                if sd.WarnPosX<10:
                    sd.WarnOffX=1;
                if sd.WarnPosX>right_margin:
                    sd.WarnOffX=-1; 
                if sd.WarnPosY<10:
                    sd.WarnOffY=1;
                if sd.WarnPosY>bottom_margin:
                    sd.WarnOffY=-1;
                cv2.putText(sd.mMergeFrameImage,sd.WarnWords,(int(sd.WarnPosX),int(sd.WarnPosY)),cv2.FONT_HERSHEY_SIMPLEX,1,(128,128,128),2)

            #-- 显示融合图片标题
            if sd.ShowAiMark:
                cv2.putText(sd.mMergeFrameImage,sd.AiMark,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.75,(200,200,200),thickness=1)
                #cv2.putText(sd.mRawFrameImageFloat,"RawImage",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),1)
            try:
                ok=self.CombineImagesForOutput()
            except :
                pass
             
            sd.MergeEndTimeTick=time.time(); 
            sd.MergeFaceTime=(sd.MergeEndTimeTick-merge_start_time_tick)*1000;
            #sd.MergeFaceTime=(mask_done_end_time_tick-merge_start_time_tick)*1000;
            sd.FrameTotalTimeUse=round(sd.MergeEndTimeTick-sd.FrameCaptureEndTimeTick,3)*1000;
            self.mMergeEndSignal.emit() 
#---------------组合输出图像
    def CombineImagesForOutput(self):
            if (sd.mRawFrameImageFloat is None) or (sd.mMergeFrameImage is None) :
                return

            merge_size=sd.mMergeFrameImage.shape;
            frame_size=sd.mRawFrameImageFloat.shape;
            
              
            if sd.OutputType==0:
                sd.mOutputImage=sd.mMergeFrameImage;
            if sd.OutputType==1:
                sd.mOutputImage=sd.mMergeFaceImage;
            if sd.OutputType==2:
                sd.mOutputImage=sd.mFrameImageFloat; 

            if sd.OutputType==3:
                if frame_size!=merge_size:
                    sd.mMergeFrameImage=cv2.resize(sd.mMergeFrameImage,frame_size)
                sd.mOutputImage=np.concatenate((sd.mRawFrameImageFloat,sd.mMergeFrameImage),1);
            if sd.OutputType==4:
                if frame_size!=merge_size:
                    sd.mMergeFrameImage=cv2.resize(sd.mMergeFrameImage,frame_size)
                sd.mOutputImage=np.concatenate((sd.mRawFrameImageFloat,sd.mMergeFrameImage),0);
                 
        
            return True


    def SetModelFile(self,filename):
        self.ChangeSwapModelFlag=True;
        self.ModelFilename=filename;
        if(os.path.exists(filename) is False):
            filename=sd.swap_model_dir+filename;
        if os.path.exists(filename):
            self.ModelFilename=filename;
        print(f"[P]设置模型文件:{self.ModelFilename}");