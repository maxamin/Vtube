#coding=utf-8
# cython:language_level=3
import numpy,cv2,threading,os,time;
from PyQt5.Qt import  QThread,QMutex,pyqtSignal

SoftName="VtubeKit"; Version="V231025";
LiveSplashImg=r"..\SrcDist\ui\icons\vtube_live.png"
LabSplashImg=r'..\SrcDist\ui\icons\vtube_lab.png'
TrainSplashImg=r"..\SrcDist\ui\icons\vtube_train.png"
LiveLogoImage=r"..\SrcDist\ui\icons\live.png"
SysDate=time.strftime('%Y-%m-%d', time.localtime());

#--AuthCheck
ShowAlert=False;  AlertMsg=""; ExitApp=False;OpenWeb=False; WebUrl="";ShowAd=False; AdWords="";
ShowWarn=True; WarnWords="VtubeKit AI"; AiMark="AI Merge"; TrainPreviewWindowTitle="Training_Preview";
WarnPosX=10;WarnPosY=100; WarnOffX=1;WarnOffY=1;

AppEnable=False;
LabExtractFaceEnable=False; LabXsegEditEnable=False; LabOcccludeEnable=False;

#--models 
xseg_model_dir=os.path.abspath(r'..\XSegModels\default_xseg');
ffmpeg_path=r"../CUDA_CuDNN_Dlls\Cuda_Torch_2.0_Cuda_11.6/ffmpeg.exe"
swap_model_dir=os.path.abspath("..\\Model_Lib\\");

last_record_video_file=""; last_record_audio_file="";

#-- shared images
mRawFrameImageFloat=None;mRawFrameImageInt=None;
mFrameImageFloat=None;mMarkFaceImage=None;LastFrameTick=0.01;FrameCapUseTime=1.0;
mFrameImageInt=None;mMarkFaceImageInt=None;mFrameImageAfterDetect=None;
mFaceSegMask=None; mAlignFullMask=numpy.ones((252,252,3),dtype=float); 

#---人脸检测设置
DetectMode=0; 
DetectRawFaces=None; DetectAlignFaces=None;
FaceSwapInfo={}; DetectFaceInfo={};
ShowAiMark=False; ShowMergeFace=True;
FaceSwapInfoList=[]

#---摄像头设置 setting
CamIdx=0; CamResolution=(1280,960); FrameRotateMode=0;  CameraFlipHorizontal=True;
CamDriveMode=0;
ResizeVideoFrame=False; ResizeVideoWidth=1000;

#beaty face (美颜脸型调整设置)
BeautyShapeEnable=False;
FaceLongValue=0.0;MalarThinValue=0.0;JawThinValue=0.0;CheekThinValue=0.0;
BigEyeValue=0.0; MouthSmallValue=0.0; EyeDistanceValue=0.0; LowCheekThinValue=0;

#--- face detect/filter/align data
ExistTargetFace=False;mParseFaceImage=None;mAlignFaceImage=None; 
DetectFaceRect=(0,0,-1,-1)
face_ulmrks=None;face_lmks_coords=None;
face_ulmrks_inSrc=None;ldmrkCoordsSrc=None;
#face_align_img=None;    face_align_lmrks_mask_img=None; aligned_to_source_uni_mat=None; 
LastTargetFaceBox=None; FastFaceDetect=True; LastFaceUniRect=None;
ChooseFaceIndex=0;

#-- face swap data
mSwapOutImage=None;mMergeFaceImage=None;
mMergeFrameImage=None;mOutputImage=None;  mMergeFrameImageInt=None;
mOutputResizeImage=None;mDstSegMark=None;mDstSegMarOnFrame=None;
mSwapOutImageForPreview=None; ShowPreviewUpdateTime=True;

#--detect and swap config
DetectThreshold=0.5; VerifyThreshold=1.0; DetectWindowSize=320;
LandmarkArea=1.4; AlignCoverage=2.4; AlignResolution=320; 
ReplaceBgNo=True; ReplaceBgGreen=False;ReplaceBgAI=False; ReplacePreview=False; BgDownSample=0.25;

#--- face beatify
SharpenValue=20.0; SmoothValue=0.0; WhiteValue=0.0;
ColorShiftH=0;ColorShiftS=0;ColorShiftV=0;
GammaR=0.0;GammaG=0.0;GammaB=0.0; GammaTarget=0;
PostGammaR=1.0;PostGammaG=1.0;PostGammaB=1.0; PostSharpenValue=0.0;
PreBright=1.0;PostBright=1.0; MorphFactor=1.0;
BeautyMode="align";

#-- merge config
MaskDstXseg=True; MaskFromSrc=False; MergeSuperResoEngine="none";
face_mask_erode=1;face_mask_dilate=25;    
OutputWidthLimit,OutputHeightLimit,OutputType=1400,800,0
NoFaceMode="画面冻结"; OutputResizeMode=0;
OffsetX=0.0;OffsetY=0.0;FaceScaleX=1.0;FaceScaleY=1.0;
ColorTransfer=True; ShowPreview=False; ColorTransferMode='none';

#---screen/window capture
hWnd_info_list=None;
CaptureRect=(0,0,1920,1080);CaptureClipMode=0;

#----- config params
PreviewMask=True;PreviewFrame=True;PreviewAlignFace=True;PreviewSwap=True;

#--- step time use
FrameCaptureEndTimeTick=0; MergeEndTimeTick=0; FrameTotalTimeUse=0;
DetectFaceTime=0.020; SrcFaceBeautyTime=0;FaceBeautyTime=0.0;FaceBeautyRemapTime=0;
AlignFaceTime=0.020;SwaFaceTime=0.020;  MaskXsegTime=0.020;
MergeFaceTime=0.020; FrameUpdateTime=0.020;last_output_time_tick=0;
OutputGapTime=1.0;
#---cfg params
options=[]

#---- Lab Data: Extract/Train
ImageList=[];   CurrImageIdx=0; ImageCount=0;
ExtractNum=1;

#--video record
RemoveRecordTemp=True; RecordAudio=True; RecordTempSave=False;


#--- face extract app(人脸提取公共参数）
ExtractFrameImage=None; VideoFrameReadTime=0.0;
VideoFileName=""; VideoFrameImage=None; VideoFrameCount=0; VideoCurrFrameIndex=0;

#--output window
OutputWindowName="VtubeKit_Output_Preview"; OutputTopMost=1;
OutputRotateMode=0;

def ReadVersionInfo():
    import configparser
    import kit.Auth as auth
    if os.path.exists("../BranchVersion.cfg") is False:
        #print("不存在版本文件，退出")
        return

    global SoftName,Version,LiveSplashImg,LabSplashImg,TrainSplashImg,LiveLogoImage
    config = configparser.ConfigParser()
    config.read("../BranchVersion.cfg")
    read_SoftName = config.get('version','name') 
    read_token = config.get('version','token')
    read_token_raw=f"{read_SoftName}-ttt"
    token=auth.get_md5_hash_str(read_token_raw)
    #print(token)
    if token==read_token:
        SoftName=read_SoftName;
        #Version=read_Version;
        LiveSplashImg=f"../SrcDist/version/{read_SoftName}/splash.jpg";
        LabSplashImg=f"../SrcDist/version/{read_SoftName}/splash.jpg";
        TrainSplashImg=f"../SrcDist/version/{read_SoftName}/splash.jpg";
        LiveLogoImage=f"../SrcDist/version/{read_SoftName}/logo.jpg";
    else:
        print("软件版本定制文件验证码错误")

    
 
