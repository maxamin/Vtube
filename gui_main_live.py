#coding=utf-8
# cython:language_level=3
import set_env;
set_env.set_env();


  
def main(): 
    from PyQt5.QtWidgets import QApplication,QMainWindow,QColorDialog,QSplashScreen,QMessageBox
    from PyQt5.QtGui import QPixmap,QImage,QFont;
    from PyQt5 import uic
    import sys,os; 
    import PyQt5.QtCore as QtCore;
    from PyQt5.QtCore import   Qt
    from multiprocessing import Process
    import kit.ShareData as sd
    from  LoginUiHandler import LoginUiHandler;
    from LiveUiHandler import LiveUiHandler
    from vvc.VoiceUiHandler import VoiceUiHandler

    liveHandler,ui=None,None;
     
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling) 
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app=QtCore.QCoreApplication.instance()
    QtCore.QCoreApplication.setAttribute(Qt.AA_DisableWindowContextHelpButton)
    if app is None:
        app = QApplication(sys.argv)

    sd.ReadVersionInfo()

    splash = QSplashScreen()
    splash.setPixmap(QPixmap(sd.LiveSplashImg))
    splash.showMessage("程序加载中,请稍等...", QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom, QtCore.Qt.red)
    splash.setFont(QFont('微软雅黑', 14))
    splash.show() 
    try:
        ui=uic.loadUi("../SrcDist/ui/live_main.ui") 
    except :
        print("load ui error") 
    
    voiceHandler=VoiceUiHandler(ui) 
    liveHandler=LiveUiHandler(ui) 
    #--  window screen capture
    
    ui.btnRefreshWindowList.clicked.connect(liveHandler.UpdateWindowListForUi)
    ui.spinCapRectLeft.valueChanged.connect(liveHandler.OnConfigDataChange)
    ui.spinCapRectTop.valueChanged.connect(liveHandler.OnConfigDataChange)
    ui.spinCapRectWidth.valueChanged.connect(liveHandler.OnConfigDataChange)
    ui.spinCapRectHeight.valueChanged.connect(liveHandler.OnConfigDataChange)
    ui.btnCaptureWindow.clicked.connect(liveHandler.SetScreenCaptureWindow)
    ui.comboClipMode.currentIndexChanged.connect(liveHandler.OnConfigDataChange)
    
    #---美颜，脸型调整
    ui.btnResetBeaty.clicked.connect(liveHandler.btnResetBeatyClick) 

    ui.spinFaceWhite.valueChanged.connect(liveHandler.OnChangeFaceBeautyConfig)
    ui.spinFaceSmooth.valueChanged.connect(liveHandler.OnChangeFaceBeautyConfig)
    ui.spinSharpen.valueChanged.connect(liveHandler.OnChangeFaceBeautyConfig)

    ui.spinSmallMouth.valueChanged.connect(liveHandler.OnChangeFaceBeautyConfig)
    ui.spinEyeDistance.valueChanged.connect(liveHandler.OnChangeFaceBeautyConfig)
    ui.spinLowCheekWidth.valueChanged.connect(liveHandler.OnChangeFaceBeautyConfig)

    ui.spinMalarThin.valueChanged.connect(liveHandler.OnChangeFaceBeautyConfig)    
    ui.spinJawThin.valueChanged.connect(liveHandler.OnChangeFaceBeautyConfig)
    ui.spinCheekThin.valueChanged.connect(liveHandler.OnChangeFaceBeautyConfig)
    ui.spinFaceLong.valueChanged.connect(liveHandler.OnChangeFaceBeautyConfig)
    #ui.spinBigEye.valueChanged.connect(liveHandler.OnChangeFaceBeautyConfig)
    ui.spinSmallMouth.valueChanged.connect(liveHandler.OnChangeFaceBeautyConfig)
    ui.spinEyeDistance.valueChanged.connect(liveHandler.OnChangeFaceBeautyConfig)
    ui.spinLowCheekWidth.valueChanged.connect(liveHandler.OnChangeFaceBeautyConfig)

    ui.sliderFaceWhite.valueChanged.connect(liveHandler.OnSlideFaceParam)
    ui.sliderSharpen.valueChanged.connect(liveHandler.OnSlideFaceParam)
    ui.sliderFaceSmooth.valueChanged.connect(liveHandler.OnSlideFaceParam)

    ui.sliderMalarThin.valueChanged.connect(liveHandler.OnSlideFaceParam)
    ui.sliderCheekThin.valueChanged.connect(liveHandler.OnSlideFaceParam)
    ui.sliderJawThin.valueChanged.connect(liveHandler.OnSlideFaceParam)
    ui.sliderFaceLong.valueChanged.connect(liveHandler.OnSlideFaceParam)
    ui.sliderSmallMouth.valueChanged.connect(liveHandler.OnSlideFaceParam)
    ui.sliderEyeDistance.valueChanged.connect(liveHandler.OnSlideFaceParam)
    ui.sliderLowCheekWidth.valueChanged.connect(liveHandler.OnSlideFaceParam)
    
        
    #---swap model file lib
    ui.btnSetSwapModelFolder.clicked.connect(liveHandler.OnSetSwapModelFolderClick)
    ui.btnOpenModelFolder.clicked.connect(liveHandler.btnOpenModelFolderClick)
    ui.btnUpdateModelList.clicked.connect(liveHandler.UpdateListForModelFiles) 
     
    ui.tableViewModelLib.doubleClicked.connect(liveHandler.OnChangeLiveModelFileClick)

    #--- cam and video play
    ui.btnUseCamera.clicked.connect(liveHandler.ApplyCameraCaptureSetting)
    ui.comboLiveCameraID.currentIndexChanged.connect(liveHandler.ApplyCameraCaptureSetting)
    ui.comboCamResolution.currentIndexChanged.connect(liveHandler.ApplyCameraCaptureSetting)
    #ui.comboCamDrive.currentIndexChanged.connect(liveHandler.ApplyCameraCaptureSetting)
    ui.checkBoxFlipCameraH.stateChanged.connect(liveHandler.OnConfigDataChange)
    ui.comboCameraRotate.currentIndexChanged.connect(liveHandler.OnConfigDataChange)
    ui.btnCameraSetting.clicked.connect(liveHandler.onClickCameraSetting)
    #-- video file play
    ui.btnBrowseMediaFile.clicked.connect(liveHandler.btnBrowseMediaFileClick) 
    ui.btnOpenLiveMediaFolder.clicked.connect(liveHandler.OnBtnOpenLiveMediaFolderClick) 
    ui.pushButtonPlay.clicked.connect(liveHandler.StartPlayVideo)
    ui.pushButtonPause.clicked.connect(liveHandler.PausePlayVideo)
    ui.pushButtonStop.clicked.connect(liveHandler.StopPlayVideo)
    ui.pushButtonToHead.clicked.connect(liveHandler.PlayJusmToHead)
    ui.pushButtonToEnd.clicked.connect(liveHandler.PlayJusmToEnd)
    ui.sliderVideoPlayFrame.sliderPressed.connect(liveHandler.StopPlayVideo) 
    ui.sliderVideoPlayFrame.sliderMoved.connect(liveHandler.MoveVideoPlayFrame) 
    ui.sliderVideoPlayFrame.sliderReleased.connect(liveHandler.PausePlayVideo) 

    #--- image folder
    ui.btnBrowseImageFolder.clicked.connect(liveHandler.OnBrowseImageFolderClick)
    ui.btnOpenImageFolder.clicked.connect(liveHandler.OnOpenImageFolderClick)
    ui.pushButtonPreImage.clicked.connect(liveHandler.pushButtonPreImageClick)
    ui.pushButtonNextImage.clicked.connect(liveHandler.pushButtonNextImageClick)

    #--- video  record
    ui.btnBrowseRecordDir.clicked.connect(liveHandler.OnBrowseRecordDirClick)
    ui.btnOpenRecordDir.clicked.connect(liveHandler.OnOpenRecordDirClick)
    ui.btnStartWriteVideo.clicked.connect(liveHandler.OnRecordStartStopClick)

    #--engine switch: xseg/detect 
    ui.btnBrowseXsegFolder.clicked.connect(liveHandler.OnBrowseXsegFolderClick) 
    ui.comboInferDevices.currentIndexChanged.connect(liveHandler.ChangeInferDevice)

    #-- 人脸检测
    ui.spinFaceDetectThreshold.valueChanged.connect(liveHandler.OnConfigDataChange)
    ui.spinDetectWindowSize.valueChanged.connect(liveHandler.OnConfigDataChange)
    ui.spinFaceReso.valueChanged.connect(liveHandler.OnConfigDataChange)

    #---身份筛选
    ui.btnAddFilterFace.clicked.connect(liveHandler.btnAddFilterFaceClick)
    ui.btnClearIdetity.clicked.connect(liveHandler.btnClearIdentityFilterFaceClick)
    ui.btnLivePopRef.clicked.connect(liveHandler.btnLivePopRefClick) 

    #--- Swap,Merge param config
    ui.btnBrowseBg.clicked.connect(liveHandler.btnBrowseBgClick) 
    ui.checkReplacePreview.stateChanged.connect(liveHandler.setReplacePreview) 
    

    ui.spinDownSample.valueChanged.connect(liveHandler.OnConfigDataChange)
    ui.radioBgReplaceNo.clicked.connect(liveHandler.OnConfigDataChange)
    ui.radioBgReplaceGreen.clicked.connect(liveHandler.OnConfigDataChange)
    ui.radioBgReplaceAI.clicked.connect(liveHandler.OnConfigDataChange)

    ui.comboMergeMode.currentIndexChanged.connect(liveHandler.OnConfigDataChange)
    ui.spinMorphFactor.valueChanged.connect(liveHandler.OnConfigDataChange)
    ui.spinFrameResizeWidth.valueChanged.connect(liveHandler.OnConfigDataChange)
    ui.checkResizeFrame.stateChanged.connect(liveHandler.OnConfigDataChange)
    ui.spinSharpen.valueChanged.connect(liveHandler.OnConfigDataChange)
    ui.checkShowAiMark.stateChanged.connect(liveHandler.OnConfigDataChange)    
    ui.spinFaceVerifyThreshold.valueChanged.connect(liveHandler.OnConfigDataChange)
    ui.spinMaskErode.valueChanged.connect(liveHandler.OnConfigDataChange)
    ui.spinMaskDilate.valueChanged.connect(liveHandler.OnConfigDataChange)
    ui.comboSuperResoEngine.currentIndexChanged.connect(liveHandler.OnConfigDataChange)    
    #--output config
    ui.radioOutSizeOrigin.clicked.connect(liveHandler.OnConfigDataChange)
    ui.radioOutFixWidth.clicked.connect(liveHandler.OnConfigDataChange)
    ui.radioOutFixHeight.clicked.connect(liveHandler.OnConfigDataChange)
    ui.spinOutputHeightLimit.valueChanged.connect(liveHandler.OnConfigDataChange)
    ui.spinOutputWidthLimit.valueChanged.connect(liveHandler.OnConfigDataChange)
    ui.comboOutputType.currentIndexChanged.connect(liveHandler.OnConfigDataChange)
    ui.comboNoFaceMode.currentIndexChanged.connect(liveHandler.OnConfigDataChange)
    
    ui.comboOutputRotate.currentIndexChanged.connect(liveHandler.OnConfigDataChange)
    ui.checkTopMost.stateChanged.connect(liveHandler.setOutputTopMost)
    ui.checkPreviewFrame.stateChanged.connect(liveHandler.OnConfigDataChange)
    ui.checkPreviewAlignFace.stateChanged.connect(liveHandler.OnConfigDataChange)
    ui.checkPreviewSwap.stateChanged.connect(liveHandler.OnConfigDataChange)
    ui.checkPreviewMask.stateChanged.connect(liveHandler.OnConfigDataChange)
    ui.comboColorTransMode.currentIndexChanged.connect(liveHandler.OnConfigDataChange)    
    ui.checkMaskFromSrc.stateChanged.connect(liveHandler.OnConfigDataChange)
    ui.checkMaskXSeg.stateChanged.connect(liveHandler.OnConfigDataChange)
    ui.checkMaskXSeg.stateChanged.connect(liveHandler.OnConfigDataChange)     
    ui.spinFaceScaleX.valueChanged.connect(liveHandler.OnConfigDataChange)
    ui.spinFaceScaleY.valueChanged.connect(liveHandler.OnConfigDataChange)
    ui.checkShowFaceMerge.stateChanged.connect(liveHandler.OnConfigDataChange)

    
    #---save and record
    ui.btnSaveOutputImage.clicked.connect(liveHandler.btnSaveOutputImageClick)
    ui.btnTogglePrivateMode.clicked.connect(liveHandler.btnTogglePrivateModeClick)

    #---- Voice Handler (变声功能）
    ui.btnBrowseRvcModel.clicked.connect(voiceHandler.btnBrowseRvcModelClick)  
    ui.btnStartVoiceConvert.clicked.connect(voiceHandler.btnStartVoiceConvertClick)
    ui.btnStopVoiceConvert.clicked.connect(voiceHandler.btnStopVoiceConvertClick)

    ui.radioAudioOutputRaw.toggled.connect(voiceHandler.OnSetVoiceConfigData) 
    ui.radioAudioOutputPitch.toggled.connect(voiceHandler.OnSetVoiceConfigData)
    ui.radioAudioOutputAI.toggled.connect(voiceHandler.OnSetVoiceConfigData)
    ui.spinVoiceThreshold.valueChanged.connect(voiceHandler.OnSetVoiceConfigData)
    ui.spinPitchUp.valueChanged.connect(voiceHandler.OnSetVoiceConfigData) 
    ui.spinBlockTime.valueChanged.connect(voiceHandler.OnSetVoiceConfigData) 
    ui.spinDelay.valueChanged.connect(voiceHandler.OnSetVoiceConfigData) 
    ui.spinAudioVolume.valueChanged.connect(voiceHandler.OnSetVoiceConfigData) 
    
    ui.sliderVoiceThreshold.valueChanged.connect(voiceHandler.OnSlideVoiceConfig)
    ui.sliderPitchUp.valueChanged.connect(voiceHandler.OnSlideVoiceConfig) 
    ui.sliderBlockTime.valueChanged.connect(voiceHandler.OnSlideVoiceConfig) 
    ui.sliderDelay.valueChanged.connect(voiceHandler.OnSlideVoiceConfig)
    ui.sliderAudioVolume.valueChanged.connect(voiceHandler.OnSlideVoiceConfig)
    
    ui.comboInputAudioDevices.currentIndexChanged.connect(voiceHandler.OnChangeAudioDevice)
    ui.comboOutputAudioDevices.currentIndexChanged.connect(voiceHandler.OnChangeAudioDevice)

    ui.show();
    splash.finish(ui)
    splash.deleteLater()
    
    liveHandler.Init_All_System()
    sys.exit(app.exec_())
    


if __name__=="__main__": 
    try: 
        main(); 
    except Exception as Ex:
        print("error",Ex) 
    
    


  
