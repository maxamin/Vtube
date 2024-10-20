#coding=utf-8
# cython:language_level=3
import set_env;
set_env.set_env();

from PyQt5.QtWidgets import QApplication,QMainWindow,QColorDialog,QSplashScreen
from PyQt5.QtGui import QPixmap,QImage,QFont,QPalette,QColor;
from PyQt5.QtCore import Qt
from PyQt5 import uic
import sys,os; 
import PyQt5.QtCore as QtCore;
import kit.ShareData as sd

from multiprocessing import Process
from LabUiHandler import LabUiHandler
from AdjustUiHandler import *
from PreviewUiHandler import PreviewUiHandler
 
extractHandler,adjustHandler,ui=None,None,None;
previewUiHandler=None;

class QDarkPalette(QPalette):
    def __init__(self):
        super().__init__()
        text_color = QColor(10,10,10)
        self.setColor(QPalette.Window, QColor(233, 233, 233))
        self.setColor(QPalette.WindowText, text_color )
        self.setColor(QPalette.Base, QColor(125, 125, 125))
        self.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        self.setColor(QPalette.ToolTipBase, text_color )
        self.setColor(QPalette.ToolTipText, text_color )
        self.setColor(QPalette.Text, text_color ) 
        self.setColor(QPalette.Button, QColor(220, 220, 240))
        self.setColor(QPalette.ButtonText, QColor(20, 20, 20))
        self.setColor(QPalette.BrightText, Qt.red)
        self.setColor(QPalette.Link, QColor(42, 130, 218))
        self.setColor(QPalette.Highlight, QColor(42, 130, 118))
        self.setColor(QPalette.HighlightedText, Qt.black)


def main(): 
    
    #-----------注册环境变量
    #print("系统环境变量",os.environ["PATH"])
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling) 
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app=QtCore.QCoreApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    #app.setPalette( QDarkPalette() )
    app.setStyle('Fusion')

    global extractHandler,trainHandler,liveHandler,adjustHandler,previewUiHandler,ui;
    sd.ReadVersionInfo()
    ui=uic.loadUi("../SrcDist/ui/lab_main.ui")

    splash = QSplashScreen()
    splash.setPixmap(QPixmap(sd.LabSplashImg))
    splash.showMessage("素材提取程序加载中,请稍等...", QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom, QtCore.Qt.red)
    splash.setFont(QFont('微软雅黑', 12))
    splash.show()

    
    extractHandler=LabUiHandler(ui)
    adjustHandler=AdjustUiHandler(ui)
    previewUiHandler=PreviewUiHandler(ui)
     

    #--- Video Frame Preview
    ui.btnBrowseVideo.clicked.connect(extractHandler.onBrowseVideoFileClick)
    ui.btnOpenVideoFolder.clicked.connect(extractHandler.btnOpenVideoFolderClick) 
    
    ui.actionGoNextFrame.triggered.connect(extractHandler.goToNextVideoFrameClick) 
    ui.actionExtractFace.triggered.connect(extractHandler.btnExtractFrameFaceClick) 
    ui.actionSaveOneFrame.triggered.connect(extractHandler.btnSaveFaceAndStepClick) 

    ui.btnFirstFrame.clicked.connect(extractHandler.goToStartVideoFrameClick)
    ui.btnPreFrame.clicked.connect(extractHandler.goToPreVideoFrameClick)
    ui.btnNextFrame.clicked.connect(extractHandler.goToNextVideoFrameClick)
    ui.btnLastFrame.clicked.connect(extractHandler.goToLastVideoFrameClick)

    #-- image folder preview
    ui.btnBrowseImageFolder.clicked.connect(extractHandler.btnBrowseImageFolderClick)
    ui.btnOpenImageFolder.clicked.connect(extractHandler.btnOpenImageFolderClick)
    ui.btnNextImage.clicked.connect(extractHandler.btnNextImageClick)  
    ui.btnPreImage.clicked.connect(extractHandler.btnPreImageClick)
    ui.btnFirstImage.clicked.connect(extractHandler.btnFirstImageClick)
    ui.btnLastImage.clicked.connect(extractHandler.btnLastImageClick)

    #--- camera capture
    ui.comboSrcCameraResolution.currentIndexChanged.connect(extractHandler.onChangeCameraResolution) 

    #--  screen capture
    ui.comboWindowList.currentIndexChanged.connect(extractHandler.SetScreenCaptureWindow) 
    ui.btnCaptureWindow.clicked.connect(extractHandler.CaptureCurrentScreenFrame)
    ui.btnRefreshWindowList.clicked.connect(extractHandler.UpdateWindowListForUi)
     
    ui.spinCaptureLeft.valueChanged.connect(extractHandler.SetScreenCaptureRect)
    ui.spinCaptureTop.valueChanged.connect(extractHandler.SetScreenCaptureRect)
    ui.spinCaptureWidth.valueChanged.connect(extractHandler.SetScreenCaptureRect)
    ui.spinCaptureHeight.valueChanged.connect(extractHandler.SetScreenCaptureRect)
    ui.comboLabClipMode.currentIndexChanged.connect(extractHandler.SetScreenCaptureRect) 


    # params serialization
    ui.LineEditVideoFile.textChanged.connect(extractHandler.SaveConfigData)
    ui.LineEditFaceSaveFolder.textChanged.connect(extractHandler.SaveConfigData)
    ui.LineEditImageFolder.textChanged.connect(extractHandler.SaveConfigData)
    ui.LineEditAdjustFolder.textChanged.connect(extractHandler.SaveConfigData)
    ui.LineEditPreviewFolder.textChanged.connect(extractHandler.SaveConfigData)
     
    ui.LineEditOccludeSaveFolder.textChanged.connect(extractHandler.SaveConfigData)

    ui.spinMinFaceSize.valueChanged.connect(extractHandler.SaveConfigData)
    ui.spinBoxExtractFaceSize.valueChanged.connect(extractHandler.SaveConfigData)
 
    #--- Face Extract Preview
    ui.btnExtractFrameFace.clicked.connect(extractHandler.btnExtractFrameFaceClick)
    ui.btnFilterFaces.clicked.connect(extractHandler.btnFilterExtractedFacesClick)
    
    #-- Identity Filter
    ui.btnAddRefImage.clicked.connect(extractHandler.btnAddRefImageClick)
    ui.btnDeleteLastRef.clicked.connect(extractHandler.btnDeleteLastRefClick)
    ui.btnBrowseIdentityImage.clicked.connect(extractHandler.btnBrowseIdentityFilterFaceImageClick)
    ui.btnClearIdetity.clicked.connect(extractHandler.btnClearIdentityFilterFaceClick)
    


    #--- 提取保存人脸
    ui.btnBrowseFaceFolder.clicked.connect(extractHandler.btnBrowseFaceFolderClick)
    ui.btnOpenFaceFolder.clicked.connect(extractHandler.btnOpenFaceFolderClick)
    ui.btnSaveFaceAndNextFrame.clicked.connect(extractHandler.btnSaveFaceAndStepClick)
    ui.btnBatchExtractThread.clicked.connect(extractHandler.onThreadBatchExtractFaceClick) 
    ui.btnStopExtract.clicked.connect(extractHandler.btnStopExtractFaceClick) 
    #--- Face Adjust PostProcess
    ui.btnBrowseAdjustFolder.clicked.connect(adjustHandler.btnBrowseAdjustFolderClick)
    ui.btnOpenAdjustFolder.clicked.connect(adjustHandler.btnOpenFaceFolderClick)

    #--- folder images batch process
    ui.btnAddPrefix.clicked.connect(adjustHandler.btnAddPrefixClick)
    ui.btnAddPostfix.clicked.connect(adjustHandler.btnAddPostfixClick)
    ui.btnRename.clicked.connect(adjustHandler.btnRenameClick) 
    ui.btnResizeFaceImages.clicked.connect(adjustHandler.btnResizeFaceImagesClick)
 
    ui.btnPartMove.clicked.connect(adjustHandler.btnPartMoveClick)
    ui.btnFilterByName.clicked.connect(adjustHandler.btnFilterByNameClick)

    ui.btnPack.clicked.connect(adjustHandler.btnPackClick)
    ui.btnUnPack.clicked.connect(adjustHandler.btnUnPackClick) 
    #ui.btnDenoise.clicked.connect(adjustHandler.btnDenoiseClick)
    #ui.btnEnhance.clicked.connect(adjustHandler.btnEnhanceClick)
    #ui.btnClassByGender.clicked.connect(adjustHandler.btnClassByGenderClick)
    
    #--遮罩外背景清除
    ui.btnEraseMaskBg.clicked.connect(adjustHandler.btnEraseMaskBgClick)

    #-- person verify(Adjust)
    ui.btnAddRefFace.clicked.connect(adjustHandler.btnAddRefFaceImageClick)
    ui.btnDeleteLast.clicked.connect(adjustHandler.btnDeleteLastRefClick)
    ui.btnClearRef.clicked.connect(adjustHandler.btnClearIdentityFilterFaceClick)
    ui.btnStartPersonVerify.clicked.connect(adjustHandler.btnStartPersonVerifyClick)

    #-- super resolution(Adjust)
    ui.btnStartSuperResolution.clicked.connect(adjustHandler.btnStartSuperResolutionClick)
    ui.btnStartMoveBlur.clicked.connect(adjustHandler.btnStartMoveBlurClick)

    #---遮挡人脸选择    
    ui.btnBrowseSrcOccludeFace.clicked.connect(adjustHandler.btnBrowseSrcOccludeFaceClick)  #选择人脸

    #-遮挡物素材
    ui.btnBrowseOccludeFile.clicked.connect(adjustHandler.btnBrowseOccludeFileClick)  
    ui.btnOpenOccludeFolder.clicked.connect(adjustHandler.btnOpenOccludeFolderClick)    
    ui.btnOpenXsegEditorForOcclude.clicked.connect(adjustHandler.btnOpenXsegEditorForOcclude)
    ui.btnAddOccludeSample.clicked.connect(adjustHandler.btnAddOccludeSampleClick) 
    #--遮挡生成单图 
    ui.btnPreviewSaveOcclude.clicked.connect(adjustHandler.btnPreviewSaveOccludeClick)
    ui.btnSaveMergeOcclude.clicked.connect(adjustHandler.btnSaveOccludeMergeResult)
    #--自动批量遮挡生成
    ui.btnBrowseOccludeSaveFolder.clicked.connect(adjustHandler.btnBrowseOccludeSaveFolderClick)
    ui.btnOpenOccludeSaveFolder.clicked.connect(adjustHandler.btnOpenOccludeSaveFolderClick)
    ui.btnStartAutoOccludeFaces.clicked.connect(adjustHandler.btnStartAutoOccludeFacesClick)
    ui.btnStopAutoOccludeFaces.clicked.connect(adjustHandler.StopAutoOccludeFacesMergeClick)

    #--------------------------------------------
    #--- Preview folder images (Preview Page)
    #------------------------------
    ui.btnBrowsePreviewFolder.clicked.connect(previewUiHandler.btnBrowsePreviewFolderClick)
    ui.btnOpenPreviewFolder.clicked.connect(previewUiHandler.btnOpenPreviewFaceFolderClick)

    ui.btnPrePageImage.clicked.connect(previewUiHandler.btnPrePageFaceClick)
    ui.btnNextPageImage.clicked.connect(previewUiHandler.btnNextPageFaceClick)
    ui.btnFirstPageFace.clicked.connect(previewUiHandler.btnFirstPageFaceClick)
    ui.btnLastPageFace.clicked.connect(previewUiHandler.btnLastFacePageClick)

    ui.btnReloadPreviewImages.clicked.connect(previewUiHandler.ReloadPreviewImages)
    ui.btnUpdateImagePreviewTable.clicked.connect(previewUiHandler.UpdateCurrentPageFace)
    ui.checkBoxPreviewLandmarks.stateChanged.connect(previewUiHandler.UpdateCurrentPageFace)
    ui.checkBoxPreviewSegMask.stateChanged.connect(previewUiHandler.UpdateCurrentPageFace)
    ui.btnFaceAngleMap.clicked.connect(previewUiHandler.GenerateFaceAngleMap)
    ui.btnEditPreviewXseg.clicked.connect(previewUiHandler.OpenPreviewXsegEditor)
    ui.btnApplyXSeg.clicked.connect(previewUiHandler.btnApplyXSegClick)

    ui.listWidgetFaceImages.itemDoubleClicked.connect(previewUiHandler.onDoubleClickImage)
    ui.listWidgetFaceImages.itemClicked.connect(previewUiHandler.onSingleClickImage)

    ui.spinPageSize.valueChanged.connect(previewUiHandler.ChangePageSize)
    
    #-- single image preview
    
    ui.actionDeleteFaceImage.triggered.connect(previewUiHandler.onDeleteSelectImage)
    ui.actionWriteAngles.triggered.connect(previewUiHandler.btnWriteFaceAnglesClick)
    ui.actionNextPage.triggered.connect(previewUiHandler.btnNextPageFaceClick)
    ui.actionPrePage.triggered.connect(previewUiHandler.btnPrePageFaceClick)
    ui.actionRefreshPage.triggered.connect(previewUiHandler.ReloadPreviewImages)

    ui.btnDeleteImage.clicked.connect(previewUiHandler.onDeleteSelectImage)
    ui.btnWriteAngle.clicked.connect(previewUiHandler.btnWriteFaceAnglesClick)
    #ui.checkBoxShowHullMask.stateChanged.connect(previewUiHandler.RefreshCurrentFaceImage)
    #ui.checkBoxShowSegMask.stateChanged.connect(previewUiHandler.RefreshCurrentFaceImage)
    #ui.btnWriteFaceAngles.clicked.connect(previewUiHandler.btnWriteFaceAnglesClick)

    ui.spinItemSize.valueChanged.connect(previewUiHandler.onSetIconSize)

    ui.show();

    splash.finish(ui)
    splash.deleteLater()

    sys.exit(app.exec_())

if __name__=="__main__":
    #print(sys.path)
    main();
    


  
