# cython:language_level=3
# coding=utf-8

import set_env;
set_env.set_env();

from PyQt5.QtWidgets import QApplication,QMainWindow,QColorDialog,QSplashScreen
from PyQt5.QtGui import QPixmap,QImage,QFont;
from PyQt5 import uic
import sys,os; 
import PyQt5.QtCore as QtCore;
import kit.ShareData as sd

from multiprocessing import Process
from  TrainUiHandlerSlot import *;

extractHandler,trainHandler,liveHandler,adjustHandler,ui=None,None,None,None,None;
previewUiHandler=None;


def main(): 
    
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling) 
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app=QtCore.QCoreApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    global extractHandler,trainHandler,liveHandler,adjustHandler,previewUiHandler,ui;
    sd.ReadVersionInfo()

    splash = QSplashScreen()
    splash.setPixmap(QPixmap(sd.TrainSplashImg))
    splash.showMessage("模型训练程序加载中,请稍等...", QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom, QtCore.Qt.red)
    splash.setFont(QFont('微软雅黑', 14))
    splash.show()
    
    
    ui=uic.loadUi("../SrcDist/ui/trainer_main.ui")
    
    trainHandler=TrainUiHandlerSlot(ui) 
     
    ui.LineEditTrainSrcFaceFolder.textChanged.connect(trainHandler.SaveConfigData)
    ui.LineEditTrainDstFaceFolder.textChanged.connect(trainHandler.SaveConfigData)
    ui.LineEditModelSaveDir.textChanged.connect(trainHandler.SaveConfigData)

    #---  Src dst model folder (Train Page)

    ui.btnOpenTrainSrcFolder.clicked.connect(trainHandler.btnOpenTrainSrcFolderClick)
    ui.btnOpenTrainDstFolder.clicked.connect(trainHandler.btnOpenTrainDstFolderClick)
    ui.btnBrowseTrainSrcFolder.clicked.connect(trainHandler.btnBrowseTrainSrcFolderClick)
    ui.btnBrowseTrainDstFolder.clicked.connect(trainHandler.btnBrowseTrainDstFolderClick) 
    ui.btnBrowseModelFolder.clicked.connect(trainHandler.btnBrowseModelFolderClick)
    ui.btnOpenModelFolder.clicked.connect(trainHandler.btnOpenModelFolderClick)

    #-- 测试预览
    
    ui.checkPreviewTopMost.stateChanged.connect(trainHandler.onTogglePreviewTopmost) 


    ui.btnBrowseTestFolder.clicked.connect(trainHandler.btnBrowseTestFolderClick)
    ui.btnOpenTestFolder.clicked.connect(trainHandler.btnOpenTestFolderClick)
    ui.btnCapScreen.clicked.connect(trainHandler.btnCapScreenClick)
    ui.btnTrainTestPreview.clicked.connect(trainHandler.btnTestPreviewClick)
    ui.btnFirstImg.clicked.connect(trainHandler.onFirstImgTestClick)
    ui.btnPreImg.clicked.connect(trainHandler.onPreImgTestClick)
    ui.btnNextImg.clicked.connect(trainHandler.onNextImgTestClick )
    ui.btnLastImg.clicked.connect(trainHandler.onLastImgTestClick )

    #-- model train btns (train page)
    ui.btnLoadModelParams.clicked.connect(trainHandler.btnLoadModelParamsClick)
    ui.btnStartTrainModel.clicked.connect(trainHandler.btnStartTrainModelClick)
    ui.btnCreateNewModel.clicked.connect(trainHandler.btnCreateNewModelClick)
    ui.btnExportDFM.clicked.connect(trainHandler.btnExportDFMClick)

    #-- train window (train page)    
    ui.btnUpdateTrainPreview.clicked.connect(trainHandler.btnUpdateTrainPreviewClick)
    ui.btnSwitchMaskPreview.clicked.connect(trainHandler.onSwitchMaskPreview)
    ui.btnCloseTrain.clicked.connect(trainHandler.btnCloseTrainWindowClick)    
    ui.btnSaveTrainModel.clicked.connect(trainHandler.btnSaveTrainModelClick)
    ui.btnBackupTrainModel.clicked.connect(trainHandler.btnBackupTrainModelClick)
    ui.btnPauseRestoreTrain.clicked.connect(trainHandler.btnPauseRestoreTrainClick)     
    
   
    #--vtfm 模型转换
    ui.btnBrowseDfmModel.clicked.connect(trainHandler.btnBrowseDfmModelClick) 
    ui.btnExportVtfm.clicked.connect(trainHandler.btnExportVtfmClick)  
    ui.btnBrowseVtfmModel.clicked.connect(trainHandler.btnBrowseVtfmModelClick) 
    ui.btnWriteModelAuthToken.clicked.connect(trainHandler.WriteModelAuthCode) 



    #--dat 操作
    ui.btnBrowseDatFile.clicked.connect(trainHandler.btnBrowseDatClick)
    ui.btnDeleteFileModelParam.clicked.connect(trainHandler.DeleteDatParamItem)
    ui.btnTrunkHistory.clicked.connect(trainHandler.onTrunkHistoryClick)


    ui.show();

    splash.finish(ui)
    splash.deleteLater()

    sys.exit(app.exec_())

if __name__=="__main__":
    #print(sys.path)
    main();
    


  
