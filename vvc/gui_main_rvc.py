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
    voiceHandler,ui=None,None;
     
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling) 
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app=QtCore.QCoreApplication.instance()
    QtCore.QCoreApplication.setAttribute(Qt.AA_DisableWindowContextHelpButton)
    if app is None:
        app = QApplication(sys.argv)


    splash = QSplashScreen()
    splash.setPixmap(QPixmap(r'ui\icons\splash.jpg'))
    splash.showMessage("RVC程序加载中,请稍等...", QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom, QtCore.Qt.red)
    splash.setFont(QFont('微软雅黑', 14))
    splash.show()
    
    ui=uic.loadUi("../SrcDist/ui/rvc_main.ui") 
    from rvc.VoiceUiHandler import VoiceUiHandler
    voiceHandler=VoiceUiHandler(ui)
    #--  window screen capture
    
    ui.btnBrowseRvcModel.clicked.connect(voiceHandler.btnBrowseRvcModelClick) 
    ui.btnBrowseAudioFile.clicked.connect(voiceHandler.btnBrowseAudioFileClick)

    ui.btnStartVoiceConvert.clicked.connect(voiceHandler.btnStartVoiceConvertClick)
    ui.btnStopVoiceConvert.clicked.connect(voiceHandler.btnStopVoiceConvertClick)

    ui.radioAudioInputDevice.toggled.connect(voiceHandler.OnSetVoiceConfigData)
    ui.radioAudioInputFile.toggled.connect(voiceHandler.OnSetVoiceConfigData)
    ui.radioAudioOutputRaw.toggled.connect(voiceHandler.OnSetVoiceConfigData) 
    ui.radioAudioOutputPitch.toggled.connect(voiceHandler.OnSetVoiceConfigData)
    ui.radioAudioOutputAI.toggled.connect(voiceHandler.OnSetVoiceConfigData)

    ui.spinVoiceThreshold.valueChanged.connect(voiceHandler.OnSetVoiceConfigData)
    ui.spinPitchUp.valueChanged.connect(voiceHandler.OnSetVoiceConfigData) 

    ui.sliderVoiceThreshold.valueChanged.connect(voiceHandler.OnSlideVoiceConfig)
    ui.sliderPitchUp.valueChanged.connect(voiceHandler.OnSlideVoiceConfig)

    ui.checkBoxHalf.stateChanged.connect(voiceHandler.OnSetVoiceConfigData) 
    
    #ui.comboClipMode.currentIndexChanged.connect(voiceHandler.OnConfigDataChange)
    
     
    

    ui.show();
    splash.finish(ui)
    splash.deleteLater()
     
    sys.exit(app.exec_())
    


if __name__=="__main__": 
    main(); 
    #try: 
    #    main(); 
    #except Exception as Ex:
    #    print("error",Ex) 
    
    


  
