# cython:language_level=3
import pickle,time,threading,cv2,os,numpy as np,sys,pickle;

from core.leras import nn
from PyQt5.QtWidgets import QFileDialog,QMessageBox,QPushButton;
from PyQt5.QtCore import QTimer,QThread,QObject;
from PyQt5.QtGui import QPixmap,QImage;
from PyQt5.QtCore import QStringListModel,QAbstractListModel,QModelIndex,QSize
from urllib.request import urlopen
import kit.ShareData as sd
import kit.Auth as Auth
import json

class LoginUiHandler:

    def __init__(self,ui):
        self.ui=ui;
        self.ui.lineEditCpuSN.setText(Auth.getMachineSn())
        
        self.ui.btnActivate.clicked.connect(self.onClickActivate)
        self.ui.btnTryRequest.clicked.connect(self.onClickTry)
      
    def onClickActivate(self):
        cpu_sn=self.ui.lineEditCpuSN.text();
        auth_code=self.ui.lineEditAuthCode.text();
        try:
            myURL = urlopen(f"http://www.vtubekit.com/kitauth.aspx?m=login&sn={cpu_sn}&code={auth_code}")
            result=myURL.read()
            #print(result)
            data=json.loads(result)
            status=data.get("status","fail")
            expire=data.get("expire","2000-00-00")
            msg=data.get("msg"," ")
            code=data.get("code"," ")

            if status=="ok":
                sign_token=Auth.makeLicenceFile(cpu_sn,expire)
                QMessageBox.warning(None,"提示",f"验证激活成功,请重新启动程序");
                os._exit(0)
            else:
                QMessageBox.warning(None,"提示",f"验证失败:{msg}");
        except Exception as ex:
            msg="发生异常： %s" % ex
            QMessageBox.warning(None,"提示",msg)
   

    def onClickTry(self):
        try:
            myURL = urlopen(f"http://www.vtubekit.com/kitauth.aspx?m=try")
            result=myURL.read()

            data=json.loads(result)
            allow=data.get("allow_use","n")
            watermark=data.get("watermark"," ")
            allow_record=data.get("record"," ")
            msg=data.get("msg"," ")
            print(data)
            if allow!="y":
                QMessageBox.warning(None,"提示",f"不允许试用：{msg}");
                return
            
        except Exception as ex:
            msg="发生异常： %s" % ex
            QMessageBox.warning(None,"提示",msg)
  

   
