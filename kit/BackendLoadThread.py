#coding=utf-8
# cython:language_level=3
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import time
import kit.ToolKit as ToolKit


class BackendLoadThread(QThread):
  # 定义信号,定义参数为str类型
  breakSignal = pyqtSignal()
  
  def __init__(self, parent=None):
    super().__init__(parent)
  
  def run(self):
      #ToolKit.InitAndLoadModels();
      DeepFaceID.loadDeepIDModel();
      self.breakSignal.emit()
       
