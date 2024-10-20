#coding=utf-8
# cython:language_level=3
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import time
  
  
class BackendThread(QThread):
  # 定义信号,定义参数为str类型
  breakSignal = pyqtSignal(int)
  
  def __init__(self, parent=None):
    super().__init__(parent)
    # 下面的初始化方法都可以，有的python版本不支持
    # super(Mythread, self).__init__()
  
  def run(self):
      #要定义的行为，比如开始一个活动什么的 
      for i in range(1,100) :
        self.breakSignal.emit(i)
        time.sleep(0.3)
       
