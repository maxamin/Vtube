#coding=utf-8
# cython:language_level=3
import os,sys,traceback,queue,threading,time,itertools,numpy as np,cv2
from pathlib import Path
from core import pathex,imagelib
from core import imagelib
from PyQt5.QtCore import QThread, pyqtSignal

from vtmodel.FaceSwapModelSAE import FaceSwapModelSAE
from core.interact import interact as io


class TrainerThread(QThread):
    model=None; target_iter=0; save_history=False;
    train_pause=False; train_end=False;
    mPreviewReadySignal=pyqtSignal();
    mModelLoadEndSignal=pyqtSignal();
    mCommandQueue = queue.Queue()
    previews=None;
    kwargs=None;
    isTraining=True;

    def __init__(self,kwargs) :
        super(TrainerThread, self).__init__()
        self.kwargs=kwargs;
        self.auto_save_min = 30
        self.auto_backup_min=122
        self.last_save_time = time.time()
        self.last_backup_time=time.time()
        self.is_reached_goal = False
        self.shared_state = { 'after_save' : False }
         
    def model_save(self):
            io.log_info ("[S]Saving Model....", end='\r')
            self.last_save_time= time.time()
            self.model.save()
            io.log_info ("[P]Saving Finished", end='\r')
            self.shared_state['after_save'] = True
                    
    def model_backup(self):
        io.log_info ("[S]Create Backup...", end='\r')
        self.last_backup_time=time.time()
        self.model.create_backup()             
        io.log_info ("[S]Backup Finished", end='\r')

    def send_preview(self,idx=0):
        self.previews = self.model.get_previews(idx)
        self.mPreviewReadySignal.emit()  

    def test_preview(self):
        #io.log_info('receive test preview op.')
        ok,test_previews=self.model.get_test_preview()
        if ok is False: return
        self.mPreviewReadySignal.emit()
         
    #----- 训练线程持续运行函数
    def run(self):

            self.model=FaceSwapModelSAE(saved_models_dir=self.kwargs["saved_models_dir"],kwargs=self.kwargs)
            self.model.load_model_from_dir()
            self.model.define_tf_output_and_loss()
            self.model.load_train_samples()
            self.save_iter =  self.model.iter
            self.isTraining=True;
            print(self.model.get_summary_text())
            self.mModelLoadEndSignal.emit()
            print("[P]Prepare for train...")
            for i in itertools.count(0,1):
                if self.train_pause is True:
                    time.sleep(0.5)
                    continue;                 
                    
                iter, iter_time = self.model.train_one_iter()
                loss_string = ""
                loss_history = self.model.loss_history
                time_str = time.strftime("[%H:%M:%S]")
                if iter_time >= 10:
                    loss_string = "{0}[#{1:06d}][{2:.5s}s]".format ( time_str, iter, '{:0.4f}'.format(iter_time) )
                else:
                    loss_string = "{0}[#{1:06d}][{2:04d}ms]".format ( time_str, iter, int(iter_time*1000) )

                #--- 输出 Loss history
                if self.shared_state['after_save']:
                    self.shared_state['after_save'] = False
                    loss_history_count=len(loss_history)
                    loss_history_span=iter-self.save_iter;
                    mean_loss=0.0;
                    if loss_history_span<loss_history_count:
                        mean_loss = np.mean ( loss_history[-loss_history_span:-1], axis=0)
                    else:
                        mean_loss = np.mean ( loss_history[-100:-1], axis=0)
                    for loss_value in mean_loss:
                        loss_string += "[%.4f]" % (loss_value)

                    io.log_info(loss_string)

                    self.save_iter = iter
                else:
                    for loss_value in loss_history[-1]:
                        loss_string += "[%.4f]" % (loss_value)
                io.log_info ("[T]"+loss_string, end='\r')
                io.log_info ("[S]"+self.model.get_train_status(), end='\r')

                #--- 判断是否到达设定迭代数
                if iter>=self.target_iter and self.target_iter>0:
                    io.log_info ('Reached target iteration.')
                    self.model_save()
                    self.train_end = True
            
                #--- 判断是否保存模型
                if time.time() - self.last_save_time >= self.auto_save_min*60:                    
                    self.model_save()
                    self.send_preview()

                if i==0:
                    self.send_preview()

                #--- 判断是否备份模型
                if time.time() - self.last_backup_time >= self.auto_backup_min*60:                    
                    self.model_backup()
                    self.send_preview()

                #--- 消息队列处理
                while not self.mCommandQueue.empty():
                    input = self.mCommandQueue.get()
                    op = input['op']
                    if op == 'save':
                        self.model_save()
                    elif op == 'backup':
                        self.model_backup()
                    elif op == 'preview':
                        self.send_preview(input.get("type",0))
                    elif op == 'test':
                        self.test_preview()
                    elif op == 'close':
                        self.model_save()
                        i = -1
                        break

                if i == -1:
                    break

            self.model.finalize()
            print("结束模型训练") 
            self.isTraining=False;
            self.train_end=True;
            del self
            #cv2.destroyAllWindows()

        
         