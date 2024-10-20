#coding=utf-8
# cython:language_level=3

from PyQt5.QtWidgets import QFileDialog,QMessageBox,QInputDialog,QDialog,QFormLayout,QApplication,QComboBox,QTableWidgetItem;
from PyQt5.QtCore import QThread, pyqtSignal,QObject,Qt
from PyQt5.QtGui import QTextCursor
import cv2,os,time,threading,pickle,shutil,sys,numpy as np;
from vtmodel.ModelTrainer import TrainerThread
from pathlib import Path
from core.leras import nn
from kit import ToolKit
import kit.screen_cap as cap
import core.pathex as pathex  
import kit.Auth as auth
from core import cv2ex
from kit.FaceEngines import FaceEngines
from core.xlib.face.FRect import FRect
from kit.ServerRequestThread import ServerRequestThread
import kit.ShareData as sd
import webbrowser
from kit.LiveSwapModel import LiveSwapModel

class Signal(QObject):
    text_update = pyqtSignal(str)
    def write(self, text):
        self.text_update.emit(str(text))
        QApplication.processEvents()
    def flush(self):
        pass;


class TrainUiHandlerSlot:

    OverrideLine=False;

    def __init__(self,ui):
        self.ui=ui;   
        self.Trainer=None
        self.previews=None;
        self.preview_idx=0;
        self.preview_raw_image=None;

        self.LoadSavedConfig()
        nn.initialize_main_env()
        cap.cap_init()
        cap.set_cap_desktop()
        sys.stdout = Signal()
        sys.stdout.text_update.connect(self.UpdateConsoleText)

        sd.SoftName=f"{sd.SoftName}_Train";
        self.ui.setWindowTitle(f"{sd.SoftName} ({sd.Version})");
         
        self.mAuthCheckThread=ServerRequestThread(app=sd.SoftName,delay=5.0,mode="auth_check",interval=8)
        self.mAuthCheckThread.mAlertSignal.connect(self.AuthCheckAlert)
        self.mAuthCheckThread.start();

        ToolKit.PreLoadEngines(yolo=True)

    def AuthCheckAlert(self):
        if sd.ShowAlert:
            ToolKit.ShowWarningError(None,"提示",sd.AlertMsg); 
        if sd.ExitApp:
            exit(0);
        if sd.OpenWeb:
            webbrowser.open(sd.WebUrl, new=0, autoraise=True);
        #self.ui.tabWidgetCmd.setEnabled(True)
     
    def UpdateConsoleText(self,text): 
        self.ui.textEditTrainData.moveCursor(QTextCursor.End)
        if text.startswith('[T]') or text.startswith('[P]'):
            self.ui.textEditTrainData.setText(text)
            self.print_endline=False;
            return
        if text.startswith('[S]'):
            self.ui.textEditTrainStatus.setText(text)
            self.print_endline=False;
            return
        if len(text)<2 and ('\n' not in text):
            self.print_endline=False;
            return
        if '\r' in text:
           return;

        cursor = self.ui.plainTextEdit.textCursor()  
        cursor.movePosition(QTextCursor.End)
        if '\n' in text:
            if  self.print_endline==True:
                cursor.insertText(text)  
        else:
            cursor.insertText(f"{text}")  
            self.print_endline=True
        

    def LoadSavedConfig(self):
        cfg_options=ToolKit.GetSavedConfigData(cfg_file="train.cfg")
        if cfg_options is None:
            #print("cfg_options is none")
            return;

        self.ui.LineEditModelSaveDir.setText(cfg_options.get("ModelSaveDir",""))
        self.ui.LineEditTrainSrcFaceFolder.setText(cfg_options.get("SrcFaceFolder",""))
        self.ui.LineEditTrainDstFaceFolder.setText(cfg_options.get("DstFaceFolder",""))
        self.ui.LineEditTestSampleDir.setText(cfg_options.get("TrainTestFolder",""))
        #self.ui.LineEditXSegModelFolder.setText(cfg_options.get("XsegFolder",""))
     
    def SaveConfigData(self,txt=None):        
        options={}

        options["ModelSaveDir"]=self.ui.LineEditModelSaveDir.text();
        options["SrcFaceFolder"]=self.ui.LineEditTrainSrcFaceFolder.text();
        options["DstFaceFolder"]=self.ui.LineEditTrainDstFaceFolder.text();
        options["TrainTestFolder"]=self.ui.LineEditTestSampleDir.text();   
        ToolKit.SavedConfigDataDict(options,"train.cfg") 
        

    def btnOpenTrainSrcFolderClick(self):
        folder=self.ui.LineEditTrainSrcFaceFolder.text();  
        try:           
            os.startfile(folder)
        except:
            QMessageBox.information(None,"错误","打开失败,请检查文件夹是否存在");
    
    def btnOpenTrainDstFolderClick(self):
        folder=self.ui.LineEditTrainDstFaceFolder.text();   
        try:           
            os.startfile(folder)
        except:
            QMessageBox.information(None,"错误","打开失败,请检查文件夹是否存在");
    def btnBrowseTrainSrcFolderClick(self):
        folder=QFileDialog.getExistingDirectory(None,"选择Src文件夹",self.ui.LineEditTrainSrcFaceFolder.text());
        if(len(folder)>2):
            self.ui.LineEditTrainSrcFaceFolder.setText(folder);
            self.SaveConfigData()

    def btnBrowseTrainDstFolderClick(self):
        folder=QFileDialog.getExistingDirectory(None,"选择Dst文件夹",self.ui.LineEditTrainDstFaceFolder.text());
        if(len(folder)>2):
            self.ui.LineEditTrainDstFaceFolder.setText(folder);
            self.SaveConfigData()

    def btnOpenModelFolderClick(self):
        folder=self.ui.LineEditModelSaveDir.text();  
        try:           
            os.startfile(folder)
        except:
            QMessageBox.information(None,"错误","打开模型文件夹失败,请检查文件夹是否存在");
    
    def btnBrowseModelFolderClick(self):
        folder=QFileDialog.getExistingDirectory(None,"选择模型文件夹",self.ui.LineEditModelSaveDir.text());
        if(len(folder)>2):
            self.ui.LineEditModelSaveDir.setText(folder);
            self.SaveConfigData()

    def btnOpenTestFolderClick(self):
        folder=self.ui.LineEditTestSampleDir.text();  
        try:           
            os.startfile(folder)
        except:
            QMessageBox.information(None,"错误","打开测试文件夹失败,请检查文件夹是否存在");
    
    def btnBrowseTestFolderClick(self):
        folder=QFileDialog.getExistingDirectory(None,"选择测试文件夹","");
        if(len(folder)>2):
            self.ui.LineEditTestSampleDir.setText(folder); 
            self.SaveConfigData()
            self.LoadTestSampleFolder()
            self.LoadTestImage(force_idx=0)

    def LoadTestSampleFolder(self):
        folder=self.ui.LineEditTestSampleDir.text(); 
        if os.path.exists(folder) is False:
            QMessageBox.information(None,"错误","测试图片文件夹路径错误，目录不存在");
            return;
        self.TestImagesList=[file for file in os.listdir(folder) if ".jpg" in file]
        self.CurrTestIdx=0;
        self.ui.spinImageCount.setValue(len(self.TestImagesList))
        if len(self.TestImagesList)==0:
            QMessageBox.information(None,"错误","文件夹内没有找到jpg图片");
            return;
 

    TestImagesList=None;
    CurrTestIdx=0;
    def LoadTestImage(self,force=False,force_idx=0,offset=0):
        if self.TestImagesList is None:
            self.LoadTestSampleFolder()
        if self.TestImagesList is None:
            QMessageBox.information(None,"错误","测试图片集为空");
            return;
        if len(self.TestImagesList) ==0:
            QMessageBox.information(None,"错误","测试图片集数量为0");
            return;
        img_count=len(self.TestImagesList)
        if force is True:
            if force_idx==0:
                self.CurrTestIdx=0;
            if force_idx==-1:
                self.CurrTestIdx=img_count-1;
        if force is False:
            self.CurrTestIdx+=offset;
            #print(self.CurrTestIdx)
            self.CurrTestIdx=max(self.CurrTestIdx,0)
            self.CurrTestIdx=min(self.CurrTestIdx,img_count-1)
        img_filename=self.TestImagesList[self.CurrTestIdx]
        img_fullpath=self.ui.LineEditTestSampleDir.text()+"/"+img_filename
        img=cv2ex.cv2_imread(img_fullpath)
        if img is None:
            QMessageBox.information(None,"错误",f"读取图片出现异常：{img_filename}");
            return;
        self.preview_raw_image=img;
        #print(f"{self.CurrTestIdx}-{img_fullpath}-{img.shape}")
        ToolKit.ShowImageInLabel(self.preview_raw_image,self.ui.TestRawImage)
        self.ui.spinImageIndex.setValue(self.CurrTestIdx)

    #第一张
    def onFirstImgTestClick(self):
        self.LoadTestImage(force=True,force_idx=0)
        if self.ui.checkBoxAutoPreview.isChecked():
            self.btnTestPreviewClick()
    #最后张
    def onLastImgTestClick(self):
        self.LoadTestImage(force=True,force_idx=-1)
        if self.ui.checkBoxAutoPreview.isChecked():
            self.btnTestPreviewClick()
    #上一张
    def onPreImgTestClick(self):
        self.LoadTestImage(force=False,force_idx=0,offset=-1)
        if self.ui.checkBoxAutoPreview.isChecked():
            self.btnTestPreviewClick()
    #下一张
    def onNextImgTestClick(self):
        try:
            self.LoadTestImage(force=False,force_idx=0,offset=1)
            if self.ui.checkBoxAutoPreview.isChecked():
                self.btnTestPreviewClick()
        except :
            print("Error Load image")
        

    #截屏预览
    def btnCapScreenClick(self):
        left=self.ui.spinCapRectLeft.value()
        top=self.ui.spinCapRectTop.value()
        width=self.ui.spinCapRectWidth.value()
        height=self.ui.spinCapRectHeight.value()
        self.preview_raw_image=cap.capture_one_frame((left,top,width,height)); 
        ToolKit.ShowImageInLabel(self.preview_raw_image,self.ui.TestRawImage)
        #print(raw_image.shape)
        #cv2.imshow("cap",self.preview_raw_image)
        if self.ui.checkBoxAutoPreview.isChecked():
            self.btnTestPreviewClick()

    def btnTestPreviewClick(self):
        if self.preview_raw_image is None:
            QMessageBox.warning(None,"错误","未设置测试图，无法预览测试效果")
            return
        if self.Trainer is  None:
            QMessageBox.warning(None,"错误","模型未启动加载,无法预览测试效果")
            return        
        if self.Trainer.model is  None:
           QMessageBox.warning(None,"错误","模型未启动加载")
           return
        if self.Trainer.train_end is True:
           QMessageBox.warning(None,"错误","模型训练已经结束")
           return
        self.PreviewModelTest(self.Trainer.model,self.preview_raw_image)                
            
        
        

    def PreviewModelTest(self,model,raw_img):
        time_start=time.time()
        raw_img=raw_img.astype(np.float32)/255.0
        H,W,C=raw_img.shape
        rects = FaceEngines.getYoloEngineOnnx().extract (raw_img, threshold=0.5,fixed_window=256)[0]
        if  len(rects)==0:
            print("[S]图中未检测到人脸,无法预览测试效果");
            return
        l,t,r,b= rects[0]
        ul,ut,ur,ub=l/W,t/H,r/W,b/H
        face_urect=FRect.from_ltrb((ul,ut,ur,ub))
        face_rect_img, face_uni_mat=face_urect.cut(raw_img,1.4,256) 
        #face_align_img, src_to_align_uni_mat = sd.face_ulmrks_inSrc.cut(frame_image_src_float, coverage=sd.AlignCoverage, 
        #                    output_size=sd.AlignResolution,exclude_moving_parts=True)
        #aligned_to_source_uni_mat = src_to_align_uni_mat.invert()
        test_align_face=face_rect_img
        h,w,c=test_align_face.shape;
        resolution=model.options.get("resolution",256)
        if h !=resolution:
            test_align_face=cv2.resize(test_align_face,(resolution,resolution))
        if test_align_face.dtype == np.uint8:
                test_align_face = test_align_face.astype(np.float32)
                test_align_face /= 255.0
        pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm=model.get_test_preview(test_align_face)
        #print("pred_dst_dstm:",pred_dst_dstm.shape)
        out_dst_dst=pred_dst_dst.transpose((0,2,3,1))[0]
        out_dst_dstm=pred_dst_dstm.transpose((0,2,3,1))[0]
        out_dst_src=pred_src_dst.transpose((0,2,3,1))[0]
        out_dst_srcm=pred_src_dstm.transpose((0,2,3,1))[0]
        out_dst_dst=out_dst_dst*out_dst_dstm+(1-out_dst_dstm)*test_align_face
        out_dst_src=out_dst_src*out_dst_srcm+(1-out_dst_srcm)*test_align_face
        cv2.putText(out_dst_dst,"dst-pred-merge",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),1)
        cv2.putText(out_dst_src,"src-pred-merge",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),1)
        preview=np.hstack([test_align_face,out_dst_dst,out_dst_src])
        #print("out_dst_src:",out_dst_src.shape)
        time_end=time.time()
        time_use=round(time_end-time_start,2)
        #print(f"测试集图片预览已生成刷新，耗时{time_use}秒")
        cv2.imshow(sd.TrainPreviewWindowTitle,preview)
        


    #---- 新建模型
    def btnCreateNewModelClick(self):
        from PyQt5.QtWidgets import QSpinBox,QDialogButtonBox
        createDialog=QDialog()
        createDialog.setWindowFlags( Qt.WindowCloseButtonHint)
        createDialog.setWindowTitle("新建模型")
        formLayout=QFormLayout(createDialog)
        create_AeDims=QSpinBox(createDialog,);  create_eDims=QSpinBox(createDialog)
        create_dDims=QSpinBox(createDialog);    create_maskDims=QSpinBox(createDialog)
        archi_combo=QComboBox(createDialog);    create_resolution=QSpinBox(createDialog)

        create_AeDims.setRange(64,1024);    create_eDims.setRange(32,1024);
        create_dDims.setRange(64,1024); create_maskDims.setRange(18,128);
        archi_combo.addItems(["liae-ud","liae-u","liae-","liae-udt","liae-d","df","df-u","df-d","df-ud","df-udt",])
        create_resolution.setRange(96,1024)

        create_AeDims.setValue(256); create_eDims.setValue(64); 
        create_dDims.setValue(64);  create_maskDims.setValue(32);create_resolution.setValue(256)

        formLayout.addRow("archi",archi_combo)
        formLayout.addRow("resolution",create_resolution)
        formLayout.addRow("AeDims",create_AeDims)
        formLayout.addRow("eDims",create_eDims)
        formLayout.addRow("dDims",create_dDims)
        formLayout.addRow("maskDims",create_maskDims)
        

        buttonBox=QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        formLayout.addRow(buttonBox)
        
        buttonBox.accepted.connect(createDialog.accept)
        buttonBox.rejected.connect(createDialog.reject)

        if (createDialog.exec() == QDialog.Accepted):
            #print("Accept:",create_AeDims.value(),create_eDims.value())
            model_folder=self.ui.LineEditModelSaveDir.text();
            HasModel=self.CheckModelFolderValid(model_folder)
            if HasModel is True:
                QMessageBox.warning(createDialog,"错误","目录包含模型文件，请重新选择模型保存路径")
                return

            archi=archi_combo.currentText(); reso=create_resolution.value(); ae_dim=create_AeDims.value();
            e_dim=create_eDims.value(); d_dim=create_dDims.value(); mask_dim=create_maskDims.value()
            th=threading.Thread(target=self.CreateNewModelFilesThreadExec,args=(model_folder,archi,reso,ae_dim,e_dim,d_dim,mask_dim))
            th.daemo=True;
            th.start();

    def CreateNewModelFilesThreadExec(self,models_path,archi,resolution,ae_dim,e_dim,d_dim,mask_dim):
        print("开始创建新的模型文件")
        from vtmodel.FaceSwapModelSAE import FaceSwapModelSAE
        kwargs={"archi":archi,"resolution":resolution,"ae_dims":ae_dim,"e_dims":e_dim,"d_dims":d_dim,"d_mask_dims":mask_dim}
        model=FaceSwapModelSAE(saved_models_dir=models_path,kwargs=kwargs)
        model.create_model_nn_archi()
        model.define_tf_output_and_loss()
        model.empty_run_nn_init()
        model.save(init=True)
        print("结束创建模型文件")

    
            
    #--- 修改素材加载选项
    def btnApplySampleOptionsClick(self):
        if self.Trainer is None:
            return
        if self.Trainer.model is None:
            return
        options=self.Trainer.model.options
        if options is not None:
            options["random_warp"]=self.ui.checkBoxRandomWarp.isChecked();
            options["random_flip_src"]=self.ui.checkBoxRandomSrcFlip.isChecked(); 
            options["random_flip_dst"]=self.ui.checkBoxRandomDstFlip.isChecked();
            options["random_hsv_power"]=self.ui.spinHSVPower.value();
            self.Trainer.model.options["uniform_yaw"]=self.ui.checkBoxUniformYaw.isChecked();
            self.Trainer.model.options["blur_out_mask"]=self.ui.checkBoxBlurOutMask.isChecked();
    
     #--- 修改训练选项
    def btnApplyTrainOptionsClick(self):
        if self.Trainer is None:
            return
        if self.Trainer.model is None:
            return       
        self.Trainer.auto_backup_min=self.ui.spinAutoBackHour.value()*60;
        self.Trainer.target_iter=self.ui.spinTargetIteration.value();
        self.Trainer.model.options["write_preview_history"]=self.ui.checkBoxWritePreviewHistory.isChecked(); 
        self.Trainer.model.options["batch_size"]=self.ui.spinBatchSize.value(); 
        self.Trainer.model.options["learning_rate"]=self.ui.spinLearningRate.value()
        self.Trainer.model.options["masked_training"]=self.ui.checkBoxMaskTraining.isChecked()
        self.Trainer.model.options["models_opt_on_gpu"]=self.ui.checkBoxModelOnGPU.isChecked();
        self.Trainer.model.options["adabelief"]=self.ui.checkBoxAdaBelief.isChecked();
        self.Trainer.model.options["lr_dropout"]=self.ui.checkBoxLrDropout.isChecked() ;  
        self.Trainer.model.options["clipgrad"]=self.ui.checkBoxClipGradient.isChecked();             
        self.Trainer.model.options["eyes_mouth_prio"]=self.ui.checkBoxEyesMouthPrio.isChecked();
        #self.Trainer.model.options["bg_style_power"]=self.ui.spinBgStylePower.value();
        #self.Trainer.model.options["face_style_power"]=self.ui.spinFaceStylePower.value();
        self.Trainer.model.options["true_face_power"]=self.ui.spinTrueFacePower.value();
        self.Trainer.model.options["gan_power"]=self.ui.spinGanPower.value();
        self.Trainer.model.options["gan_patch_size"]=self.ui.spinGanPatchSize.value();
        self.Trainer.model.options["gan_dims"]=self.ui.spinGanDims.value();
    
        #print("ct_mode:",options.get('ct_mode', 'none'))

        #---------保存模型配置参数------------
    def btnSaveModelParamsClick(self):
        iter=self.ui.spinIteration.value();
        options={} 
        #--- saehd model first run
        options["resolution"]=self.ui.spinBoxModelResolution.value();
        options["face_type"]=self.ui.comboBoxModelFaceType.currentText();
        options["archi"]=self.ui.comboBoxModelArchi.currentText();
        options["ae_dims"]=self.ui.spinAutoEncoderDims.value();
        options["e_dims"]=self.ui.spinEncoderDims.value();
        options["d_dims"]=self.ui.spinDecodeDims.value();
        options["d_mask_dims"]=self.ui.spinMaskDims.value();
        options["masked_training"]=self.ui.checkBoxMaskTraining.isChecked(); #bool       
        model_data = {
            'iter': iter,
            'options': options,
            'loss_history': [],
            'sample_for_preview' : None,
            'choosed_gpu_indexes' : 0,
        }
        folder=self.ui.LineEditModelSaveDir.text();
        model_name=self.ui.comboBoxModelName.currentText();
        model_class_name=self.ui.comboBoxModelClass.currentText();
        if(len(model_name)<2):
            QMessageBox.warning(None,"提示",f"模型名称{model_name}需要至少2个字符")
            return;

        model_data_path=folder+"\\"+model_name+"_"+model_class_name+"_data.dat"

        if(os.path.exists(model_data_path)==True):
            QMessageBox.warning(None,"提示",f"模型文件{model_data_path}已经存在，将覆盖该文件")
           

        with open(model_data_path,'wb') as write_file:
            pickle.dump(model_data,write_file)
            write_file.close()
        QMessageBox.warning(None,"提示",f"已经成功写入文件{model_data_path}")
         

       
          
    #----打印显示模型参数
    def btnLoadModelParamsClick(self):
        print("[S]读取模型架构信息",end='\r')
        models_path = self.ui.LineEditModelSaveDir.text()
        if os.path.exists(models_path)==False:
            QMessageBox.warning(None,"Error","模型文件目录不存在");
            return;
        dat_file_fullpath=pathex.get_first_file_end_with(models_path,"_data.dat",exclude="eg_data",fullpath=True)
        if len(dat_file_fullpath)<4:
            QMessageBox.warning(None,"错误","模型目录未找到模型参数文件 *_data.dat");
            return;
         

        try:
            model_data=pickle.loads(Path(dat_file_fullpath).read_bytes())
            iter=model_data.get("iter",0)
            options=model_data.get("options",None)
            archi=options.get('archi',"liae-ud")
            resolution = options['resolution']
            ae_dims =options['ae_dims']
            e_dims =options['e_dims']
            d_dims =options['d_dims']
            d_mask_dims =options['d_mask_dims']
        except :
            QMessageBox.warning(None,"错误",f'''读取默认的dat模型配置文件发生错误.读取文件为{dat_file_fullpath},请检查模型配置文件路径是否正确.''');
            return;
        
      
        #---model archi

        self.ui.tableWidget.setItem(0,0,QTableWidgetItem(str(resolution)));
        self.ui.tableWidget.setItem(0,1,QTableWidgetItem(str(archi)));
        self.ui.tableWidget.setItem(0,2,QTableWidgetItem(str(ae_dims)));
        self.ui.tableWidget.setItem(0,3,QTableWidgetItem(str(e_dims)));
        self.ui.tableWidget.setItem(0,4,QTableWidgetItem(str(d_dims)));
        self.ui.tableWidget.setItem(0,5,QTableWidgetItem(str(d_mask_dims)));

        self.ui.spinBatchSize.setValue(options.get('batch_size', 8));
        self.ui.checkBoxWritePreviewHistory.setChecked(options.get('write_preview_history', False));
        self.ui.spinTargetIteration.setValue(options.get('target_iter', 8));
        self.ui.spinAutoBackHour.setValue(options.get('autobackup_hour', 2));
        self.ui.checkBoxRandomSrcFlip.setChecked(options.get('random_flip_src', True));
        self.ui.checkBoxRandomDstFlip.setChecked(options.get('random_flip_dst', True));


        self.ui.checkBoxModelOnGPU.setChecked(options.get('models_opt_on_gpu', False));
        self.ui.checkBoxMaskTraining.setChecked(options.get('masked_training', False));
        self.ui.checkBoxEyesMouthPrio.setChecked(options.get('eyes_mouth_prio', False));
        self.ui.checkBoxHighLossPrio.setChecked(options.get('high_loss_prio', True));
        self.ui.checkBoxAdaBelief.setChecked(options.get('adabelief',True));
        self.ui.checkBoxLrDropout.setChecked(options.get('lr_dropout',True)==True); 
        self.ui.checkBoxClipGradient.setChecked(options.get('clipgrad', False));
        self.ui.checkBoxRandomWarp.setChecked(options.get('random_warp', False));
       
        self.ui.checkBoxHSVShift.setChecked(options.get('random_hsv_shift', False));
        self.ui.spinTrueFacePower.setValue(options.get('true_face_power', 0));
        self.ui.spinGanPower.setValue(options.get('gan_power', 0));
        self.ui.spinGanPatchSize.setValue(options.get('gan_patch_size', 4));
        self.ui.spinGanDims.setValue(options.get('gan_dims', 4));
        print("[S]读取模型架构信息完成",end='\r')

    def btnNewModelNameClick(self):
        self.ui.comboBoxModelName.setEditable(True);

    def btnExportDFMClick(self):
        models_path = self.ui.LineEditModelSaveDir.text()
        if os.path.exists(models_path)==False:
            QMessageBox.warning(None,"Error","模型文件目录不存在");
            return;

        if self.Trainer is not None:
            if self.Trainer.isTraining==True:
                QMessageBox.warning(None,"Error","模型正在训练中，请关闭模型后再导出");
                return

        #获得包含文件路径+文件名的元组
        save_path,name = QFileDialog.getSaveFileName(None, '选择保存路径', 'unnamed_live.dfm', 'dfm(*.dfm)')
        if(len(save_path)<2):
            print("路径选择取消，导出过程取消")
            return;

        th=threading.Thread(target=self.ExportDfmThreadRun,args=(models_path,save_path))
        th.daemo=True;
        th.start();


    def ExportDfmThreadRun(self,models_path,save_path):
        print("开始实时转换模型的导出...")
        print("[S]正在导出模型，请稍等...",end='\r')
        from vtmodel.FaceSwapModelSAE import FaceSwapModelSAE
        params={"saved_models_dir":models_path,"masked_training":True}
        model = FaceSwapModelSAE(saved_models_dir=models_path,kwargs=params)
        model.export_dfm (save_path) 
        print(f"完成导出,输出到{save_path}")

    def btnBrowsePretrainModelFolderClick(self):
        folder=QFileDialog.getExistingDirectory(None,"选择预训练模型文件夹","F:/DeepModels");
        if(len(folder)>2):
            self.ui.LineEditPretrainModelFolder.setText(folder);
    def btnBrowsePretrainDataFolderClick(self):
        folder=QFileDialog.getExistingDirectory(None,"选择与训练数据文件夹","F:/");
        if(len(folder)>2):
            self.ui.LineEditPretrainDataFolder.setText(folder);

    
        

    #---- 开始训练模型-------------
    #--------------------------
    def btnStartTrainModelClick(self):

        if sd.AppEnable is False:
            QMessageBox.warning(None,"提示","Train使用功能过期,请进入启动器激活Train使用权限");
            return;

        training_data_src_path=self.ui.LineEditTrainSrcFaceFolder.text();
        training_data_dst_path=self.ui.LineEditTrainDstFaceFolder.text();
        saved_models_path=self.ui.LineEditModelSaveDir.text();
        
        if os.path.exists(training_data_src_path)==False:
           QMessageBox.information(None,"error","Src源人脸路径不存在");
           return;
        if os.path.exists(training_data_dst_path)==False:
           QMessageBox.information(None,"error","Dst目标人脸路径不存在");
           return;
        if os.path.exists(saved_models_path)==False:
           QMessageBox.information(None,"error","模型文件夹不存在");
           return;

       #----检查文件夹中是否存在模型
        folder_valid=self.CheckModelFolderValid(saved_models_path);
        if folder_valid is False:
           QMessageBox.information(None,"错误","模型文件夹中没有模型文件，请先创建模型或者复制预训练模型到文件夹中");
           return;

        kwargs={
                "training_data_src_path":training_data_src_path,
               "training_data_dst_path":training_data_dst_path,
                "saved_models_dir":saved_models_path,
                 
                  "src_data_subdir":self.ui.checkBoxSrcSubDir.isChecked(),
                  "dst_data_subdir":self.ui.checkBoxDstSubDir.isChecked(),
                  "random_hsv_shift":self.ui.checkBoxHSVShift.isChecked(),
                  "random_warp":self.ui.checkBoxRandomWarp.isChecked(),
                  "random_flip_src":self.ui.checkBoxRandomSrcFlip.isChecked(),
                  "random_flip_dst":self.ui.checkBoxRandomDstFlip.isChecked(),
                  "batch_size":self.ui.spinBatchSize.value(),

                  "high_loss_prio":self.ui.checkBoxHighLossPrio.isChecked(),
                  "masked_training":self.ui.checkBoxMaskTraining.isChecked(),                  
                  "eyes_mouth_prio":self.ui.checkBoxEyesMouthPrio.isChecked(),
                  "adabelief":self.ui.checkBoxAdaBelief.isChecked(),
                  "models_opt_on_gpu":self.ui.checkBoxModelOnGPU.isChecked(),
                  "true_face_power":self.ui.spinTrueFacePower.value(),                  
                  "clipgrad":self.ui.checkBoxClipGradient.isChecked(),
                  "gan_power":self.ui.spinGanPower.value(),
                  "gan_patch_size":self.ui.spinGanPatchSize.value(),
                  "gan_dims":self.ui.spinGanDims.value(),
                  "lr_dropout":self.ui.checkBoxLrDropout.isChecked(),
                  "write_preview_history":self.ui.checkBoxWritePreviewHistory.isChecked(),
                  "target_iter":self.ui.spinTargetIteration.value(),       
                  "ssim_power":self.ui.spinSSIM_Power.value(),
                  "mse_power":self.ui.spinMSE_Power.value(),
                  "emp_power":self.ui.spinEMP_Power.value(),
                  "mask_power":self.ui.spinMask_Power.value(),
                  "hloss_cycle":self.ui.spinHigLossCycle.value(),
                }
     
        self.Trainer=TrainerThread(kwargs)
        self.Trainer.mPreviewReadySignal.connect(self.onTrainPreviewReady);
        #self.Trainer.mModelLoadEndSignal.connect(self.displayOptionsDataForUI);
        self.Trainer.start()

        
    def CheckModelFolderValid(self,folder):
       files=pathex.get_file_paths_str(folder,subdir=False)
       dat_ok=False; encoder_ok=False; 
       for file in files:
           if "data.dat" in file:
               dat_ok=True;
           if "encoder.npy" in file:
                encoder_ok=True;
       return dat_ok and encoder_ok 

   #-----暂停训练
    def btnPauseRestoreTrainClick(self):
        if self.Trainer is None:
            return;
        if self.Trainer.train_pause is False:
            self.Trainer.train_pause=True;
            self.ui.btnPauseRestoreTrain.setText("恢复训练")
        else:
            self.Trainer.train_pause=False;
            self.ui.btnPauseRestoreTrain.setText("暂停训练")
    
    #-----测试训练效果预览
    def btnTrainTestPreviewClick(self):
        if self.Trainer is not None:
            self.Trainer.mCommandQueue.put ( {'op': 'test'} )
        else:
            QMessageBox.warning(None,"warning","The train has not be started, you can not do this command")
            return


    #-----训练预览图刷新
    def btnUpdateTrainPreviewClick(self):
        if self.Trainer is None:
            QMessageBox.warning(None,"warning","The train has not be started, you can not do this command")
            return
        
        self.Trainer.mCommandQueue.put ( {'op': 'preview'} )
        
    #-----（回调）预览图就绪、显示预览
    def onTrainPreviewReady(self):      
        if self.Trainer is None:
            QMessageBox.warning(None,"warning","The train has not be started, you can not do this command")
            return
        if self.Trainer.previews is None:
            QMessageBox.warning(None,"warning","The train has not be started, you can not do this command")
            return
        self.previews=self.Trainer.previews
        cv2.imshow("training_preview",self.previews[self.preview_idx])
        cv2.waitKey(1000)

    def onSwitchMaskPreview(self):  
        self.preview_idx=(self.preview_idx+1)%2;
        self.onTrainPreviewReady()

    #-----结束训练、关闭窗口
    def btnCloseTrainWindowClick(self):
        if self.Trainer is not None:
            self.Trainer.mCommandQueue.put ( {'op': 'close'} )
            cv2.destroyAllWindows()
        else:
            QMessageBox.warning(None,"warning","The train has not be started, you can not do this command")
            return

    #-----保存模型
    def btnSaveTrainModelClick(self):
        if self.Trainer is not None:
            self.Trainer.mCommandQueue.put ( {'op': 'save'} )
        else:
            QMessageBox.warning(None,"warning","The train has not be started, you can not do this command")
            return

    #-----备份模型
    def btnBackupTrainModelClick(self):
        if self.Trainer is not None:
            self.Trainer.mCommandQueue.put ( {'op': 'backup'} )
        else:
            QMessageBox.warning(None,"warning","The train has not be started, you can not do this command")
            return
         




    def  btnRenameModelFilesClick(self):
        print("重命名模型参数文件")
        new_model_name=self.ui.comboBoxModelName.currentText();
        model_folder=self.ui.LineEditModelSaveDir.text();

        if len(new_model_name)<2:
            QMessageBox.warning(None,"提示","模型名称太短")
            return;
       
        files_list=[file for file in os.listdir(model_folder) if os.path.isfile(os.path.join(model_folder,file)) ]
        files_list.sort()
        for file in files_list:
            #print(file)
            suffix=file[file.find('_'):]
            old_full_name=f"{model_folder}/{file}"
            new_file_name=f"{new_model_name}{suffix}"
            new_full_file_name=f"{model_folder}/{new_file_name}"
            if os.path.exists(new_full_file_name):
                print("文件名冲突：",file,new_file_name)
                continue
            print("重命名文件:",file,"改为：",new_file_name)
            os.rename(old_full_name,new_full_file_name)
        QMessageBox.warning(None,"提示","处理完成，可打开文件夹查看处理结果")

    

    def btnBrowseDfmModelClick(self):
        file,ok=QFileDialog.getOpenFileName(None,"选择dfm文件","","dfm files (*.dfm)");
        if ok:
            self.ui.lineEditDfmModelPath.setText(file);

    def btnExportVtfmClick(self):
        dfm_model_path=self.ui.lineEditDfmModelPath.text()
        if os.path.exists(dfm_model_path) is False:
            ToolKit.ShowWarningError(None,"错误","dfm模型文件不存在")
            return
        #--保存路径选择
        default_vtfm_path=self.ui.lineEditDfmModelPath.text().replace(".dfm",".vtfm")
        vtfm_model_path,name=QFileDialog.getSaveFileName(None,"保存vtfm文件",default_vtfm_path,"vtfm file(*.vtfm)")
        if len(vtfm_model_path)<2:
            return;

        #--构建meta信息，并签名保护
        
        dict={};
        dict["auth_mode"]="machine" if self.ui.radioMachineAuth.isChecked() else "pwd";
        dict["model_id"]=self.ui.lineEditModeID.text();
        dict["pwd_hash"]=auth.get_md5_hash_str(self.ui.lineEditModelPwd.text()+"ppp");
        dict["author"]=self.ui.lineEditModelAuthor.text();
        dict["auth_token"]="";
        auth.sign_dict(dict)

        print("[S]开始转换模型")
        #转换vtfm模型
        
        LiveSwapModel.conver_dfm_to_vtfm(dfm_model_path,vtfm_model_path,dict)
        #ToolKit.ShowWarningError(None,"提示","转换vtfm模型完成")
        print("[S]完成了模型转换")

    def btnBrowseVtfmModelClick(self):
        file,ok=QFileDialog.getOpenFileName(None,"选择vtfm文件","","vtfm files (*.vtfm)");
        if ok:
            self.ui.lineEditVtfmModelPath.setText(file);

    def WriteModelAuthCode(self,image):
        machine_sn=self.ui.lineEditVtfmMachineSN.text();
        if len(machine_sn)<7:
            QMessageBox.warning(None,"提示","机器码不正确，请核对授权的机器码")
            return;

        vtfm_file=self.ui.lineEditVtfmModelPath.text()
        if os.path.exists(vtfm_file) is False:
            QMessageBox.warning(None,"提示","模型路径不正确，文件不存在")
            return;
        info=LiveSwapModel.read_model_metadata(vtfm_file)
        #print(info)
        auth_mode=info.get("auth_mode","-")
        model_id=info.get("model_id","-")
        if "machine" in auth_mode:
            request_code=machine_sn[0:6]+"-"+auth.get_md5_hash_str(f"{model_id}")[0:4]
            right_token=auth.get_md5_hash_str(f"{request_code}{request_code}-zzz")[0:6]
            #print("request_code:",request_code)
            #print("right_token:",right_token)
            LiveSwapModel.write_model_metadata(vtfm_file,"auth_token",right_token)
            print("[S]已经向模型文件写入授权码")

    def btnBrowseDatClick(self):
        file,ok=QFileDialog.getOpenFileName(None,"选择dat文件","","dat files (*.dat)");
        if ok:
            self.ui.lineEditDatPath.setText(file); 


    def DeleteDatParamItem(self,image):
        dat_file_fullpath=self.ui.lineEditDatPath.text();
        if os.path.exists(dat_file_fullpath) is False:
            QMessageBox.warning(None,"提示","文件路径不正确，文件不存在")
            return;

        param_name=self.ui.lineEditParamName.text()
        model_data=pickle.loads(Path(dat_file_fullpath).read_bytes()) 
        options=model_data.get('options',None)
        if param_name in options:
                options.pop(param_name)
        with open(dat_file_fullpath,'wb') as f:
            data = pickle.dump(model_data,f)
        print(f"删除{param_name}项配置结束")

    def onTrunkHistoryClick(self):
        dat_file_fullpath=self.ui.lineEditDatPath.text();
        if os.path.exists(dat_file_fullpath) is False:
            QMessageBox.warning(None,"提示","文件路径不正确，文件不存在")
            return;
        model_data=pickle.loads(Path(dat_file_fullpath).read_bytes()) 
        loss_history=model_data.get('loss_history',[])
        if len(loss_history)==0:
            QMessageBox.warning(None,"提示","未能读取到loss历史")
            return;
        try:
            num=self.ui.spinReserveHistoryNum.value()
            num=0-num;
            loss_history=loss_history[num:-1]
            model_data["loss_history"]=loss_history
            with open(dat_file_fullpath,'wb') as f:
                data = pickle.dump(model_data,f)
            print("已经截取loss历史记录数据")
        except  Exception as ex :
            print(f"截取loss历史发生异常{ex}")
        
    def onTogglePreviewTopmost(self):
        try:
            import win32gui, win32con
            hwnd = win32gui.FindWindow(None, sd.TrainPreviewWindowTitle)
            if hwnd==0:
                return;
            left, top, right, bot = win32gui.GetWindowRect(hwnd)
            if self.ui.checkPreviewTopMost.isChecked(): 
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, left, top, right-left, bot-top, win32con.SWP_SHOWWINDOW) 
                print("[P]输出窗口设为置顶")
            else: 
                win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, left, top, right-left, bot-top, win32con.SWP_SHOWWINDOW) 
                print("[P]输出窗口取消置顶")
        except :
            pass