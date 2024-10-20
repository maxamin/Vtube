#coding=utf-8
# cython:language_level=3
from PyQt5.QtWidgets import QApplication,QFileDialog,QMessageBox,QLabel,QListWidgetItem;
from PyQt5.QtCore import QTimer,QSize;
from PyQt5.QtGui import QIcon,QPixmap,QImage;
import numpy as np

from pathlib import Path
import os,time,threading,pickle,shutil,cv2,numpy;
from core.leras.device import Device,Devices
import core.pathex as pathex
from facelib import FANExtractor,LandmarksProcessor,FaceType
from core.DFLIMG import DFLIMG,DFLJPG
from kit import ToolKit
from kit.BackendLoadThread import *
import core.cv2ex as cv2ex
from facelib.facenet import facenet
import kit.ShareData as sd


class PreviewUiHandler():

    def __init__(self,ui):
        self.ui=ui;
        Devices.initialize_main_env(); 
        #self.ShowAiMarkPreviewList(1,ReloadFileList=True);
      

    #---------设置预览人脸文件夹----------
    def btnBrowsePreviewFolderClick(self):
        folder=QFileDialog.getExistingDirectory(None,"选择人脸文件夹", os.path.dirname(self.ui.LineEditPreviewFolder.text()));
        if(len(folder)>2):
            self.ui.LineEditPreviewFolder.setText(folder);
            self.ShowAiMarkPreviewList(1,ReloadFileList=True)
    #------打开预览的人脸文件夹
    def btnOpenPreviewFaceFolderClick(self):
        folder=self.ui.LineEditPreviewFolder.text();  
        try:           
            os.startfile(folder)
        except:
            ToolKit.ShowWarningError(None,"错误","打开人脸文件夹失败,请检查文件夹是否存在");

    def btnFirstPageFaceClick(self): 
        self.ui.spinBoxPreviewStartNum.setValue(0)
        self.ShowAiMarkPreviewList(1,False);

    def btnLastFacePageClick(self):
        n=self.ui.spinBoxPreviewTotalNum.value();
        page_size=self.page_size;
        start_idx=(n//page_size)*page_size;
        self.ui.spinBoxPreviewStartNum.setValue(start_idx)
        self.ShowAiMarkPreviewList(0,False);

    def btnPrePageFaceClick(self):
        start_idx=self.ui.spinBoxPreviewStartNum.value(); 
        page_size=self.page_size;
        if(start_idx>=page_size):
            start_idx=start_idx-page_size;
            self.ui.spinBoxPreviewStartNum.setValue(start_idx);
            self.ShowAiMarkPreviewList();

    def btnNextPageFaceClick(self):
        start_idx=self.ui.spinBoxPreviewStartNum.value(); 
        img_count=self.ui.spinBoxPreviewTotalNum.value(); 
        page_size=self.page_size;
        if(start_idx<img_count-page_size):
            start_idx=start_idx+page_size;
            self.ui.spinBoxPreviewStartNum.setValue(start_idx); 
            self.ShowAiMarkPreviewList();

    def ReloadPreviewImages(self):
        self.ShowAiMarkPreviewList(ForcePage=0,ReloadFileList=True)

    def UpdateCurrentPageFace(self):
        self.ShowAiMarkPreviewList(ForcePage=0,ReloadFileList=False)
         
    page_size=55;
    folder_jpg_file_list=[];



    def onSetIconSize(self):
        size=self.ui.spinItemSize.value();
        self.ui.listWidgetFaceImages.setIconSize(QSize(int(size),int(size)))
        self.ui.listWidgetFaceImages.setGridSize(QSize(int(size+2),int(size+20)))


    #------------- 显示图片列表------------
    def ShowAiMarkPreviewList(self,ForcePage=0,ReloadFileList=False):
        face_dir=self.ui.LineEditPreviewFolder.text();
        if os.path.exists(face_dir)==False or len(face_dir)<2:
            QMessageBox.information(None,"Error","人脸文件夹不存在");
            return;

        if ReloadFileList is True:
            self.folder_jpg_file_list.clear();
            dir_and_files= os.listdir(face_dir)
            for file in dir_and_files: #遍历文件夹
                if os.path.isdir(file):
                    continue;
                if os.path.splitext(file)[1]==".jpg":
                    self.folder_jpg_file_list.append(file);
            self.ui.spinBoxPreviewTotalNum.setValue(len(self.folder_jpg_file_list))
        
        if ForcePage ==0:
            start_idx=self.ui.spinBoxPreviewStartNum.value();
            end_idx=start_idx+self.page_size;
        else:
            start_idx=(ForcePage-1)*self.page_size;
            end_idx=start_idx+self.page_size;

        start_idx=min(start_idx,len(self.folder_jpg_file_list)-1)
        end_idx=min(end_idx,len(self.folder_jpg_file_list)); 
        grid_show_jpg_files_list=self.folder_jpg_file_list[start_idx:end_idx]

        show_landmark=self.ui.checkBoxPreviewLandmarks.isChecked();
        show_xseg_mask=self.ui.checkBoxPreviewSegMask.isChecked();
        
        not_exist_img=cv2ex.cv2_imread("../SrcDist/ui/icons/delete.jpg");
        if not_exist_img is None:
            not_exist_img=np.zeros([256,256,3],dtype=np.uint8)

        self.ui.listWidgetFaceImages.clear()
        for idx,jpg_file in enumerate(grid_show_jpg_files_list):
             
            jpg_file_fullname=face_dir+"/"+jpg_file;
            if os.path.exists(jpg_file_fullname) ==False:
                img_list_item=QListWidgetItem(self.GetIconFromNumpyImage(not_exist_img),jpg_file)
                self.ui.listWidgetFaceImages.addItem(img_list_item)
                continue;

            img=cv2ex.cv2_imread(jpg_file_fullname); 
            dfl_info=DFLJPG.load(jpg_file_fullname)            
            img=self.GetDrawMarkFaceImage(img,dfl_info,show_landmark,False,show_xseg_mask)
            img_list_item=QListWidgetItem(self.GetIconFromNumpyImage(img),jpg_file)
            self.ui.listWidgetFaceImages.addItem(img_list_item)
       
            #print(label.name)

    #-- 转为Icon
    def GetIconFromNumpyImage(self,img):
        img_src = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        temp_imgSrc = QImage(img_src, img_src.shape[1], img_src.shape[0], QImage.Format_RGB888)
        pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc)
        return QIcon(pixmap_imgSrc)

    #-------- 获取带遮罩的人脸图------
    def GetDrawMarkFaceImage(self,img,dfl_info,show_landmark=True,show_hull_mask=False,show_xseg_mask=True):
        if img is None or dfl_info is None:
            return img;
        if dfl_info.has_data()==False:
            img=cv2.putText(img, f"NoFaceAlign", (20,150 ), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            return img;
        if show_landmark is True:
            landmarks=dfl_info.get_landmarks();
            if landmarks is not None:
                LandmarksProcessor.draw_landmarks(img,landmarks,transparent_mask=False);
        
        if  show_hull_mask  is True:
                landmarks=dfl_info.get_landmarks();
                if landmarks is not None:
                    mask = LandmarksProcessor.get_image_hull_mask (img.shape, landmarks )
                    mask=np.repeat(mask,repeats=3,axis=2)
                    mask[:,:,1]=0.5;
                    img[...] = ( img * (1-mask) + img * mask / 2 )[...]

        if show_xseg_mask is True:
            mask=dfl_info.get_xseg_mask()
            if mask is not None:
                iw,ih,ic=img.shape
                mask=np.repeat(mask,repeats=3,axis=2)
                #mask[:,:,2]=0.5;
                mask=cv2.resize(mask,(iw,ih))  
                img[...] = ( img * (1-mask)*0.4 + img * mask )[...]

        return img; 

 
        
    #-----双击图片处理---
    def onDoubleClickImage(self):
        con_1 = self.ui.listWidgetFaceImages.selectedItems()[0].text() 
        #print("双事件：",con_1)
        file_name = self.ui.listWidgetFaceImages.selectedItems()[0].text() 
        full_path= self.ui.LineEditPreviewFolder.text()+"/"+file_name;
        if os.path.exists(full_path) is False:
            self.ui.statusbar.showMessage(f"{file_name}图片已经不存在",200000)
            return
        img=cv2ex.cv2_imread(full_path); 
        dfl_info=DFLJPG.load(full_path) 

        show_landmark=self.ui.checkBoxPreviewLandmarks.isChecked(); 
        show_xseg_mask=self.ui.checkBoxPreviewSegMask.isChecked();         
        mask_merge_img=self.GetDrawMarkFaceImage(img,dfl_info,show_landmark,False,show_xseg_mask)
        cv2.imshow("image_preview_mask",mask_merge_img)

    #--单击图片处理----
    def onSingleClickImage(self):
        file_name = self.ui.listWidgetFaceImages.selectedItems()[0].text() 
        full_path= self.ui.LineEditPreviewFolder.text()+"/"+file_name;
        if os.path.exists(full_path) is False:
            self.ui.statusbar.showMessage(f"{file_name}图片已经不存在",200000)
            return
          
        img=cv2ex.cv2_imread(full_path); 
        dfl_info=DFLJPG.load(full_path)
        if dfl_info is None or dfl_info.has_data()==False:
            self.ui.statusbar.showMessage(f"{file_name}图片中不包含人脸元数据",200000)
            return

        if dfl_info.has_data():
           
            has_landmarks="无"; landmark_shape = 0;
            pitch=None;yaw=None;xseg_shape=None;
        
            dict= dfl_info.get_dict() 
            if "landmarks" in dict :
                ldmks=dfl_info.get_landmarks();
                landmark_shape=ldmks.shape[0] 
                has_landmarks="有"; 
            if "pitch" in dict :
                pitch=dict.get("pitch",0.001) 

            if "yaw" in dict :
                yaw=dict.get("yaw",0.001) 
                 
            if dfl_info.has_xseg_mask() is True: 
                xseg_shape=dfl_info.get_xseg_mask().shape; 
                has_xseg_mask="有"
            
            dfl_info_str=f"{file_name}|| 图片大小:{img.shape}||人脸特征点:({has_landmarks},{landmark_shape}点) || 遮罩({xseg_shape}) || 角度(pitch:{pitch},yaw:{yaw})"
            
            self.ui.statusbar.showMessage(dfl_info_str,200000)    
            self.ui.listWidgetFaceImages.setToolTip(dfl_info_str)
        


    #----------获取选择的图片完整路径----
    def getSelectImageFullPath(self):
        folder=self.ui.LineEditPreviewFolder.text();
        file_name = self.ui.listWidgetFaceImages.selectedItems()[0].text() 
        return folder+"/"+file_name;
    
 

    #-----写入当前图片的角度信息-----
    def btnWriteFaceAnglesClick(self):
        full_path=self.getSelectImageFullPath();
        if os.path.exists(full_path) ==False:
            ToolKit.ShowWarningError(None,"","image path invalid(图像路径错误)")
            return;
        ok,tips=ToolKit.WriteFaceAngleForDflImageFile(full_path)
        ToolKit.ShowWarningError(None,"提示",tips)

        #--删除选择图片
    def onDeleteSelectImage(self):
        full_path=self.getSelectImageFullPath();
        if os.path.exists(full_path) ==False:
            ToolKit.ShowWarningError(None,"","image path invalid(图像路径错误)")
            return;

        if self.ui.listWidgetFaceImages.selectedItems() is None:
            return

        delete_move_dir=self.ui.LineEditPreviewFolder.text()+"/delete"
        delete_move_filepath=delete_move_dir+"/"+self.ui.listWidgetFaceImages.selectedItems()[0].text() ;
        if os.path.exists(delete_move_dir) is False:
            os.mkdir(delete_move_dir)
        shutil.move(full_path,delete_move_filepath)
        #self.ReloadPreviewImages()
        not_exist_img=cv2ex.cv2_imread("../SrcDist/ui/icons/delete.jpg");
        self.ui.listWidgetFaceImages.selectedItems()[0].setIcon(self.GetIconFromNumpyImage(not_exist_img))

    def ChangePageSize(self):
        self.page_size=self.ui.spinPageSize.value();

    #-------------生成角度分布图----
    def GenerateFaceAngleMap(self):
        angle_list=[];
        folder=self.ui.LineEditPreviewFolder.text();
        import core.pathex as pathex
        img_list=pathex.get_image_paths(folder,[".jpg"])
        for img_path in img_list:
            #print(img_path)
            dflimg=DFLJPG.load(img_path)
            if dflimg is None:
                continue;
            if dflimg.has_data()==False:
                continue;
            dict=dflimg.get_dict()
            if "pitch" in dict and  "yaw" in dict:
                pitch=round(dict["pitch"]); yaw=round(dict["yaw"])
                angle_list.append((pitch,yaw,img_path))
                continue;
            elif "landmarks" in dict:
                landmarks=dict["landmarks"]
                pitch, yaw, roll=LandmarksProcessor.estimate_pitch_yaw_roll_by_degree(landmarks)
                dict["pitch"]=pitch
                dict["yaw"]=yaw
                dict["roll"]=roll
                dflimg.save()
                angle_list.append((dict["pitch"],dict["yaw"],img_path))

        

        degree_pixel=8; pitch_limit=40;  yaw_limit=60;
        width=yaw_limit*degree_pixel*2;
        height=pitch_limit*degree_pixel*2
        angle_map_img=np.zeros((height,width,3),np.uint8)
        #cv2.putText(angle_map_img, folder, (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(angle_map_img, "Yaw(Turn Left/Right)", (10,height//2-20 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for yaw_tick in range(-yaw_limit,yaw_limit,5):
            cv2.putText(angle_map_img, str(yaw_tick), ((yaw_tick+yaw_limit)*degree_pixel,height//2 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(angle_map_img, "Pitch (Turn Up/Down)", (width//2+20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        for pitch_tick in range(-pitch_limit,pitch_limit,5):
            cv2.putText(angle_map_img, str(pitch_tick), (width//2,height-(pitch_tick+pitch_limit)*degree_pixel ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        for info in angle_list:
            pitch=info[0];  yaw=info[1];
            pos_x=(yaw+yaw_limit)*degree_pixel;
            pos_y=(pitch+pitch_limit)*degree_pixel;
            #print(info)
            cv2.circle(angle_map_img, (pos_x, height-pos_y), 2, (0,0,255), lineType=cv2.LINE_AA)
        cv2.imshow("Pitch-Yaw-Map",angle_map_img)
        cv2.waitKey(0)


    #----打开遮罩编辑器-----
    def  OpenPreviewXsegEditor(self):
        if sd.AppEnable is False:
            QMessageBox.warning(None,"提示","Lab使用功能过期,请进入启动器激活Lab使用权限");
            return;
        folder=self.ui.LineEditPreviewFolder.text()        
        ToolKit.OpenXsegEditor(folder)

    #---写入遮罩----
    def btnApplyXSegClick(self):
        
        input_dir=self.ui.LineEditPreviewFolder.text();
        if os.path.exists(input_dir)==False:
            QMessageBox.information(None,"Error","人脸文件夹不存在");
            return;
      
        th=threading.Thread(target=self.ApplyXsegThreadRun,args=(input_dir,))
        th.daemo=True;
        th.start();

    def ApplyXsegThreadRun(self,input_dir):
        print("[P]开始处理遮罩",end='\r')
        images_paths = pathex.get_image_paths(input_dir, return_Path_class=True)
        total=len(images_paths)
        idx=0;
        for filepath in images_paths:
            dflimg = DFLIMG.load(filepath)
            ToolKit.GetXsegMaskForFaceImage(dflimg)
            dflimg.save()
            idx=idx+1;
            print(f"[P]{filepath}保存遮罩成功[{idx}/{total}]",end='\r')