#coding=utf-8
# cython:language_level=3

from PyQt5.QtWidgets import QApplication,QFileDialog,QMessageBox,QLabel,QColorDialog;
from PyQt5.QtCore import Qt;
from PyQt5.QtGui import QColor;
import numpy as np
from pathlib import Path
import os,time,threading,pickle,shutil,cv2,numpy;
from core.leras import nn
import core.pathex as pathex
from facelib import FANExtractor,LandmarksProcessor,FaceType
from core.DFLIMG import DFLIMG,DFLJPG
from kit import ToolKit
import core.cv2ex as cv2ex
from facelib.facenet.facenet import facenet
import kit.ShareData as sd
import kit.RandomOcclude as RandomOcclude


class AdjustUiHandler():

    def __init__(self,ui):
        self.ui=ui;
        
    

    #---------浏览调整人脸文件夹----------
    def btnBrowseAdjustFolderClick(self):
        folder=QFileDialog.getExistingDirectory(None,"选择人脸文件夹",os.path.dirname(self.ui.LineEditAdjustFolder.text()));
        if(len(folder)>2):
            self.ui.LineEditAdjustFolder.setText(folder);

    #------打开调整人脸文件夹
    def btnOpenFaceFolderClick(self):
        folder=self.ui.LineEditAdjustFolder.text();  
        try:           
            os.startfile(folder)
        except:
            ToolKit.ShowWarningError(None,"错误","打开人脸文件夹失败,请检查文件夹是否存在");
 
    def  btnCombineSubDirsClick(self):
        face_folder=self.ui.LineEditAdjustFolder.text();
        th=threading.Thread(target=self.CombineSubDirsThreadRun,args=(face_folder,))
        th.daemo=True;
        th.start();

    def CombineSubDirsThreadRun(self,folder):
        if os.path.exists(folder) is False:
            return
        for root, dirs, files in os.walk(folder):
            for file in files:
                src_file = os.path.join(root, file)
                shutil.move(src_file, folder+"/"+file)
                print(f"[P]move:{src_file}",end='\r')

    def VerifyFaceLabelClick(self,m,e):
        print("VerifyFaceLabelClick")
        self.btnBrowseVerifyFaceClick()


    
    def btnAddRefFaceImageClick(self):
        file,ok=QFileDialog.getOpenFileName(None,"选择人脸文件",self.ui.LineEditAdjustFolder.text(),'image files (*.jpg *.png )');
        if ok:
           face_img= cv2ex.cv2_imread(file);
           facenet.add_ref_face_img(face_img);
           ToolKit.ShowImageInLabel(facenet.getRefImagesMerge(),self.ui.AdjustRefLabel,bgr_cvt=True,scale=True)

    def btnDeleteLastRefClick(self):
        facenet.del_last_face_img()
        ToolKit.ShowImageInLabel(facenet.getRefImagesMerge(),self.ui.AdjustRefLabel,bgr_cvt=True,scale=True)

    def btnClearIdentityFilterFaceClick(self):
        facenet.clear_standard_face();
        self.ui.AdjustRefLabel.clear() 

    def btnStartPersonVerifyClick(self): 
        folder=self.ui.LineEditAdjustFolder.text();
        if(os.path.exists(folder)==False):
            ToolKit.ShowWarningError(None,"错误","文件夹未设置");
            return;
        move_folder=f"{folder}/others"
        if os.path.exists(move_folder) ==False:
            os.mkdir(move_folder)

        th=threading.Thread(target=self.VerifyFilterFacesThreadRun,args=(folder,move_folder))
        th.daemo=True;
        th.start();
   
    def VerifyFilterFacesThreadRun(self,folder,move_folder):
        files=[file for file in os.listdir(folder) if (".jpg" in file or ".png" in file) ]
        for file in files:
            file_full_path=f"{folder}/{file}"            
            img=cv2ex.cv2_imread(file_full_path)
            ok,dist=facenet.getVerifyResult(img,self.ui.spinAdjustVerifyThreshold.value())
            print(f"[P]检查:{file},相似度:{round(dist,2)},同一人:{ok}",end='\r')
            if ok is False:
                move_file_fullpath=f"{move_folder}/{file}"
                shutil.move(file_full_path,move_file_fullpath)
        print(f"完成{folder}文件夹身份筛选 ")

    def btnAddPrefixClick(self):
        prefix=self.ui.lineEditInsertWord.text();
        face_folder=self.ui.LineEditAdjustFolder.text();
        files = [file for file in os.listdir(face_folder) if (".jpg" in file or ".png" in file)] 
        for file in files:
            old_name=face_folder+"/"+file;
            new_name=face_folder+"/"+prefix+"_"+file;
            print("file:",old_name,new_name)
            os.rename(old_name,new_name)
        ToolKit.ShowWarningError(None,"tips","添加文件名前缀完成");
    
    def btnAddPostfixClick(self):
        postfix=self.ui.lineEditInsertWord.text();
        face_folder=self.ui.LineEditAdjustFolder.text();
        files = [file for file in os.listdir(face_folder) if (".jpg" in file or ".png" in file)] 
        for file in files:
            old_name=face_folder+"\\"+file;
            root,ext=os.path.splitext(file) 
            new_name=face_folder+"\\"+root+postfix+ext;
            print(f"[P]添加后缀:{old_name},{new_name}",end='\r')
            os.rename(old_name,new_name)
        ToolKit.ShowWarningError(None,"tips","添加文件名后缀完成");

    def btnRenameClick(self):
        key_word=self.ui.LineEditNameKeyWord.text();
        if(len(key_word)==0):
            ToolKit.ShowWarningError(None,"tips","要替换的字符为空"); 
            return;
        replace_word=self.ui.lineEditReplaceWord.text();
        face_folder=self.ui.LineEditAdjustFolder.text();
        if(os.path.exists(face_folder)==False):
            ToolKit.ShowWarningError(None,"tips",f"{face_folder}文件夹不存在");
            return;
        try: 
            for dir,subdirs,files in os.walk(face_folder):
                for file in files:
                    old_name=dir+"\\"+file;
                    new_name=dir+"\\"+file.replace(key_word,replace_word);
                    print(f"[P]rename file:{old_name} {file}",end='\r')
                    os.rename(old_name,new_name)
        except:
            ToolKit.ShowWarningError(None,"错误","替换失败、发生错误、可能出现了文件重名");
        ToolKit.ShowWarningError(None,"提示",f"重命名完成");

    def btnResizeFaceImagesClick(self):
        face_folder=self.ui.LineEditAdjustFolder.text(); 
        th=threading.Thread(target=self.ResizeThreadRun,args=(face_folder,))
        th.daemo=True;
        th.start();
    def ResizeThreadRun(self,face_folder):
        import cv2;
        folder=self.ui.LineEditAdjustFolder.text();
        resize_folder=folder+"/resize/";
        if(os.path.exists(resize_folder)==False):
            os.mkdir(resize_folder)
        size=self.ui.spinBoxAdjustFaceSize.value();
        print(f"调整文件夹 {folder} 图片，到{size}像素大小",)
        files=[file for file in os.listdir(folder) if ".jpg" in file]
        for file in files:
            old_img_path=f"{folder}/{file}"
            new_img_path=f"{resize_folder}/{file}"
            dfl_info_old=DFLJPG.load(old_img_path)
            old_image=dfl_info_old.get_img()
            img_resize=cv2.resize(old_image,(size,size))

            old_size,w,c=old_image.shape
            ratio=float(size)/float(old_size)
            lmkrs=dfl_info_old.get_landmarks()
            if lmkrs is not None:
                lmkrs=lmkrs*ratio
                dfl_info_old.set_landmarks(lmkrs.astype("uint"))

            cv2ex.cv2_imwrite(new_img_path,img_resize)
            dfl_info_new=DFLJPG.load(new_img_path)
            dfl_info_new.dfl_dict=dfl_info_old.dfl_dict;
            dfl_info_new.save()
            print(f"[P]处理图片{file}缩放完成",end='\r')

    def OnSetBgColor(self):
       color:QColor= QColorDialog.getColor(Qt.gray,None,"选择背景颜色") 
       #self.ui.btnSetEraseBgColor.setStyleSheet(f"QPushButton:\{background-color:rgb({color.red},{color.green},{color.blue})\}")

    def btnEraseMaskBgClick(self):
        face_folder=self.ui.LineEditAdjustFolder.text(); 
        th=threading.Thread(target=self.EraseMaskBgRun,args=(face_folder,))
        th.daemo=True;
        th.start();


    def EraseMaskBgRun(self,face_folder):
        import cv2;
        folder=self.ui.LineEditAdjustFolder.text();
        erase_folder=folder+"/bg_erase/";
        if(os.path.exists(erase_folder)==False):
            os.mkdir(erase_folder)


        files=[file for file in os.listdir(folder) if file.endswith(".jpg")]
        for file in files:
            old_img_path=f"{folder}/{file}"
            new_img_path=f"{erase_folder}/{file}"

            dfl_info_old=DFLJPG.load(old_img_path)
            if dfl_info_old is None:
                continue;
            if dfl_info_old.has_data()==False:
                continue;
            if dfl_info_old.has_xseg_mask() is False:
                continue;
            old_image=dfl_info_old.get_img()
            old_mask=dfl_info_old.get_xseg_mask()
            w,h,c=old_image.shape
            old_mask=cv2.resize(old_mask,(w,h))
            old_mask=np.expand_dims(old_mask,2)
            bg=np.ones(old_image.shape,dtype=np.uint8)*128
            new_img=old_image*old_mask+(1-old_mask)*bg
            cv2.imshow("bg_erase",new_img)
            cv2.waitKey(10)
            cv2ex.cv2_imwrite(new_img_path,new_img)
            dfl_info_new=DFLJPG.load(new_img_path)
            dfl_info_new.dfl_dict=dfl_info_old.dfl_dict;
            dfl_info_new.save()
            print(f"[P]{file}擦除背景完成",end='\r')
        print(f"[全部文件擦除背景完成")


    def btnStartFaceSortClick(self):
        pass

    def btnPackClick(self):
        face_folder=self.ui.LineEditAdjustFolder.text(); 
        th=threading.Thread(target=self.PackThreadRun,args=(face_folder,))
        th.daemo=True;
        th.start();
    def PackThreadRun(self,face_folder):
        from core.samplelib import PackedFaceset
        PackedFaceset.pack( Path(face_folder) );

       


    def btnUnPackClick(self):
        face_folder=self.ui.LineEditAdjustFolder.text(); 
        file,ok=QFileDialog.getOpenFileName(None,"选择pak文件",self.ui.LineEditAdjustFolder.text(),'faceset files (*.pak  )');
        if ok:
            th=threading.Thread(target=self.UnpackThreadRun,args=(file,face_folder))
            th.daemo=True;
            th.start();
    def UnpackThreadRun(self,samples_dat_path,face_folder):
        from core.samplelib.PackedFaceset import PackedFaceset
        from core.interact import interact as io

        samples_dat_path = Path(samples_dat_path)
        samples_path=samples_dat_path.parent
        if not samples_dat_path.exists():
            io.log_info(f"{samples_dat_path} : file not found.")
            return

        samples = PackedFaceset.load(samples_dat_path,isPakFile=True)
        if samples is None:
            print(f"load no samples in {samples_dat_path}")
            return

        for sample in io.progress_bar_generator(samples, "Unpacking"):
            person_name = sample.person_name
            if person_name is not None:
                person_path = samples_path / person_name
                person_path.mkdir(parents=True, exist_ok=True)

                target_filepath = person_path / sample.filename
            else:
                target_filepath = samples_path / sample.filename

            with open(target_filepath, "wb") as f:
                f.write( sample.read_raw_file() )

        samples_dat_path.unlink()


    def btnDenoiseClick(self):
        face_folder=self.ui.LineEditAdjustFolder.text();
        th=threading.Thread(target=self.DenoiseThreadRun,args=(face_folder,))
        th.daemo=True;
        th.start();

    def DenoiseThreadRun(self,face_folder):
        #VideoEd.denoise_image_sequence (face_folder, 10)
        print("------------------Finish Denoise images----------------")

 

    def btnContourizeFacesClick(self):
        face_folder=self.ui.LineEditAdjustFolder.text();
        th=threading.Thread(target=self.ContourizeThreadRun,args=(face_folder,))
        th.daemo=True;
        th.start();

    def ContourizeThreadRun(self,folder):
        files=[file for file in os.listdir(folder) if ".jpg" in file]
        files=files[0:10]
        move_folder=f"{folder}_contourize"
        for idx,file in enumerate(files):
            print(f"process {file}........,[{idx}/{len(files)}")
            file_full_path=f"{folder}/{file}"      
            save_full_path=f"{move_folder}/{file}"

            dfl_info_old=DFLJPG.load(file_full_path)
            img=dfl_info_old.get_img()
            gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
           


    def btnPartMoveClick(self):
        face_folder=self.ui.LineEditAdjustFolder.text();
        size=self.ui.spinPickSize.value();
        num=self.ui.spinPickNum.value();
        th=threading.Thread(target=self.PartMoveThreadRun,args=(face_folder,size,num))
        th.daemo=True;
        th.start();
    def PartMoveThreadRun(self,folder,size,num):
        files=[file for file in os.listdir(folder) if ".jpg" in file]
        move_folder=f"{folder}/move"
        if os.path.exists(move_folder)   is False:
            os.mkdir(move_folder)
        idx=0;
        for file in files:
            file_full_path=f"{folder}/{file}"  
            move_full_path=f"{move_folder}/{file}"  
            if (idx % size)<num:
                shutil.move(file_full_path,move_full_path)
                print(f"[P]move file:{file}",end='\r')
            idx=idx+1;
     
    def btnFilterByNameClick(self):
        face_folder=self.ui.LineEditAdjustFolder.text();
        word=self.ui.LineEditFilterWord.text();
        if len(word)<2:
            ToolKit.ShowWarningError(None,"错误","查找字符太短")
            return
        th=threading.Thread(target=self.FilterByNameThreadRun,args=(face_folder,word))
        th.daemo=True;
        th.start();
    def FilterByNameThreadRun(self,folder,word):
        import datetime
        files=[file for file in os.listdir(folder) if ".jpg" in file]
        timestr=datetime.datetime.now().strftime("%H%M%S")
        move_folder=f"{folder}/Filter{timestr}"
        if os.path.exists(move_folder)   is False:
            os.mkdir(move_folder)

        idx=0;
        total_num=len(files)
        for file in files:
            idx=idx+1;
            if word in file:
                file_full_path=f"{folder}/{file}"  
                move_full_path=f"{move_folder}/{file}"  
                shutil.move(file_full_path,move_full_path)
            print(f"[P]处理图片{file},{idx}/{total_num}",end='\r')


    def btnRestoreNameClick(self):
        folder=self.ui.LineEditAdjustFolder.text();
        for dir,subdirs,files in os.walk(folder):
            for file in files:
                old_name=dir+"\\"+file;
                pos=file.find("-");
                if pos>0:
                    new_name=file[pos+1:];
                    new_name='%s\\%s' % (dir,new_name)
                    os.rename(old_name,new_name)
                    #print(file,new_name)

    def btnClassByGenderClick(self):
        face_folder=self.ui.LineEditAdjustFolder.text(); 
        th=threading.Thread(target=self.ClassByGenderThreadRun,args=(face_folder,))
        th.daemo=True;
        th.start();
    def ClassByGenderThreadRun(self,face_folder):
        from deepface import DeepFace  

        face_folder=self.ui.LineEditAdjustFolder.text();
        man_folder=face_folder+"\\Man\\"
        woman_folder=face_folder+"\\Woman\\"
        if(os.path.exists(man_folder)==False):
            os.mkdir(man_folder)
        if(os.path.exists(woman_folder)==False):
            os.mkdir(woman_folder)
        for dir,subdirs,files in os.walk(face_folder):
            for file in files:
                img_path=dir+"\\"+file;
                man_name=man_folder+file;
                woman_name=woman_folder+file; 
                try:
                    analysis = DeepFace.analyze(img_path, actions=[ "gender"],models = None, enforce_detection = False, detector_backend = 'opencv')
                    gender=analysis.get("gender","Man");
                    #print(gender) 
                    if(gender=="Man"):
                        print(img_path,man_name)
                        shutil.move(img_path,man_name) 
                    else:
                        print(img_path,woman_name)
                        shutil.move(img_path,woman_name) 
                except:
                    print(img_path,"人脸性别判断，错误")
                    continue;

  

    def btnVerifyChooseFaceClick(self):
        face_file=self.ui.LineEditVerifyFace.text();
        face_folder=self.ui.LineEditRecognizeDirectory.text();
        if(os.path.exists(face_file)==False ):
            ToolKit.ShowWarningError(None,"错误","选取的人脸文件不存在"+face_file);
            return;
        if(os.path.exists(face_folder)==False):
            ToolKit.ShowWarningError(None,"错误","选取的人脸文件夹不存在"+face_folder);
            return;
        thres=self.ui.spinVerifyThreshold.value();
        model_name=self.ui.comboVerifyModel.currentText();
        
        th=threading.Thread(target=self.VerifyChooseFaceThreadRun,args=(face_folder,face_file,thres))
        th.daemo=True;
        th.start();

        #th=threading.Thread(target=self.RecognizeFaceThread,args=(face_folder,face_file))
        #th.daemo=True;
        #th.start();

    def RecognizeFaceThread(self,face_folder,face_file):
        for dir,subdirs,files in os.walk(face_folder):
            for file in files:
                print(file)
                img_path=dir+"\\"+file;
                img=cv2.imread(img_path)
                cv2.imshow("img",img)
                cv2.waitKey(1000)

    def VerifyChooseFaceThreadRun(self,face_folder,face_file,thres):
        print("开始挑选人脸照片");
        ref_face=cv2.imread(face_file)
        

    def btnOpenXsegEditorForOcclude(self):
        if sd.AppEnable is False:
            QMessageBox.warning(None,"提示","Lab使用功能过期,请进入启动器激活Lab使用权限");
            return;  
        ToolKit.OpenXsegEditor(os.path.abspath("../Occlusion"))




    #---Copies faces containing XSeg polygons in <input_dir>_xseg dir.
    def btnFetchXSegClick(self):
        print("btnFetchXSegClick")
        from core.mainscripts import XSegUtil
        input_dir=self.ui.LineEditAdjustFolder.text();
        if os.path.exists(input_dir)==False:
            QMessageBox.information(None,"Error","人脸文件夹不存在");
            return;
        XSegUtil.fetch_xseg (Path(input_dir))
    #---Remove applied XSeg masks from the extracted faces
    def btnRemoveXSegClick(self):
        print("btnRemoveXSegClick")

    def btnTrainXSegClick(self):
        print("btnTrainXSegClick")

    def btnBrowseXSegFolderClick(self):
        folder=QFileDialog.getExistingDirectory(None,"选择XSeg模型文件夹");
        if(len(folder)>2):
            self.ui.LineEditXSegModelFolder.setText(folder);

    #---------设置预览人脸文件夹----------
    def btnBrowsePreviewFolderClick(self):
        folder=QFileDialog.getExistingDirectory(None,"选择人脸文件夹",self.ui.LineEditPreviewFolder.text());
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
        page_size=self.img_num_cols*self.img_num_rows;
        start_idx=(n//page_size)*page_size;
        self.ui.spinBoxPreviewStartNum.setValue(start_idx)
        self.ShowAiMarkPreviewList(0,False);

    def btnPrePageFaceClick(self):
        start_idx=self.ui.spinBoxPreviewStartNum.value(); 
        page_size=self.img_num_cols*self.img_num_rows;
        if(start_idx>page_size):
            start_idx=start_idx-page_size;
            self.ui.spinBoxPreviewStartNum.setValue(start_idx);
            self.ShowAiMarkPreviewList();

    def btnNextPageFaceClick(self):
        start_idx=self.ui.spinBoxPreviewStartNum.value(); 
        img_count=self.ui.spinBoxPreviewTotalNum.value(); 
        page_size=self.img_num_cols*self.img_num_rows;
        if(start_idx<img_count-page_size):
            start_idx=start_idx+page_size;
            self.ui.spinBoxPreviewStartNum.setValue(start_idx); 
            self.ShowAiMarkPreviewList();

    
     


    def btnStartSuperResolutionClick(self): 
        if sd.AppEnable is False:
            QMessageBox.warning(None,"提示","Lab使用功能过期,请进入启动器激活Lab使用权限");
            return;

        folder=self.ui.LineEditAdjustFolder.text();
        engine=self.ui.comboBoxSuperEngine.currentText()
        if(os.path.exists(folder)==False):
            ToolKit.ShowWarningError(None,"错误","图片文件夹未设置");
            return;
        move_folder=f"{folder}/HD"
        if os.path.exists(move_folder) ==False:
            os.mkdir(move_folder)

        th=threading.Thread(target=self.SuperResolutionThreadRun,args=(folder,move_folder,engine,False))
        th.daemo=True;
        th.start();


   
    def SuperResolutionThreadRun(self,folder,move_folder,engine,only_blur=False,blur_threshold=15):
        from facelib.SuperReso import gpen
        from facelib.SuperReso import CodeFormerRun
        import shutil
        files=[file for file in os.listdir(folder) if (".jpg" in file or ".png" in file) ]
        #files.sort()
        backup_folder=f"{folder}/Backup"
        if os.path.exists(backup_folder) ==False:
            os.mkdir(backup_folder)
        #print("总共需要处理:",len(files))
        for idx,file in enumerate(files):
            file_full_path=f"{folder}/{file}"      
            save_full_path=f"{move_folder}/{file}"
            backup_full_path=f"{backup_folder}/{file}"
            #old_image=cv2ex.cv2_imread(file_full_path)
            dfl_info_old=DFLJPG.load(file_full_path)
            old_image=dfl_info_old.get_img()

            if only_blur:
                score = round(cv2.Laplacian(old_image, cv2.CV_64F).var(),3)
                if(score>blur_threshold):
                    continue;
            #---convert
            time_start=time.time()
            if "GPEN" in engine :
                out_same_size=gpen.process(old_image,out_dtype=np.uint8)
            else:
                out_same_size=CodeFormerRun.process(old_image,out_dtype=np.uint8)
            time_end=time.time()
            time_use=round(time_end-time_start,3)*1000
            print(f"[P]清晰化处理图片{file},[{time_use}ms],进度[{idx}/{len(files)}]",end='\r')
            cv2ex.cv2_imwrite(save_full_path,out_same_size)
            dfl_info_new=DFLJPG.load(save_full_path)
            dfl_info_new.dfl_dict=dfl_info_old.dfl_dict;
            dfl_info_new.save()
            print(f"\r处理图片{file}结束",end='\r')
            shutil.move(file_full_path, backup_full_path)
            #---- 显示预览
            cv2.putText(old_image,f'{file}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),1)
            compare=np.hstack((old_image,out_same_size))
            cv2.imshow("compare",compare)
            cv2.waitKey(20)
        print("清晰化处理处理结束")

    BlurThreadStatus="run";
    def btnStartMoveBlurClick(self):
        if '开始' in self.ui.btnStartMoveBlur.text():
            folder=self.ui.LineEditAdjustFolder.text();
            blur_threshhold=self.ui.spinAdjustBlurThreshold.value();
            if(os.path.exists(folder)==False):
                ToolKit.ShowWarningError(None,"错误","图片文件夹未设置");
                return;
            th=threading.Thread(target=self.MoveBlurImageThreadRun,args=(folder,blur_threshhold))
            th.daemo=True;
            th.start();
            self.ui.btnStartMoveBlur.setText("停止处理")
            self.BlurThreadStatus="run"
        if '停止' in self.ui.btnStartMoveBlur.text():
            self.ui.btnStartMoveBlur.setText("开始处理")
            self.BlurThreadStatus="stop"

    def MoveBlurImageThreadRun(self,folder,blur_threshhold):
        files=[file for file in os.listdir(folder) if (".jpg" in file or ".png" in file) ]
        idx=0; total=len(files);
        print(f"[P]开始模糊判断{total}",end="\r")
        blur_folder=folder+"/blur"
        if os.path.exists(blur_folder) is False:
            os.mkdir(blur_folder)
        cv2.namedWindow("blur")
        for file in files:
            if  self.BlurThreadStatus=="stop":
                print("[P]处理被用户停止，退出处理",end='\r')
                return
            file_full_path=f"{folder}/{file}"
            move_full_path=f"{blur_folder}/{file}"
            dflinfo=DFLJPG.load(file_full_path)
            image=dflinfo.get_img()
            mask=dflinfo.get_xseg_mask()
            if mask is not None:
                mh,mw,mc=mask.shape
                image=cv2.resize(image,(mw,mh))
                image=image*mask
                image = np.clip( image, 0, 255 ).astype(np.uint8)
            #image=cv2ex.cv2_imread(file_full_path) 
            idx=idx+1;
            score = round(cv2.Laplacian(image, cv2.CV_64F).var(),3)
            print(f"[P]判断图片{file},[{idx}/{total}]",end='\r')
            cv2.putText(image, f"value:{str(score)}", (35,30 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1) 
            if score<blur_threshhold:
                shutil.move(file_full_path,move_full_path)
                cv2.imshow("BlurImage",image)
                cv2.waitKey(50)


    

    def OnPreFaceImageClick(self):
        pass

    def OnNextFaceImageClick(self):
        pass

    def OnPreOccludeClick(self):
        pass

    def OnNextFaceImageClick(self):
        pass


    #---------选择要添加遮挡的人脸
    def btnBrowseSrcOccludeFaceClick(self):
       file,ok=QFileDialog.getOpenFileName(None,"选择人脸文件",self.ui.LineEditAdjustFolder.text(),'image files (*.jpg)');
       if ok: 
            dflimg=DFLJPG.load(file)
            if dflimg is None:
                ToolKit.ShowWarningError(None,"错误","该图片不含人脸信息，不能添加遮挡")
                return

            if dflimg.has_xseg_mask() is False:
                ToolKit.ShowWarningError(None,"错误","该图片不含有遮罩信息，请应用遮罩.只有包含Xseg遮罩信息的人脸图片才可以进行遮挡物添加")
                return
            img=dflimg.get_img();
            ToolKit.ShowImageInLabel(img,self.ui.Adjust_Src_Face_Label)
            self.ui.LineEditOccludeSrcFace.setText(file);

    #---选择遮挡物图片
    def btnBrowseOccludeFileClick(self):
       dir=os.path.abspath("../Occlusion")
       file,ok=QFileDialog.getOpenFileName(None,"选择遮挡物图片",dir,'image files (*.jpg)');
       if ok: 
            dflimg=DFLJPG.load(file)
            if dflimg.has_xseg_mask() is False:
                ToolKit.ShowWarningError(None,"错误","该图片不含有遮罩信息，请使用工具中添加遮挡素材、然后编辑遮挡物轮廓")
                return
            img=dflimg.get_img();
            ToolKit.ShowImageInLabel(img,self.ui.Adjust_Occlusion_Label)
            self.ui.LineEditOccludeFile.setText(file);

    #------打开遮挡素材文件夹
    def btnOpenOccludeFolderClick(self):
        folder=os.path.abspath("../Occlusion")
        try:           
            os.startfile(folder)
        except:
            ToolKit.ShowWarningError(None,"错误","打开遮挡素材文件夹失败,请检查文件夹是否存在");

    #--生成单张遮挡预览图
    def btnPreviewSaveOccludeClick(self):
        face_file=self.ui.LineEditOccludeSrcFace.text();
        occlude_file=self.ui.LineEditOccludeFile.text();
        if os.path.exists(face_file) is False:
            ToolKit.ShowWarningError(None,"提示","原脸图片加载错误，请重新选择设置");
            return None
        if os.path.exists(occlude_file) is False:
            ToolKit.ShowWarningError(None,"提示","遮挡物素材错误，请重新选择设置");
            return None

        scale=self.ui.spinScale.value()
        rotAngle=self.ui.spinRotAngle.value()
        transX=self.ui.spinTransX.value()
        transY=self.ui.spinTransY.value()
        dfl_merge,merge_img,merge_mask_preview=RandomOcclude.getMergeDFL(face_file,occlude_file,scale,rotAngle,transX,transY)
        self.merge_dfl=dfl_merge;
        ToolKit.ShowImageInLabel(merge_mask_preview,self.ui.Adjust_occlude_merge)
        #cv2.imshow("mask",dfl_merge.get_xseg_mask())
        
        return dfl_merge,merge_img
       

    #--保存生成单张遮挡预览图
    def btnSaveOccludeMergeResult(self):        
        dfl_merge,merge_img=self.btnPreviewSaveOccludeClick()
        if dfl_merge is None:
            return
        save_file_path,type=QFileDialog.getSaveFileName(None,"保存遮挡生成人脸","",'image files (*.jpg)')
        if len(save_file_path)<5:
            return
       
        try:
            cv2ex.cv2_imwrite(save_file_path,merge_img)
            dflnew=DFLJPG.load(save_file_path)
            dflnew.set_landmarks(dfl_merge.get_landmarks())
            dflnew.set_xseg_mask(dfl_merge.get_xseg_mask())
            dflnew.save()
            pass
        except :
            ToolKit.ShowWarningError(None,"提示","保存遮挡生成人脸-发生错误");

        #---------设置遮挡保存文件夹----------
    def btnBrowseOccludeSaveFolderClick(self):
        folder=QFileDialog.getExistingDirectory(None,"选择遮挡保存文件夹", os.path.dirname(self.ui.LineEditOccludeSaveFolder.text()));
        if(len(folder)>2):
            self.ui.LineEditOccludeSaveFolder.setText(folder);
    #------打开遮挡保存文件夹
    def btnOpenOccludeSaveFolderClick(self):
        folder=self.ui.LineEditOccludeSaveFolder.text();  
        try:           
            os.startfile(folder)
        except:
            ToolKit.ShowWarningError(None,"错误","打开遮挡素材文件夹失败,请检查文件夹是否存在");
            
    def StopAutoOccludeFacesMergeClick(self): 
        if hasattr(self,"AutoMergeThreadRun"):
            self.AutoMergeThreadRun=False;
    #-----------------------------------
    #---开始随机生成遮挡图（新版）--------
    def btnStartAutoOccludeFacesClick(self):
        if sd.AppEnable is False:
            QMessageBox.warning(None,"提示","随机遮挡生成功能未开启,请进入启动器激活Lab使用权限");
            return;
        #-检查保存目录是否存在
        save_folder=self.ui.LineEditOccludeSaveFolder.text()
        if  os.path.exists(save_folder)==False:
            ToolKit.ShowWarningError(None,"错误","保存文件夹不存在，请检查保存路径");
            return
        #-检查人脸和遮挡物是否设置
        face_file=self.ui.LineEditOccludeSrcFace.text();
        occlude_file=self.ui.LineEditOccludeFile.text();
        if os.path.exists(face_file) is False:
            ToolKit.ShowWarningError(None,"提示","原脸图片没有设置");
            return
        if os.path.exists(face_file) is False:
            ToolKit.ShowWarningError(None,"提示","遮挡物素材没有设置");
            return

        #-获取要生成的数量
        scale_count=self.ui.spinScaleCount.value()
        trans_x_count=self.ui.spinTransXCount.value()
        trans_y_count=self.ui.spinTransYCount.value()
        rot_count=self.ui.spinRotCount.value()

        scale_list=np.linspace(self.ui.spinMinScale.value(),self.ui.spinMinScale.value(),int(scale_count))
        rot_list=np.linspace(self.ui.spinMinRot.value(),self.ui.spinMaxRot.value(),int(rot_count))
        trans_x_list=np.linspace(self.ui.spinMinTransX.value(),self.ui.spinMaxTransX.value(),int(trans_x_count))
        trans_y_list=np.linspace(self.ui.spinMinTransY.value(),self.ui.spinMaxTransY.value(),int(trans_y_count))
            

        self.AutoMergeThreadRun=True;
        th=threading.Thread(target=self.AutoOccludeMergeThreadRun,args=(face_file,occlude_file,save_folder,scale_list,rot_list,trans_x_list,trans_y_list))
        th.daemo=True;
        th.start(); 


    def AutoOccludeMergeThreadRun(self,face_file,occlude_file,save_dir,scale_list,rot_list,trans_x_list,trans_y_list):
        face_raw_short_name=os.path.basename(face_file).split(".")[0]
        occlude_short_name=os.path.basename(occlude_file).split(".")[0]
        for scale in scale_list:
            for rot in rot_list:
                for trans_x in trans_x_list:
                    for trans_y in trans_y_list:
                        scale=round(scale,2)
                        rot=round(rot)
                        trans_x=round(trans_x)
                        trans_y=round(trans_y)
                        save_file_path=f"{save_dir}/{face_raw_short_name}-{occlude_short_name}_s{scale}_r{rot}_tx{trans_x}_ty{trans_y}.jpg"
                        print(save_file_path)
                        try:
                            pass
                        except :
                            print("发生错误")
                        dfl_merge,merge_img,merge_mask_preview=RandomOcclude.getMergeDFL(face_file,occlude_file,scale,rot,trans_x,trans_y)
                        cv2ex.cv2_imwrite(save_file_path,merge_img)
                        dflnew=DFLJPG.load(save_file_path)
                        dflnew.set_landmarks(dfl_merge.get_landmarks())
                        dflnew.set_xseg_mask(dfl_merge.get_xseg_mask())
                        dflnew.save()
                        cv2.imshow("merge_occlude",merge_img)
                        cv2.waitKey(10)
                        if self.AutoMergeThreadRun is False:
                            print("线程被终止、结束生成");
                            return
        print("遮挡生成结束")            


    def btnAddOccludeSampleClick(self):
        save_to_dir=os.path.abspath("../Occlusion")
        file,ok=QFileDialog.getOpenFileName(None,"选择遮挡物图片","",'image files (*.jpg)');
        if ok: 
            img=cv2ex.cv2_imread(file)
            h,w,c=img.shape
            min_size=min(h,w)
            img_rect=img[0:min_size,0:min_size,:]
            occlude_img=cv2.resize(img_rect,(512,512));
            
            raw_short_name=os.path.basename(file).split(".")[0]
            save_to_path=f"{save_to_dir}/{raw_short_name}.jpg"
            if os.path.exists(save_to_path):
                save_to_path=f"{save_to_dir}/{raw_short_name}_new.jpg"

            cv2ex.cv2_imwrite(save_to_path,occlude_img)
            dflimg=DFLJPG.load(save_to_path)
            dflimg.set_landmarks([[0,0],[12,4]]);
            dflimg.save()
            ToolKit.ShowWarningError(None,"提示","已经将图片处理添加到素材目录，请用遮罩编辑来绘制遮挡物轮廓")