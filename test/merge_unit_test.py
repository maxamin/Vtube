#import tensorflow as tf

import os,numpy as np,time
import cv2
import numexpr as ne
from core.xlib.image import ImageProcessor

height=720;width=1280;
mask_size=256;face_size=256

start=time.time()
origin_frame_image=np.random.randn(height,width,3)
face_swapped_to_frame_img=np.ones(shape=(height,width,3),dtype=float)
face_mask_to_frame=np.random.randn(height,width,3)
face_mask=np.random.randn(256,256,3)
time_use=time.time()-start;
print("create numpy array time use:",time_use*1000)

start=time.time()
#fim=ImageProcessor(face_mask).ch(3).to_uint8().get_image('HWC') 
face_mask_rdn=np.random.randn(256,256,3)
face_mask = np.ones(shape=(face_size, face_size,3), dtype=np.float32)
time_use=time.time()-start;
print("create imageprocessor time use:",time_use*1000)

start=time.time()
face_mask = ImageProcessor(face_mask).erode_blur(0, 15, fade_to_border=True).get_image('HWC')
face_mask = ImageProcessor(face_mask).to_ufloat32().get_image('HWC')
time_use=time.time()-start;
print("erode_blur time use:",time_use*1000)


start=time.time()
out_merged_frame=origin_frame_image*(1.0-face_mask_to_frame)+face_swapped_to_frame_img*face_mask_to_frame
time_use=time.time()-start;
print("numpy time use:",time_use*1000)

start=time.time()
out_merged_frame = ne.evaluate('origin_frame_image*(1.0-face_mask_to_frame) + face_swapped_to_frame_img*face_mask_to_frame')
time_use=time.time()-start;
print("numexpr time use:",time_use*1000)

#a=(1.2,1.3,1.4,1.5)
#b=[1.2,1.3,1.4,1.5]
#l,t,t,b=int(a)
#print(l,t,t,b)

#resolution="600*400px"
#reso=resolution.replace("px","").split("*");
#print(reso)
#print("width:",int(reso[0]),type(int(reso[0])))

#cv2.VideoCapture vedio;
#vedio.open(0);
#if（vedio.isOpened())
#{
#vedio.set(CAP_PROP_FRAME_WIDTH, 800);
#vedio.set(CAP_PROP_FRAME_HEIGHT, 600);
#}


#folder=r"F:\FaceImageSet\元成\aligned"
#folder=r"F:\FaceImageSet\FFHQ_ASIAN\Man5K\Align_wf"
#files=[file for file in os.listdir(folder) if ".jpg" in file]
#files=files[0:-1:6]
#print(len(files))
#for file in files:
#    fullpath=f"{folder}/{file}"
#    img=cv2ex.cv2_imread(fullpath)
#    #img=img.as_type(np.float32)
#    #g_img=cv2.Sobel(img,-1,1,0)
#    g_img=cv2.Laplacian(img,-1,ksize=3)

#    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#    gray_img=np.expand_dims(gray_img,2)
#    gray_img=np.repeat(gray_img,3,2)

#    out_img=np.hstack((img,g_img))
#    cv2.imshow("img",img)
#    cv2.imshow("g_img",g_img)
#    cv2.imshow("gray_img",gray_img)
#    cv2.waitKey(1000)
from core import cv2ex
from core.DFLIMG import DFLJPG
def ContourizeThreadRun(folder):
    files=[file for file in os.listdir(folder) if ".jpg" in file]
    files=files[0:30]
    move_folder=f"{folder}_contourize"
    if os.path.exists(move_folder)==False:
        os.mkdir(move_folder)
    for idx,file in enumerate(files):
        
        file_full_path=f"{folder}/{file}"      
        save_full_path=f"{move_folder}/{file}"
        dfl_info_old=DFLJPG.load(file_full_path)
        old_image=dfl_info_old.get_img()

        #out=cv2.cvtColor(old_image,cv2.COLOR_BGR2GRAY)
        #out=cv2.Sobel(old_image,-1,1,0)
        out=cv2.Laplacian(old_image,cv2.CV_16S,ksize=3,scale=3,delta=10)
        #---convert
        print(f"正在处理图片{file}..............[{idx}/{len(files)}]",out.shape,end='\r')
        cv2.imshow("out",out)
        cv2.waitKey(500)
        
        cv2ex.cv2_imwrite(save_full_path,out)
        dfl_info_new=DFLJPG.load(save_full_path)
        dfl_info_new.dfl_dict=dfl_info_old.dfl_dict;
        dfl_info_new.save()
        #dfl_info.save()
    print("\n处理结束")


def PickSubImages(folder,move_folder,interval=3):
    import shutil
    files=[file for file in os.listdir(folder) if ".jpg" in file]
    files=files[0:-1:interval]
    if os.path.exists(move_folder) is False:
        os.mkdir(move_folder)
    for idx,file in enumerate(files):
        file_full_path=f"{folder}/{file}"      
        save_full_path=f"{move_folder}/{file}"
        print(f"save to {save_full_path}.............[{idx}/{len(files)}]",end='\r')
        shutil.copyfile(file_full_path,save_full_path)
#PickSubImages("F:\FaceImageSet\景甜","F:\FaceImageSet\景甜2600",5)
#ContourizeThreadRun(r"F:\FaceImageSet\FFHQ_ASIAN\Man5K\Align_wf")



#train_samples=[['a1','a2','a3','a4'],['b1','b2','b3','b4']]
#test_sample=['t1','t2','t3','t4']
#preview_sample=[]
#preview_sample.append(train_samples[0])
#preview_sample.append(test_sample)
#print(preview_sample)

#( (warped_src, target_src, target_srcm, target_srcm_em),
#          (warped_dst, target_dst, target_dstm, target_dstm_em) ) = preview_sample
#print(warped_src)


#class a:
#    def __init__(self):
#        self.v1='v1'
#    def test(self,str):
#        self.v1=str
#    def vset(self,str):
#        a.v2=str
#    def print_v2(self):
#        if a.v2 is None:
#            print('v2 is none')
#        else:
#            print(a.v2)

#aa=a()
#print(aa.v1)
#aa.vset('ppp')
#print(aa.v2)
#time_tick=time.time()
#a=np.random.randn(256,256,3)
#b=np.random.randn(256,256,3)
#c=a*b
#print(time.time()-time_tick)
#print("---------------")
#print(b)
#print("---------------")
#print(a*b)

#print("adsad",end='\r');
#time.sleep(0.5)
#print("new line")
#for i in range(20):
#    print("progress start ",i,end='\r')
#    time.sleep(0.4)
#    print("progress end",i,end='\r')


#from PyQt5.QtWidgets import QFileDialog,QMessageBox,QInputDialog,QDialog,QSpinBox,QApplication;
#import PyQt5.QtCore as QtCore
#app=QtCore.QCoreApplication.instance()
#if app is None:
#    app = QApplication([])
#str=QInputDialog.getText(None,"name","input name");
#print(str)


#a=110;
#def test():
#    global a;a=100;
#    print(a)
#test()
#print(a)
#folder=r"F:\DeepModels\王凯-224WF-80W"
#new_model_name="王凯"

#files_list=[file for file in os.listdir(folder) if os.path.isfile(os.path.join(folder,file)) ]
#files_list.sort()
#for file in files_list:
#    #print(file)
#    suffix=file[file.find('_'):]
#    old_full_name=f"{folder}/{file}"
#    new_full_file_name=f"{folder}/{new_model_name}{suffix}"
#    if os.path.exists(new_full_file_name):
#        print("文件名冲突：",old_full_name,new_full_file_name)
#        continue
#    print(new_full_file_name)
#    os.rename(old_full_name,new_full_file_name)


#a = tf.constant([[1.6,2.6,1.6],[2.5,3.6,4.6]])
#print(a.get_shape(),isinstance(a,tf.Tensor))
#print(tf.reduce_mean(a,1)) 


#a=102;
#print(' a is 10') if a==10 else print('no')
#b='is 10' if a==10 else 'not 10'
#print(b)



#def func(a,b,**kvargs):
#    for arg in kvargs:
#        print(arg)
#    d=0
#    if "d" in kvargs:   d=kvargs.get("d")
#    if "e" in kvargs: print("hav e")
#    print(d)
#func(a=11,b=12,c=13,e=44)

#a=[1,3,4,5,9]
#del_list=[0,1,3]
#del_list.reverse()
#print(del_list)
#for i in del_list:
#    a.pop(i)

#print(a)