from pathlib import Path
from core.DFLIMG.DFLJPG import DFLJPG
import cv2,numpy as np,os,time,random
from kit.FaceEngines import FaceEngines
import kit.ToolKit as ToolKit
from core import pathex,cv2ex

def getMergeDFL(originFaceImage:str,occudeImage:str,scale=1.0,rotAngle=0,transX=0,transY=0):
    #---获得人脸图
    dfl_origin=DFLJPG.load(originFaceImage)
    origin_img=dfl_origin.get_img().astype(float)/255
    face_h,face_w,oc=origin_img.shape
    origin_mask=dfl_origin.get_xseg_mask()
    origin_mask=cv2.resize(origin_mask,(face_w,face_h))
    origin_mask=np.expand_dims(origin_mask,2).repeat(3,axis=2)
    if origin_mask is None:
        print(f"[P]原图{originFaceImage}没有遮罩，停止生成随即遮挡")
        return None,None

    #---获得遮挡物图片
    dfl_occlude=DFLJPG.load(occudeImage)
    occlude_img=dfl_occlude.get_img().astype(float)/255 
    occlude_mask=dfl_occlude.get_xseg_mask()
    
    #--旋转缩放
    occlude_img=cv2.resize(occlude_img,(face_w,face_h))
    occlude_mask=cv2.resize(occlude_mask,(face_w,face_h))
    M_rot_scale = cv2.getRotationMatrix2D(center=(face_h/2,face_h/2),angle=rotAngle,scale=scale)
    occlude_img = cv2.warpAffine(occlude_img,M_rot_scale,(face_w,face_h))
    occlude_mask = cv2.warpAffine(occlude_mask,M_rot_scale,(face_w,face_h))
    #---平移
    M_trans = np.float32([[1, 0, transX], [0, 1, transY]])
    occlude_img = cv2.warpAffine(occlude_img,M_trans,(face_w,face_h))
    occlude_mask = cv2.warpAffine(occlude_mask,M_trans,(face_w,face_h))

    occlude_mask=np.expand_dims(occlude_mask,2).astype(np.float32)

    merge_img=occlude_mask*occlude_img+(1-occlude_mask)*origin_img;
    anti_occlude_mask = 1-occlude_mask
    merge_mask=origin_mask*anti_occlude_mask
    

    merge_mask_preview=merge_img*(1-merge_mask)*0.2+merge_img*merge_mask
    merge_mask_preview=(merge_mask_preview*255.0).astype(np.uint8)
     
    merge_img=(merge_img*255.0).astype(np.uint8)
    dfl_origin.img=merge_img
    dfl_origin.set_xseg_mask(merge_mask) 
    return dfl_origin,merge_img,merge_mask_preview
     
def getRandomOccludeFace(originFaceImage:str,occudeImage:str,SavePath=None,minScale=0.4,maxScale=1.3):
    #---获得人脸图
    dfl_origin=DFLJPG.load(originFaceImage)
    origin_img=dfl_origin.get_img().astype(float)/255
    face_h,face_w,oc=origin_img.shape
    origin_mask=dfl_origin.get_xseg_mask()
    origin_mask=cv2.resize(origin_mask,(face_w,face_h))
    origin_mask=np.expand_dims(origin_mask,2).repeat(3,axis=2)
    if origin_mask is None:
        print(f"[P]原图{originFaceImage}没有遮罩，停止生成随即遮挡")
        return
    #---获得遮挡物图片
    dfl_occlude=DFLJPG.load(occudeImage)
    occlude_img=dfl_occlude.get_img().astype(float)/255
    occlude_h,occlude_w,occlude_c=occlude_img.shape
    seg_ie_polys=dfl_occlude.get_seg_ie_polys()
    occlude_mask=np.zeros((face_h,face_w,3)).astype(float)
    if seg_ie_polys is not None:
        if seg_ie_polys.has_polys():
            seg_ie_polys.overlay_mask(occlude_mask)
    


    #--遮挡物随机旋转-缩放
    random_rotate_angle=random.random()*360
    random_scale=random.random()*(maxScale-minScale)+minScale;
    M = cv2.getRotationMatrix2D(center=(face_h/2,face_h/2),angle=random_rotate_angle,scale=random_scale)
    occlude_img = cv2.warpAffine(occlude_img,M,(face_w,face_h))
    occlude_mask = cv2.warpAffine(occlude_mask,M,(face_w,face_h))

        #--缩放成相同大小
    #if face_h!=occlude_h:
    #    occlude_img=cv2.resize(occlude_img,(face_w,face_h))
    #    occlude_mask=cv2.resize(occlude_mask,(face_w,face_h))

    #---融合人脸和遮挡物
    merge_img=occlude_mask*occlude_img+(1-occlude_mask)*origin_img;
    anti_occlude_mask = 1-occlude_mask
    merge_mask=origin_mask*anti_occlude_mask
    #merge_mask=np.expand_dims(merge_mask,2).repeat(3,axis=2)

    preview_mask=merge_mask.copy()
    preview_mask[:,:,1]=0
    merge_masked=merge_img.copy()
    merge_masked[...] = ( merge_masked * (1-preview_mask) + merge_masked * preview_mask / 2 )[...]
    #----合成一张图片展示预览
    stack_img_1=np.hstack([origin_img,occlude_img,occlude_mask])
    stack_img_2=np.hstack([merge_img,merge_masked,merge_mask])
    stack_img=np.vstack([stack_img_1,stack_img_2])
    merge_h,merge_w,mc=stack_img.shape
    if merge_h>600:
        ratio=600/merge_h
        stack_img=cv2.resize(stack_img,(0,0),fx=ratio,fy=ratio)
    cv2.imshow("random_occlude_img_preview",stack_img)

    #---保存融合后的图
    if SavePath is not None:
        merge_img=merge_img*255
        cv2ex.cv2_imwrite(SavePath,merge_img)
        save_dfl=DFLJPG.load(SavePath)
        save_dfl.set_landmarks(dfl_origin.get_landmarks())
        save_dfl.set_xseg_mask(merge_mask)
        save_dfl.save()

def GenerateRandomOccludeFaces(face_images_folder,occlude_folder,save_folder,generate_total_count,sleep_time=600):
    if os.path.exists(face_images_folder)==False or os.path.exists(occlude_folder)==False or os.path.exists(save_folder)==False:
        print("发生错误，文件夹不存在")
        return
    cv2.destroyAllWindows()
    face_images=pathex.get_image_paths(face_images_folder,[".jpg"],False)
    occlude_imgs=pathex.get_image_paths(occlude_folder,[".jpg"],False)
    if len(face_images)==0:
        print("发生错误：人脸文件夹中没有人脸文件")
        return
    if len(occlude_imgs)==0:
        print("发生错误：遮挡素材文件夹中没有文件")
        return
    file_prefix=time.strftime("%H%M%S",time.localtime())
    for i in range(generate_total_count):
        face_idx=random.randint(0,len(face_images)-1)
        occlude_idx=random.randint(0,len(occlude_imgs)-1)
        originFace=face_images[face_idx]
        occlude_img=occlude_imgs[occlude_idx]        
        save_path=f"{save_folder}/{file_prefix}_{i}.jpg"
        print(f"[P]生成随即遮挡进度({i}/{generate_total_count}),保存到{save_path}")
        getRandomOccludeFace(originFace,occudeImage=occlude_img,SavePath=save_path)
        cv2.waitKey(sleep_time)
    print(f"[P]随机遮挡生成结束")

if __name__=="__main__":     
    generate_total_count=100
    face_images_folder="F:\TrainFaceLib\dy王66\HD";
    occlude_folder="F:\TrainFaceLib\遮挡物\遮罩后"
    save_dir="F:\TrainFaceLib\遮挡物\生成"
    seep_time=100
    GenerateRandomOccludeFaces(face_images_folder,occlude_folder,save_dir,generate_total_count,seep_time)
    cv2.waitKey(0)
