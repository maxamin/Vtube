from kit.FaceDeformLtw import FaceDeformLtw as deform
import cv2,numpy as np,time
import kit.ToolKit as ToolKit
from core import cv2ex
import kit.ShareData as sd
import set_env
from core.DFLIMG.DFLJPG import DFLJPG

if __name__=="__main__":  

    #frame_img=cv2ex.cv2_imread(r"F:\AvVideo\真人原始图片\低颜值生tu\凤姐.jpg")
    #set_env.set_env()
    #raw_images,aligned_images,landmarks_list,angles_list=ToolKit.GetDetectAndAlignFace(frame_img)
    #if len(raw_images)==0:
    #    print("没有检测到人脸")
    #cv2.imshow("img",frame_img)

    #--- 读取参考人脸轮廓
    #ref_defl=DFLJPG.load(r"F:\TrainFaceLib\开发\HD\c10_0.jpg")
    #ref_face=ref_defl.get_img()
    #ref_landmarks=ref_defl.get_landmarks();

    #---读取要处理的原人脸
    dfl=DFLJPG.load(r"F:\TrainFaceLib\开发\男\chuanyu.jpg")
    origin_face=dfl.get_img()
    origin_landmarks=dfl.get_landmarks();
    for i in range(16):
        pt=origin_landmarks[i].astype(int)
        #cv2.circle(origin_face,pt,2,(0,255,0),thickness=-1)

    #---变形参考人脸轮廓
    ref_landmarks=origin_landmarks.copy()
    ref_face=origin_face.copy()
    ref_landmarks[0:7,0]=ref_landmarks[0:7,0]*1.2
    ref_landmarks[9:16,0]=ref_landmarks[9:16,0]*0.95
    for i in range(16):
        pt=ref_landmarks[i].astype(int)
        cv2.circle(ref_face,pt,2,(i*10,255,i*10),thickness=-1)
   
    #--- 变形处理原人脸
    thin_face=deform.face_deform_thin(origin_face,origin_landmarks,ref_landmarks)
    time_start=time.time()
    #thin_face=deform.thin_chin_cheek(origin_face,origin_landmarks,True,1.0,1.0,True,1.0,1.0)
    this_face=deform.face_deform_thin(origin_face,origin_landmarks,ref_landmarks)
    time_end=time.time()
    time_use=time_end-time_start
    print("瘦脸耗时：",time_use)

    for i in range(16):
        pt_o=origin_landmarks[i].astype(int)
        pt_f=ref_landmarks[i].astype(int)
        cv2.circle(thin_face,pt_o,2,(i*10,0,255),thickness=-1)
        cv2.circle(thin_face,pt_f,2,(i*10,255,0),thickness=-1)

    stack_preview=np.hstack([origin_face,ref_face,thin_face])
    cv2.imshow("face",stack_preview)
    cv2.waitKey(0)