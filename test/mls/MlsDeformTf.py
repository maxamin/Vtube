import numpy as np,time,random,cv2
import kit.ShareData as sd
import set_env
set_env.set_env()
import tensorflow as tf
from core.DFLIMG.DFLJPG import DFLJPG

class MlsDeformTf():
    v=None
    @staticmethod
    def create_img_coordinate(img): 
        time_start=time.time()
        width, height = img.shape[:2]
        pcth = np.repeat(np.arange(height).reshape(height, 1), [width], axis=1)
        pctw = np.repeat(np.arange(width).reshape(width, 1), [height], axis=1).T
        img_coordinate = np.swapaxes(np.array([pcth,pctw]), 1, 2).T
        MlsDeformTf.v= tf.convert_to_tensor(img_coordinate,dtype=tf.float32)
        time_end=time.time()
        print(f"[P]创建坐标变量,耗时{time_end-time_start},({img_coordinate.dtype}),{img_coordinate.shape}")

    @staticmethod
    def assert_vars(img):
        if MlsDeformTf.v is None:
            MlsDeformTf.create_img_coordinate(img)
        else:
            h,w=img.shape[0:2]
            cw,ch=MlsDeformTf.v.shape[0:2]
            if h!=ch or w!=cw:
                MlsDeformTf.create_img_coordinate(img)

    @staticmethod
    def deformation_tf(img,pi,qi,dbg=False):
        MlsDeformTf.assert_vars(img);
        time_start=time.time()
        width,height=img.shape[0:2]
        n=pi.shape[0]
        v=MlsDeformTf.v
        # height*width*N
        with tf.device("/gpu:0"):
            pi=tf.convert_to_tensor(pi,dtype=tf.float32)
            qi=tf.convert_to_tensor(qi,dtype=tf.float32)
            print("[P]--------------------------")

            #--- 计算wi--[H*W*N]
            v_reshape=tf.reshape(v,(height, width, 1, 2));
            pi_reshape=tf.reshape(pi,(1, 1, -1, 2));
            pv=pi_reshape-v_reshape 
            t=tf.reduce_sum(pv*pv,axis=3)+0.000000001
            wi=1/t 
            if dbg: print(f"[P]wi计算结束{wi.shape}，耗时{round(time.time()-time_start,4)}秒")
            #--- 计算 p*--[h*w*2] q*[h*w*2]
            sigma_wipi=tf.matmul(wi,pi)  #[h*w*n]*[n*2]=[h*w*2]
            sigma_wi=tf.reduce_sum(wi,axis=2)  #[h*w*1]
            sigma_wi=tf.expand_dims(sigma_wi,axis=2)
            pstar=tf.divide(sigma_wipi,sigma_wi) 
            if dbg: print(f"[P]pstar计算结束{pstar.shape}，耗时{round(time.time()-time_start,4)}秒")

            sigma_wiqi=tf.matmul(wi,qi)
            qstar=tf.divide(sigma_wiqi,sigma_wi)
            if dbg: print(f"[P]qstar计算结束{qstar.shape}，耗时{round(time.time()-time_start,4)}秒")
        
            #--- 计算 p^--[h*w*n*2] q^[h*w*n*2]
            phat=tf.subtract(tf.reshape(pi,(1,1,-1,2)),tf.reshape(pstar,(height,width,1,2)))
            qhat=tf.subtract(tf.reshape(qi,(1,1,-1,2)),tf.reshape(qstar,(height,width,1,2)))
            if dbg: print(f"[P]p^和q^计算结束{phat.shape}，耗时{round(time.time()-time_start,4)}秒")

            #--- 计算 s2--[h*w*n*1*2]  
            reshape_phat=tf.reshape(phat,(height,width,n,1,2))
            cat_phat=reshape_phat[:,:,:,:,1],-reshape_phat[:,:,:,:,0]
            s2=tf.concat(cat_phat,axis=3)
            s2=tf.reshape(s2,(height,width,n,1,2))
            if dbg: print(f"[P]s2计算结束{s2.shape}，耗时{round(time.time()-time_start,4)}秒")
        
            #print("v shape:",v.shape,"pstar.shape:",pstar.shape)
            v_pstar=tf.subtract(v,pstar)
            t=v_pstar[:,:,1], -v_pstar[:,:,0]
            t=tf.transpose(t,(1,2,0))
            t=tf.reshape(t,(height,width,1,2,1))
            z2=tf.repeat(t,n,axis=2)
            if dbg: print(f"[P]z2计算结束{z2.shape}，耗时{round(time.time()-time_start,4)}秒")

            #--v_pstar [h*w*n*2*1]
            v_pstar = tf.repeat(tf.reshape(v_pstar,(height,width,1,2,1)), [n], axis=2)
            a = tf.matmul(reshape_phat, v_pstar) # height*width*n*1*1
            b = tf.matmul(reshape_phat, z2)
            c = tf.matmul(s2, v_pstar)
            d = tf.matmul(s2, z2)
            if dbg: print(f"[P]a,b,c,d计算结束{d.shape}，耗时{round(time.time()-time_start,4)}秒")

            #-- A-[h*w*n*2*2]
            reshape_wi = tf.repeat(tf.reshape(wi,(height,width,n,1)),[4],axis=3) 
            sz=tf.concat((a,b,c,d), axis=3)
            sz=tf.reshape(sz,(height,width,n,4))
            A=tf.reshape(reshape_wi*sz,(height,width,n,2,2));
            if dbg: print(f"[P]A计算结束{A.shape}{qhat.shape}，耗时{round(time.time()-time_start,4)}秒")

            # frv--[h*w*n*1*2]*[h,w,n,2,2]--[h*w*n*1*2]--sum---[h*w*1*2]
            qhat=tf.reshape(qhat,(height,width,n,1,2))
            frv=tf.reduce_sum(tf.matmul(qhat, A),axis=2)
            #frv=tf.reshape(frv,(height,width,2))
            if dbg: print(f"[P]frv计算结束{frv.shape}，耗时{round(time.time()-time_start,4)}秒")

            #--fv
            fv=tf.norm(v_pstar[:,:,0,:,:],axis=2)/(tf.norm(frv,axis=3)+0.0000000001) * frv[:,:,0,:] + qstar
            if dbg: print(f"[P]fv计算结束{fv.shape},控制点{n}，耗时{round(time.time()-time_start,4)}秒")
            mapxy=tf.transpose(fv,(1,0,2))
            return mapxy

def getControlPoints(landmarks,malar_thin=0,jaw_thin=0,cheek_thin=0,small_mouth=0,long_face=0):
    if landmarks is None:
        return None,None
    pi=landmarks.copy().astype(int)
    qi=landmarks.copy().astype(int)
     
    #---颧骨
    if malar_thin!=0:  
        pts=[1,15]
        for idx in pts:
            distance=landmarks[30][0]-pi[idx][0]
            nPixel=(malar_thin*distance).astype(int)
            qi[idx][0]+=nPixel 
    #---面颊
    if cheek_thin!=0:
        pts=[4,5,12,11]
        for idx in pts:
            distance=landmarks[30]-pi[idx]
            nPixel=(cheek_thin*distance).astype(int)
            qi[idx]+=nPixel 
     
    #---下巴
    if jaw_thin!=0:
        pts=[6,12]
        for idx in pts:
            distance=landmarks[57]-pi[idx]
            nPixel=(jaw_thin*distance).astype(int)
            qi[idx]+=nPixel 

    #---长短脸
    if long_face!=0:  
        pts=[7,8,9]
        for idx in pts:
            distance=pi[idx][1]-landmarks[30][1]
            nPixel=(long_face*distance).astype(int)
            qi[idx][1]+=nPixel 

    #小嘴
    if small_mouth!=0:
        pts=[48,54]
        for idx in pts:
            distance=landmarks[62]-pi[idx]
            nPixel=(small_mouth*distance).astype(int)
            qi[idx]+=nPixel 

    deform_idx=[]
    for idx in range(len(pi)):
        if pi[idx][0]!=qi[idx][0] or pi[idx][1]!=qi[idx][1]:
            deform_idx.append(idx)
    return pi[deform_idx],qi[deform_idx]


def DeformFaceImage(img,landmarks,malar_thin=0,jaw_thin=0,cheek_thin=0,small_mouth=0,long_face=0):
    
    pi,qi=getControlPoints(landmarks,malar_thin=malar_thin,jaw_thin=jaw_thin,cheek_thin=cheek_thin,
                               small_mouth=small_mouth,long_face=long_face)
   
    if len(pi)==0:
        new_img=img
    else:
        #print(f"添加前pi.shape{pi.shape},qi.shape{qi.shape}-----------------------------")
        h,w,c=img.shape
        pi=np.append(pi,[[0,0],[h-2,0],[0,w-2],[h-2,w-2]],axis=0)
        qi=np.append(qi,[[0,0],[h-2,0],[0,w-2],[h-2,w-2]],axis=0)

        #pi=np.append(pi,[[0,0],[h-2,w-2]],axis=0)
        #qi=np.append(qi,[[0,0],[h-2,w-2]],axis=0)

        #print(f"pi.shape{pi.shape},qi.shape{qi.shape}-----------------------------")
        qi = pi * 2 - qi

        mapxy=MlsDeformTf.deformation_tf(img,pi,qi,dbg=True)
        time1=time.time()
        mapxy=mapxy.numpy()
        new_img = cv2.remap(img, mapxy[:, :, 0], mapxy[:, :, 1], borderMode=cv2.BORDER_REPLICATE, interpolation=cv2.INTER_LINEAR)
        time2=time.time()
        sd.FaceBeautyRemapTime=round((time2-time1)*1000)
    return new_img

def test_non_sqare_image():
    import core.cv2ex as cv2ex
    img_path=r'F:\Ai_VideoImage\真人原始图片\顾佳慧2.jpg' 
    img_path=r'D:\zjm.jpg' 
    img=cv2ex.cv2_imread(img_path)
    for i in range(4):
        pi=np.array([[3,5],[213,18],[22,266],[299,298]])
        qi=np.array([[3,15],[203,18],[29,216],[239,315]])
        qi = pi * 2 - qi
        time1=time.time()
        mapxy=MlsDeformTf.deformation_tf(img,pi,qi,dbg=True).numpy()
        new_img = cv2.remap(img, mapxy[:, :, 0], mapxy[:, :, 1], borderMode=cv2.BORDER_REPLICATE, interpolation=cv2.INTER_LINEAR)
        time2=time.time()
        print("一针耗时：",time2-time1)
        preview=np.hstack([img,new_img])
        cv2.imshow("img",preview); 
        cv2.waitKey(0)

if __name__ == '__main__':
    test_non_sqare_image()
    #img_path=r'F:\Ai_VideoImage\切脸\a_f91_0.jpg'
    #dfl=DFLJPG.load(img_path)
    #img=dfl.get_img()
    #landmarks=dfl.get_landmarks()
    #for i in range(40): 
    #    new_img=DeformFaceImage(img,landmarks,malar_thin=random.uniform(0.1,0.2),jaw_thin=random.uniform(0.1,0.15),
    #                           small_mouth=random.uniform(0.03,0.05),long_face=random.uniform(0.03,0.05))
    #    preview=np.hstack([img,new_img])
    #    cv2.imshow("img",preview); 
    #    cv2.waitKey(100)
    #    time.sleep(0.8)
    #cv2.waitKey(0)