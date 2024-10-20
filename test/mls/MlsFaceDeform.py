import numpy as np,time,cv2
import random
class MlsFaceDeform():
    img_coordinate=None;
    def __init__(self, img):
        self.create_img_coordinate(img)
        
    @staticmethod
    def create_img_coordinate(img):
       
        time_start=time.time()
        width, height = img.shape[:2]
        pcth = np.repeat(np.arange(height).reshape(height, 1), [width], axis=1)
        pctw = np.repeat(np.arange(width).reshape(1,width), [height], axis=0)
        MlsFaceDeform.img_coordinate = np.swapaxes(np.array([pcth, pctw]), 1, 2).T 
        time_end=time.time()
        print(f"创建img_coordinate变量,耗时{time_end-time_start},({MlsFaceDeform.img_coordinate.dtype}),{MlsFaceDeform.img_coordinate.shape}")
        print(MlsFaceDeform.img_coordinate[3,5])
    @staticmethod
    def assert_vars(img):
        if MlsFaceDeform.img_coordinate is None:
            MlsFaceDeform.create_img_coordinate(img)
        else:
            h,w=img.shape[0:2]
            ch,cw=MlsFaceDeform.img_coordinate.shape[0:2]
            if h!=ch or w!=cw:
                MlsFaceDeform.create_img_coordinate(img)
 
    @staticmethod
    def deformation(img,pi,qi):
        MlsFaceDeform.assert_vars(img);
        width, height = img.shape[:2]
        n=pi.shape[0]
        img_coordinate=MlsFaceDeform.img_coordinate
        qi = pi * 2 - qi

        time_start=time.time()
        # wi--[height*width*n]
        wi = np.reciprocal(np.power(np.linalg.norm(np.subtract(pi, img_coordinate.reshape(height, width, 1, 2)) + 0.000000001, axis=3),2))
        print(f"耗时统计-(wi计算){wi.shape}-{time.time()-time_start}")

        # p*---[height*width*2]
        pstar = np.divide(np.matmul(wi,pi), np.sum(wi, axis=2).reshape(height,width,1))
        print(f"耗时统计-(p*计算){pstar.shape}-{time.time()-time_start}")

        # p^--[height*width*n*2]
        phat = np.subtract(pi, pstar.reshape(height, width, 1, 2))
        print(f"耗时统计-(p^计算){phat.shape}-{time.time()-time_start}")

        # q*---[height*width*2]    q^--[height*width*n*2]
        qstar = np.divide(np.matmul(wi,qi), np.sum(wi, axis=2).reshape(height,width,1))
        qhat = np.subtract(qi, qstar.reshape(height, width, 1, 2)).reshape(height, width, qi.shape[0], 1, 2)
        print(f"耗时统计-(q*和q^计算)-{time.time()-time_start}")

        

        # s2--[height*width*n*1*2]
        reshape_phat = phat.reshape(height,width,n,1,2)
        s2 = np.concatenate((reshape_phat[:,:,:,:,1], -reshape_phat[:,:,:,:,0]), axis=3).reshape(height,width,n,1,2)
        print(f"耗时统计-(s2计算){s2.shape}-{time.time()-time_start}")

        #--- z2
        v_pstar = np.subtract(img_coordinate, pstar) #[h*w*2]
        z2 = np.repeat(np.swapaxes(np.array([v_pstar[:,:,1], -v_pstar[:,:,0]]), 1, 2).T.reshape(height,width,1,2,1), [n], axis=2)
        print(f"耗时统计-(z2计算)-{time.time()-time_start}")

        # height*width*n*2*1
        v_pstar = np.repeat(v_pstar.reshape(height,width,1,2,1), [n], axis=2)

        a = np.matmul(reshape_phat, v_pstar) # height*width*n*1*1
        b = np.matmul(reshape_phat, z2)
        c = np.matmul(s2, v_pstar)
        d = np.matmul(s2, z2)

        # 重构wi形状 [H*W*N*1---H*W*N*4]
        reshape_wi = np.repeat(wi.reshape(height,width,n,1),[4],axis=3)     

        # height*width*n*2*2
        A = (reshape_wi * np.concatenate((a,b,c,d), axis=3).reshape(height,width,n,4)).reshape(height,width,n,2,2)
         
        print(f"耗时统计-(A计算)-{A.shape}{time.time()-time_start}")

        
        frv = np.sum(np.matmul(qhat, A),axis=2)

        print(f"耗时统计-(frv计算)-{frv.shape}{time.time()-time_start}")
        fv = np.linalg.norm(v_pstar[:,:,0,:,:],axis=2) / (np.linalg.norm(frv,axis=3)+0.0000000001) * frv[:,:,0,:] + qstar
        print(f"耗时统计-(fv计算)-{fv.shape}{time.time()-time_start}")
       
        mapxy = np.swapaxes(np.float32(fv), 0, 1)
        new_img = cv2.remap(img, mapxy[:, :, 0], mapxy[:, :, 1], borderMode=cv2.BORDER_REPLICATE, interpolation=cv2.INTER_LINEAR)
        return  new_img
      
     

def getControlPoints(landmarks,narrow_face=0,jaw_thin=0,small_mouth=0,long_face=0):
    if landmarks is None:
        return None,None
    pi=landmarks.copy().astype(int)
    qi=landmarks.copy().astype(int)
     
    #---窄脸
    if narrow_face!=0:  
        pts=[1,15]
        for idx in pts:
            distance=landmarks[30][0]-pi[idx][0]
            nPixel=(narrow_face*distance).astype(int)
            qi[idx][0]+=nPixel 

    #---长短脸
    if long_face!=0:  
        pts=[8]
        for idx in pts:
            distance=pi[idx][1]-landmarks[30][1]
            nPixel=(long_face*distance).astype(int)
            qi[idx][1]+=nPixel 

    #---下巴
    if jaw_thin!=0:
        pts=[5,11]
        for idx in pts:
            distance=landmarks[33]-pi[idx]
            nPixel=(jaw_thin*distance).astype(int)
            qi[idx]+=nPixel 
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

if __name__ == '__main__':
    from core import cv2ex
    from core.DFLIMG.DFLJPG import DFLJPG
    img_path=r'F:\Ai_VideoImage\切脸\a_f91_0.jpg'
    dfl=DFLJPG.load(img_path)
    img=dfl.get_img()
    landmarks=dfl.get_landmarks()
    print("img.shape",img.shape)
    print("landmarks.shape",landmarks.shape)
     
    for i in range(5):
        print("------------------")
        time0=time.time()
        pi,qi=getControlPoints(landmarks,narrow_face=random.uniform(0.05,0.1),jaw_thin=random.uniform(0.1,0.15),
                               small_mouth=0,long_face=random.uniform(0.03,0.05))
        print("控制点pi shape:",pi.shape)
        print("变形点qi shape:",qi.shape)
        
        deform_img = MlsFaceDeform.deformation(img, pi,qi)
        time2=time.time() 
        print(f"变形总耗时：{time2-time0}秒" )
        target_width=512 

        for idx in range(len(pi)):
            x,y=int(pi[idx][0]),int(pi[idx][1])
            nx,ny=int(qi[idx][0]),int(qi[idx][1])         
            #cv2.putText(img,f'{idx}',(x, y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
            #cv2.circle(img, (x, y), int(4), (0, 255, 0), -1)
            #cv2.circle(img, (nx, ny), int(4), (0, 0, 255), -1)
            cv2.circle(deform_img, (x, y), int(4), (0, 255, 0), -1)
        #img=cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
        preview=np.hstack([deform_img,img])
        cv2.imshow('preview', preview)
        cv2.waitKey(2000)
        time.sleep(2)

    cv2.waitKey(0)
