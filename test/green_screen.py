import set_env
set_env.set_env()
import torch,numpy,cv2,time;

import os,sys


def GetScreenMatImage(img,bg_tensor):
    time_start=time.time() 
    img=img.astype(numpy.float32)/255
    time_end=time.time()
    time_use=round((time_end-time_start)*1000);
    print(f"图片从uint8转换为浮点数变量时间:{time_use}ms")


    time_start=time.time() 
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    hsv_tensor=torch.from_numpy(hsv)
    img_tensor=torch.from_numpy(img)
    time_end=time.time()
    time_use=round((time_end-time_start)*1000);
    print(f"转换为torch张量:{time_use}ms,max(tensor")

    time_start=time.time() 
    
    mask_h_high=torch.where(hsv_tensor[:,:,0]>115,1,0)
    mask_h_low=torch.where(hsv_tensor[:,:,0]<125,1,0)
    mask_s_low=torch.where(hsv_tensor[:,:,0]>1,1,0)
    mask_v_low=torch.where(hsv_tensor[:,:,0]>1,1,0)

    mask_bg=mask_h_high*mask_h_low*mask_s_low*mask_v_low
    mask_bg=mask_bg.unsqueeze(2).float()
    cv2.imshow("mask",mask_bg.cpu().numpy())

    mask_fg=1-mask_bg 
    print(f"mask:{mask_fg.shape}")
    merge=mask_fg*img_tensor+mask_bg*bg_tensor
    time_use=round((time_end-time_start)*1000);
    time_end=time.time()
    print(f"torch张量计算绿幕:{time_use}ms")

    merge=merge.cpu().float().numpy()
    return merge
    
def GetScreenMatImageInt(img,bg):
    time_start=time.time() 
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV);
    green_mask=cv2.inRange(hsv, (35, 43, 46), (77, 255, 255) );
    human_mask=cv2.bitwise_not(green_mask) 
    if img.shape!=bg.shape:
        bg=cv2.resize(bg,(img.shape[1],img.shape[0]))
    bg = cv2.bitwise_and(bg,bg,mask=green_mask)
    man = cv2.bitwise_and(img,img,mask=human_mask)
    result=cv2.add(man,bg)

    time_end=time.time()
    time_use=round((time_end-time_start)*1000);
    print(f"融合耗时:{time_use}ms")
    return result
 
    

if __name__=="__main__":
    from core import cv2ex
    img=cv2ex.cv2_imread("F:/fore_green.jpg")
    bg=cv2ex.cv2_imread("F:/Hall_02.jpg")
    #bg=bg.astype(numpy.float32)/255
    #bg_tensor=torch.from_numpy(bg)
    merge=GetScreenMatImageInt(img,bg)
    cv2.imshow("merge",merge)
    cv2.waitKey(0)