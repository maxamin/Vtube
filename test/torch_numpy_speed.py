import set_env
set_env.set_env()
import torch,numpy,cv2,time;
print(torch.__version__)
#print(torch.cuda.is_available())
print("cuda version:",torch.version.cuda)
import os,sys

from core import cv2ex
img=cv2ex.cv2_imread("F:/fore_green.jpg")
bg=cv2ex.cv2_imread("F:/Bg_Hall.jpg")
bg=bg.astype(numpy.float32)/255
bg_tensor=torch.from_numpy(bg)

time_start=time.time() 
img=img.astype(numpy.float32)/255
time_end=time.time()
time_use=round((time_end-time_start)*1000);
print(f"图片从uint8转换为浮点数变量时间:{time_use}ms")


time_start=time.time() 
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
hsv_tensor=torch.from_numpy(hsv)
img_tensor=torch.from_numpy(img)
time_end=time.time()
time_use=round((time_end-time_start)*1000);
print(f"转换为torch张量:{time_use}ms,max(tensor")

time_start=time.time() 
time_end=time.time()
mask_h_high=torch.where(hsv_tensor[:,:,0]>110,1,0)
mask_h_low=torch.where(hsv_tensor[:,:,0]<130,1,0)
mask_s_low=torch.where(hsv_tensor[:,:,0]>60,1,0)
mask_v_low=torch.where(hsv_tensor[:,:,0]>60,1,0)

mask_bg=mask_h_high*mask_h_low*mask_s_low*mask_v_low
mask_bg=mask_bg.unsqueeze(2)
mask_fg=1-mask_bg 
print(f"mask:{mask_fg.shape}")
merge=mask_fg*img_tensor+mask_bg*bg_tensor
time_use=round((time_end-time_start)*1000);
print(f"torch张量计算绿幕:{time_use}ms")

merge=merge.cpu().float().numpy()
cv2.imshow("merge",merge)
cv2.waitKey(0)


time_start=time.time()
np_arr_int=numpy.random.randint(low=0,high=255,size=(720,1280,3))
np_arr_int=np_arr_int.astype(numpy.float32)/255
time_end=time.time()
time_use=round((time_end-time_start)*1000);
print(f"创建Numpy整数变量时间:{time_use}ms")

time_start=time.time() 
np_arr_float=numpy.random.randn(720,1280,3) 
time_end=time.time()
time_use=round((time_end-time_start)*1000);
print(f"创建正态随机变量时间:{time_use}ms,类型{np_arr_float.dtype}")

time_start=time.time()
np_arr=np_arr_float.astype(numpy.float32)/255
time_end=time.time()
time_use=round((time_end-time_start)*1000);
print(f"doubley变量转换为float时间:{time_use}ms")

print("----")
for i in range(3):
    time_start=time.time()
    tensor=torch.from_numpy(np_arr)
    tensor=tensor.to("cuda:0")
    torch.cuda.synchronize()
    time_end=time.time()
    time_use=round((time_end-time_start)*1000);
    print(f"Numpy转为TorchTensor变量时间:{time_use}ms")

print("----")
time_start=time.time()
np_arr=np_arr_float.astype(numpy.float32)/255
time_end=time.time()
time_use=round((time_end-time_start)*1000);
print(f"torch张量计算:{time_use}ms")

print("----")
for i in range(3):
    time_start=time.time()
    nparr=tensor.cpu().numpy()
    torch.cuda.synchronize()
    time_end=time.time()
    time_use=round((time_end-time_start)*1000);
    print(f"TorchTensor转为Numpy变量时间:{time_use}ms")