#coding=utf-8
# cython:language_level=3
import cv2,time
import onnxruntime as ort
import numpy as np
import set_env
import torch
from torchvision.transforms import Resize 

class MattingRvm:
    sess=None;

    def __init__(self,mode="onnx",half=False):
        self.half=half;
        if half==True:
            model_path="../weights/rvm_mobilenetv3_fp16.onnx"
            infer_dtype=np.float16;
        else:
            model_path="../weights/rvm_mobilenetv3_fp32.onnx"
            infer_dtype=np.float32;

        ep_flags = {'device_id':0}
        self.sess = ort.InferenceSession(model_path,providers=[ ("CUDAExecutionProvider", ep_flags) ]) 
        self.io = self.sess.io_binding() 
        self.recur = [ ort.OrtValue.ortvalue_from_numpy(np.zeros([1, 1, 1, 1], dtype=infer_dtype), 'cuda') ] * 4
        self.downsample_ratio = ort.OrtValue.ortvalue_from_numpy(np.asarray([0.3], dtype=np.float32), 'cuda')
        for name in ['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o']:
            self.io.bind_output(name, 'cuda')
             
    def MatFrame(self,frame):
        beg = time.time()
        if frame.dtype==np.uint8:
            if self.half:
                frame=frame.astype(np.float16)/255
            else:
                frame=frame.astype(np.float32)/255
        src=np.expand_dims(frame,0) 
        src=np.transpose(src,[0,3,1,2]);
        self.io.bind_cpu_input('src', src)
        self.io.bind_ortvalue_input('r1i', self.recur[0])
        self.io.bind_ortvalue_input('r2i', self.recur[1])
        self.io.bind_ortvalue_input('r3i', self.recur[2])
        self.io.bind_ortvalue_input('r4i', self.recur[3])
        self.io.bind_ortvalue_input('downsample_ratio', self.downsample_ratio)
        convert_end=time.time()
        time_use = round( (convert_end - beg)*1000, 1)
        print(f"数据转换耗时:{time_use}ms") 
        
        self.sess.run_with_iobinding(self.io)        
        fgr, pha, *rec = self.io.get_outputs()
        infer_end = time.time()
        time_use = round( (infer_end - convert_end)*1000, 1)
        print(f"推理耗时:{time_use}ms") 
        #fgr = fgr.numpy().astype(np.float32)
        pha = pha.numpy().astype(np.float32)
        #fgr=np.transpose(fgr[0],[1,2,0]);
        pha=np.transpose(pha[0],[1,2,0]);
        post_end = time.time()
        time_use = round( (post_end - infer_end)*1000, 1)
        print(f"推理后处理耗时:{time_use}ms") 
        return fgr,pha

class MattingRvmTorch:
    def __init__(self,mode="torch",device="cuda:0",half=False):
        self.half=half;
        self.device="cuda:0"
        self.mat_model=None;
        self.width=1280;
        self.height=720;
        self.SetBackgroundImage(None);

    def GetMatModel(self,reload=False):
        from matting.MattingNetwork import MattingNetwork 

        if (self.mat_model is None) or (reload==True):            
            self.mat_model = MattingNetwork(variant='mobilenetv3').eval().to(self.device)
            self.mat_model.load_state_dict(torch.load('../weights/rvm_mobilenetv3.pth'))
            self.rec = [None] * 4
            print("完成加载人像分割Matting模型")
        return self.mat_model

    #------------- 设置背景图--------------
    #----- background_tensor(float32)(C,H,W)   background_img(C,H,W)(float32)
    def SetBackgroundImage(self,background_img:np.ndarray=None):
        if background_img is None:
            self.background_tensor=torch.ones([3,self.height,self.width])
            self.background_tensor[0,:,:]=0.4
            self.background_img=np.ones((self.height,self.width,3),dtype=np.float32)
            self.background_img_int=np.zeros((self.height,self.width,3),dtype=np.uint8)
            self.background_img_int[:,:,0]=255
        else:
            if background_img.dtype==np.uint8:
                self.background_img_int=background_img
                background_img_float=background_img.astype(np.float32)/255
                
            self.background_tensor=torch.from_numpy(background_img_float)
            self.background_tensor=self.background_tensor.to(self.device)
            self.background_tensor=torch.permute(self.background_tensor,(2,0,1))
            self.background_img=background_img_float  
        self.background_tensor=self.background_tensor.to(self.device)
        #print("设置了背景图")

    #---------- 调整背景图的大小--------------
    def ResizeBackgroundToImage(self,frame:np.ndarray):
        Hbg=self.background_tensor.shape[1]
        Wbg=self.background_tensor.shape[2]
        height,width=frame.shape[0:2]
        if Hbg==height and Wbg==width:
            return
        torch_resize = Resize([height,width])
        self.background_tensor = torch_resize(self.background_tensor)
        self.background_tensor=self.background_tensor.to(self.device);
        self.background_img=cv2.resize(self.background_img,(width,height))
        self.background_img_int=cv2.resize(self.background_img_int,(width,height))
        print(f"调整背景大小{width}x{height}，原来背景{Wbg}x{Hbg}")


    #================绿幕抠图背景融合=====================
    def GetGreenScreenFilter(self,img):
        self.ResizeBackgroundToImage(img)
        time_start=time.time() 
        hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV);
        green_mask=cv2.inRange(hsv, (35, 43, 47), (77, 255, 255) );
        human_mask=cv2.bitwise_not(green_mask) 

        bg = cv2.bitwise_and(self.background_img_int,self.background_img_int,mask=green_mask)
        man = cv2.bitwise_and(img,img,mask=human_mask)
        if bg.shape!=man.shape:
            return img
        result=cv2.add(man,bg)

        time_end=time.time()
        time_use=round((time_end-time_start)*1000);
        #print(f"融合耗时:{time_use}ms")
        return result
 

    #========================获得抠图融合后的图像==============
    def  GetAiMattingMerge(self,raw_frame,r=0.5):
        beg = time.time()
        self.ResizeBackgroundToImage(raw_frame)
        #---- 修改背景图尺寸匹配输入图 
        h,w,c=raw_frame.shape
        Hb,Wb=self.background_tensor.shape[1],self.background_tensor.shape[2]
        if h!=Hb or w!=Wb: 
            self.ResizeBackgroundToImage(raw_frame)
            self.GetMatModel(reload=True)
            return raw_frame
        #----- 转为tensor
        #if raw_frame.dtype==np.uint8:
        #   raw_frame=raw_frame.astype(np.float32)/255
        src=np.expand_dims(raw_frame,0)
        src=np.transpose(src,[0,3,1,2]);
        src_t=torch.from_numpy(src).to(self.device).float()/255 
        
        convert_end=time.time()
        convert_time_use = round( (convert_end - beg)*1000, 1)
         
        
        #--- 模型推理计算人像遮罩
        mat_model=self.GetMatModel(reload=False)
        with torch.no_grad():
            #print(f"rec:{len(self.rec)}")
            fg,mask, *self.rec = mat_model(src_t, *self.rec, downsample_ratio=r,segmentation_pass=False)

            raw_frame_t=src_t[0]
            mask=mask[0]
            wm=mask.shape[1]; wi=raw_frame_t.shape[1]; wb=self.background_tensor.shape[1]
            if wm!=wi or wm!=wb:
                return raw_frame
            if self.background_tensor.shape!=raw_frame_t.shape:
                return raw_frame
            merge_frame_t = mask * raw_frame_t + (1 - mask) * self.background_tensor

        infer_end = time.time()
        infer_time_use = round( (infer_end - convert_end)*1000, 1)
        
        #-----转换回Numpy数组
        merge_frame_t=torch.permute(merge_frame_t,(1,2,0))*255
        merge_frame_t=merge_frame_t.byte()
        merge_frame = merge_frame_t.cpu().numpy()    
        post_end = time.time()
        post_time_use = round( (post_end - infer_end)*1000, 1)
        #print(f"[P]转张量:{convert_time_use}ms,推理{infer_time_use}ms,转Numpy耗时:{post_time_use}ms")  
        
        return merge_frame

if __name__=="__main__":
    print("main")
    
    mat=MattingRvmTorch()

    from  core import cv2ex 

    bg=cv2ex.cv2_imread("F:/Hall_02.jpg")
    mat.SetBackgroundImage(bg)
 
    cap_video = cv2.VideoCapture(0)
    if not cap_video.isOpened():
        raise IOError("Error opening video stream or file.")
    
    width=1280;height=720;    
    cap_video.set(3,int(width))
    cap_video.set(4,int(height)) 
     
    
    while cap_video.isOpened():
        ret, raw_frame = cap_video.read()
        if ret:
            
            beg = time.time()
            raw_frame_float=raw_frame.astype(np.float32)/255
            
            merge_frame=mat.GetGreenScreenMerge(raw_frame_float,raw_frame)

            end = time.time()
            time_use = round( (end - beg)*1000, 1)
            print(f"总耗时:{time_use}ms\n") 
 
            #cv2.imshow('fgr', fgr)
            cv2.imshow('Matting', merge_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap_video.release()
    cv2.destroyWindow()