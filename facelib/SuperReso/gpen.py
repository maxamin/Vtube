import set_env
set_env.set_env()
from torch._C import device
from facelib.SuperReso.face_model.gpen_model import FullGenerator, FullGenerator_SR
import torch,cv2,numpy as np,time;
from pathlib import Path
from core import cv2ex


in_resolution=512; n_mlp=8; channel_multiplier=2; narrow=1;
device="cuda"; model=None;

def Load_model():
    print(f"开始加载GPEN超分辨模型，请等待")
    torch.backends.cudnn.benchmark=False
    global model,in_resolution,n_mlp,channel_multiplier,narrow,device
    start_time_tick=time.time()
    model = FullGenerator(in_resolution, 512, n_mlp, channel_multiplier, narrow=narrow, device=device)
    #model_path=Path(__file__).parent/'GPEN-BFR-512.pth'
    pretrained_dict = torch.load('../weights/GPEN-BFR-512.pth')
    model.load_state_dict(pretrained_dict)
    model.half().to(device)
    model.eval()
    end_time_tick=time.time()
    print(f"加载GPEN超分辨模型耗时:{end_time_tick-start_time_tick}秒",end='\r')

def img2tensor( img):
        img = cv2.resize(img, (512, 512))
        if img.dtype==np.uint8:
            img_t = torch.from_numpy(img).to(device)/255.
        else :
            img_t = torch.from_numpy(img).to(device)
        
        img_t = (img_t - 0.5) / 0.5
        img_t = img_t.permute(2, 0, 1).unsqueeze(0).flip(1) # BGR->RGB        
        return img_t.half()

def tensor2img( img_t,imtype=np.uint8):
    img_t = img_t * 0.5 + 0.5
    img_t = img_t.squeeze(0).permute(1, 2, 0).flip(2) # RGB->BGR
    img_np = img_t.float().cpu().numpy()
    if imtype==np.uint8:
        return (img_np*255).astype(np.uint8)
    return img_np

def process(img,out_dtype=np.float):
    global model,in_resolution,n_mlp,channel_multiplier,narrow,device
    if model is None: Load_model();
    start_time_tick=time.time()
    
    input_tensor=img2tensor(img)    
    end_time_tick=time.time()
    #print(f"GPEN转Tensor:{end_time_tick-start_time_tick}秒")
    with torch.no_grad():
        out_512px,_=model(input_tensor)
        torch.cuda.synchronize() 
        end_time_tick=time.time()
        #print(f"GPEN模型推理耗时:{end_time_tick-start_time_tick}秒")
    
    out_512px = tensor2img(out_512px,imtype=out_dtype)
    old_height,old_width=img.shape[0:2]
    out_same_size=cv2ex.cv2_resize(out_512px,(old_height,old_width)) 
    end_time_tick=time.time()
    #print(f"GPEN转换回图片耗时:{end_time_tick-start_time_tick}秒")
    return out_same_size
    



def export_onnx():
    model = FullGenerator(512, 512, 8, channel_multiplier=2, narrow=1, device='cpu')
    model.eval()
    pretrained_dict = torch.load(r'F:\AiML\GPEN-BFR-512.pth')
    model.load_state_dict(pretrained_dict)
    
    input_tensor= torch.randn(1,3,512,512)
    onnx_path="F:/AiML/GPEN-BFR-512.onnx"
    
    print("start export gpen-bfr-512 onnx")
    try:
        torch.onnx.export(model,input_tensor,onnx_path,verbose=False)
        print("finish export gpen-bfr-512 onnx")
    except Exception as E:
        print("error export",E)

def test_enhance(path):   
    image=cv2.imread(path)
    start_time_tick=time.time()
    image=cv2.resize(image,(512,512))
    img,out=process(image)
    compare=np.hstack((image,out))
    end_time_tick=time.time()
    print(f"GPen超分辨图片耗时:{end_time_tick-start_time_tick}秒")
    cv2.imshow("compare",compare)
    cv2.waitKey(0)

#export_onnx()
if __name__=="__main__":
    #export_onnx()
     for i in range(8):
        #img = cv2.imread("F:/f.jpg") 
        test_enhance(r"F:/f.jpg") 
        cv2.waitKey(0)
    