from facelib.SuperReso.CodeFormerArchi import CodeFormer
import torch,cv2,numpy as np,time;

model=None;

def img2tensor( img,device):
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)   
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
    img_np = np.clip(img_t.float().cpu().numpy(), 0, 1)
    if imtype==np.uint8:
        return (img_np*255).astype(np.uint8)
    return img_np

def process(img,fidelity=0.8,device="cuda",out_dtype=np.float):
    global model;
    
    if model is None: 
        start_time_tick=time.time()
        model = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=['32', '64', '128', '256'])
        ckpt_path = '../weights/codeformer.pth'
        checkpoint = torch.load(ckpt_path)['params_ema']
        model.load_state_dict(checkpoint)
        model.half().to(device)
        model.eval()
        end_time_tick=time.time()
        print(f"加载CodeFormer超分辨模型耗时:{end_time_tick-start_time_tick}秒")
    
    h,w,c=img.shape    
    with torch.no_grad():
        start_time_tick=time.time()
        input=img2tensor(img,device)
        output_tensor = model(input, w=fidelity, adain=True)[0]
        end_time_tick=time.time()
        #print(f"CodeFormer模型推理耗时:{end_time_tick-start_time_tick}秒")
        sr_img=tensor2img(output_tensor,out_dtype)
        sr_img=cv2.resize(sr_img,(h,w))
    return sr_img

if __name__=="__main__":
     
    for i in range(10):
        img = cv2.imread("F:/f.jpg")
        start_time_tick=time.time()
        sr_img=process(img)
        end_time_tick=time.time()
        print(f"超分辨图片耗时:{end_time_tick-start_time_tick}秒")
        cv2.imshow("img",sr_img)
        cv2.waitKey(0)
    