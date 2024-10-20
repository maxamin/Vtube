import torch,cv2, numpy as np,time

class MlsDeformModule(torch.nn.Module):
    def __init__(self):
        super(MlsDeformModule, self).__init__()
        self.create_coordinate_buffer(1500,1500)

    def create_coordinate_buffer(self,width, height):
        #--- pytorch meshgrig 创建方式
        gridY = torch.arange(0,height)
        gridX = torch.arange(0,width)
        vy, vx = torch.meshgrid(gridX, gridY) 
        vy=vy.reshape( width,height,1)
        vx=vx.reshape( width,height,1)
        self.coordinate_buffer= torch.cat((vy, vx),dim=2).to("cuda")
        #print(f"[P]创建坐标变量,({self.coordinate_buffer.dtype}),{self.coordinate_buffer.shape}")

    def ensure_coordinate_vars(self,img):
        width_v, height_v,p = self.coordinate_buffer.shape
        height,width,c=img.shape
        #if height>height_v or width>width_v:
        #    self.create_coordinate_buffer(width, height)
        
    def forward(self,img,pi,qi):
        rigid = self.mls_rigid_deformation( img,pi, qi)
        return rigid

    def mls_rigid_deformation(self, image,pi, qi):
    
        time_start=time.time()
        

        self.ensure_coordinate_vars(image)
        qi = qi.float()
        pi = pi.float()
        print("-----------------------------")

        H,W = image.shape[0:2]  
        N = pi.shape[0]  # control points
        #print(f"图片的高度:{H},图片宽度:{W}")

        # Compute
        v=self.coordinate_buffer[0:W,0:H,:]
        reshaped_p = pi.reshape(1,1,N, 2)       # [1,1,N, 2]
        reshaped_v = v.reshape(W, H,1,2)    # [ H, W,1,2]
        #print(f"reshaped_p:{reshaped_p.shape},reshaped_v:{reshaped_v.shape}")
        print(f"步骤:计算完w,耗时:{time.time()-time_start}秒")
    
        
    
        #--w[ H, W,N]
        w = 1.0 / (torch.sum((reshaped_p - reshaped_v).float() ** 2, dim=3) + 0.000000001)    # [ H, W,N]
        #w /= torch.sum(w, dim=0, keepdim=True)                                               # [ H, W,N]
        #print(f"w维度:{w.shape},w[12,17]:{w[12,17]}")
        print(f"步骤:计算完w,耗时:{time.time()-time_start}秒")
    
        #--pstar[ H, W,2]
        sigma_wipi=torch.matmul(w,pi)  #[h*w*n]*[n*2]=[h*w*2]
        sigma_wi=torch.sum(w,axis=2,keepdim=True)  #[h*w*1] 
        #print(f"sigma_wipi:{sigma_wipi.shape},sigma_wi:{sigma_wi.shape},sigma_wipi[12,17]:{sigma_wipi[12,17]}")
        pstar=torch.divide(sigma_wipi,sigma_wi) #[h*w*2] 
        

        #--qstar[ H, W,2]
        sigma_wiqi=torch.matmul(w,qi)  #[h*w*n]*[n*2]=[h*w*2]
        qstar=torch.divide(sigma_wiqi,sigma_wi)   #[h*w*2] 
        #print(f"pstar维度:{pstar.shape},qstar维度:{qstar.shape},sigma_wiqi[12,17]:{sigma_wiqi[12,17]}")

        #--- 计算 pi^--[h*w*n*2]
        phat=torch.subtract(pi.reshape(1,1,N,2),pstar.reshape(W,H,1,2) )
        #--- 计算 qi^[h*w*n*2]
        qhat=torch.subtract(qi.reshape(1,1,N,2),qstar.reshape(W,H,1,2) )
        #print(f"phat维度:{phat.shape},qhat维度:{qhat.shape},qhat[12,17]:{qhat[12,17]}")

        # left mul matrix( pi^,-p^_vert)
        reshape_phat = phat.reshape( W, H,N,1,2)
        cat_phat=reshape_phat[:,:,:,:,1],-reshape_phat[:,:,:,:,0]
        phat_minus_vert=torch.cat(cat_phat,dim=3)
        phat_minus_vert=torch.reshape(phat_minus_vert,(W,H,N,1,2))
        #print(f"[P]phat_minus_vert维度:{phat_minus_vert.shape}") 

        #--- right mul matrix(v_pstar,-vpstar_vert) [H*W*2*2]
        #print("[P]v shape:",v.shape,"pstar.shape:",pstar.shape) #[W*H*2]
        vpstar=v-pstar
        reshaped_vpstar = vpstar.reshape( W,H,1,2)
        cat_vpstar=reshaped_vpstar[:,:,:,1],-reshaped_vpstar[:,:,:,0]
        vpstar_minus_vert=torch.cat(cat_vpstar,dim=2)
        #print(f"###数值对照vpstar_minus_vert[12,17]:{vpstar_minus_vert[12,17]}")
        #print(f"[P]vpstar_minus_vert维度:{vpstar_minus_vert.shape}") 
        vpstar_minus_vert=vpstar_minus_vert.reshape(W,H,1,2,1) #[W,H,1,2,1]
        vpstar_minus_vert=vpstar_minus_vert.repeat(1,1,N,1,1) #[W,H,n,2,1]
        reshaped_vpstar = vpstar.reshape( W,H,1,2,1).repeat(1,1,N,1,1)
        #print(f"[P]reshaped_vpstar计算结束{reshaped_vpstar.shape}") 
        print(f"步骤:计算完vpstar,耗时:{time.time()-time_start}秒,{reshaped_vpstar.device}")
        
        #return v.permute(1,0,2).float()
        #print(f"reshape_phat维度:{reshape_phat.shape},phat_minus_vert维度:{phat_minus_vert.shape}")
        #print(f"reshaped_vpstar维度:{reshaped_vpstar.shape},vpstar_minus_vert维度:{vpstar_minus_vert.shape}")
        a = torch.matmul(reshape_phat, reshaped_vpstar) # height*width*n*1*1
        b = torch.matmul(reshape_phat, vpstar_minus_vert)
        c = torch.matmul(phat_minus_vert, reshaped_vpstar)
        d = torch.matmul(phat_minus_vert, vpstar_minus_vert)
        sz=torch.cat((a,b,c,d), dim=3)
        sz=torch.reshape(sz,(W,H,N,4))
       

        # A-[H, W,N,2,2]
        reshape_w = w.reshape( W,H,N,1).repeat(1,1,1,4)
        A=torch.reshape(reshape_w*sz,(W,H,N,2,2));

        #print(f"A维度:{A.shape},数值对照A[12,17]:{A[12,17]}")
        print(f"步骤:计算完A,耗时:{time.time()-time_start}秒")
     
        # frv=[W,H,1,2]
        qhat=qhat.reshape(W,H,N,1,2)
        frv=torch.sum(torch.matmul(qhat, A),axis=2)   #[W,H,n,1,2]-sum- [W,H,1,2]
        #print(f"frv:{frv.shape},qstar:{qstar.shape},vpstar:{vpstar.shape}")
        vpstar_norm=torch.norm(reshaped_vpstar[:,:,0,:,:],dim=2)
        frv_norm=torch.norm(frv,dim=3)+0.0000000001
        frv=frv.reshape(W,H,2)
        #print(f"最后一步：vpstar_norm:{vpstar_norm.shape},frv_norm:{frv_norm.shape}")
        fv=vpstar_norm* frv /frv_norm   + qstar
        mapxy=fv.permute(1,0,2)
        torch.cuda.synchronize()
        print(f"步骤:计算完mapxy,耗时:{time.time()-time_start}秒")
        print(f"[P]模型运行总结,控制点{N},耗时{round(time.time()-time_start,4)}秒")
        
        
        return mapxy

   

def save_as_onnx_models(img):
    model=MlsDeformModule();
    pi = np.array([
        [155, 30], [155, 125], [155, 225],
        [235, 100], [235, 160], [295, 85], [293, 180]
    ])
    qi = np.array([
        [211, 42], [155, 125], [100, 235],
        [235, 80], [235, 140], [295, 85], [295, 180]
    ])

    device="cuda"
    model=model.to(device)
    img=torch.from_numpy(img).float().to(device)
    pi=torch.from_numpy(pi).to(device)
    qi=torch.from_numpy(qi).to(device)
    input_names=["image","pi","qi"]
    output_names=["mapxy"]
    torch.onnx.export(model,(img,pi,qi),"E:/MlsPytorchDeform32.onnx",opset_version=11,input_names=input_names,output_names=output_names,
                  dynamic_axes={"image":{0:"width",1:"height",2:"channel"},
                                "pi":{0:"NP"},
                                "qi":{0:"NQ"},
                                 "mapxy":{0:"height",1:"width"}},    );
    
def preview_mls_deform(img):
    model=MlsDeformModule();
    device="cuda"
    model=model.to(device)
    pi=np.array([[3,5],[213,18],[22,266],[269,298],[221,266],[199,298],[199,298]])
    qi=np.array([[3,15],[203,18],[29,216],[239,315],[225,266],[199,298],[199,298]])

    qi = pi * 2 - qi
    imgt=torch.from_numpy(img).to(device)
    pi=torch.from_numpy(pi).to(device)
    qi=torch.from_numpy(qi).to(device)
    for i in range(3):
        
        time_start=time.time()
        torch.cuda.synchronize()
        mapxy=model(imgt,pi,qi)
        print("输出的坐标映射维度和数据类型:",mapxy.shape,mapxy.dtype)
        #torch.cuda.synchronize()
        time_end_calc=time.time()
        print("计算步骤耗时:",time_end_calc-time_start,mapxy.dtype)

        #time.sleep(0.2)
        
        time_start_cpu=time.time()
        torch.cuda.synchronize()
        mapxy=mapxy.cpu()
        torch.cuda.synchronize()
        time_end_cpu=time.time()
        print("Tensor转到CPU步骤耗时:",time_end_cpu-time_start_cpu,mapxy.dtype)
        
        time_remap_start=time.time()
        mapxy=mapxy.numpy()
        new_img = cv2.remap(img, mapxy[:, :, 0], mapxy[:, :, 1], borderMode=cv2.BORDER_REPLICATE, interpolation=cv2.INTER_LINEAR)
        time_remap_end=time.time()
        print("映射坐标步骤time use:",time_remap_end-time_remap_start,new_img.dtype)
        cv2.imshow("new_img",np.hstack([img,new_img]))
        
        cv2.waitKey(1000)
    cv2.waitKey(0)

if __name__=="__main__":
    
   
    img=cv2.imread("D:/mai.jpg").astype(float)/255
    cv2.circle(img,(100,100),40,(0,0,1),thickness=-1)
    cv2.circle(img,(200,300),40,(0,1,0),thickness=-1)
    cv2.circle(img,(300,300),40,(1,0,0),thickness=-1)

    print("-----save_as_onnx_models----")
    #save_as_onnx_models(img)

    print("---show preview----")
    preview_mls_deform(img)