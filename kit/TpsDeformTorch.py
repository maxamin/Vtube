 
import torch
 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(DEVICE)
class TPS_Module(torch.nn.Module):
      
    #Thin plate splines (TPS_Module) are a spline-based technique for datainterpolation and smoothing.
    #see: https://en.wikipedia.org/wiki/Thin_plate_spline
     
    def __init__(self, size: tuple = (256, 256), device=DEVICE):
        super().__init__()
        h, w = size
        self.size = size
        self.device = device 
        self.grid=None;
        

    def create_grid(self,img):
        h, w ,c= img.shape
        self.grid = torch.ones( h, w, 2, device=self.device)
        self.grid[ :, :, 0] = torch.arange(0,w)
        self.grid[ :, :, 1] = torch.arange(0, h).unsqueeze(-1)
        #print("self.grid:",self.grid.shape)

    def forward(self, img,p, q):
        self.create_grid(img)
        h, w ,c= self.grid.shape
        self.grid = self.grid.view( h*w, 2)
        
        U = self.K_matrix(self.grid, p)  #--[hw*n]
        P = self.P_matrix(self.grid)     #--[hw*3]
        W, A = self.calc_WA(p, q)     #
        mapxy = P @ A + U @ W
        mapxy=mapxy.view( h, w, 2)
        return mapxy


    def calc_WA(self,X,Y):

        n, k = X.shape[:2]
        device = X.device

        Z = torch.zeros( n + 3, 2, device=device)
        P = torch.ones( n, 3, device=device)
        L = torch.zeros(n + 3, n + 3, device=device)
        K = self.K_matrix(X, X)

        P[ :, 1:] = X
        Z[ :n, :] = Y
        L[ :n, :n] = K
        L[ :n, n:] = P
        L[ n:, :n] = P.permute(1, 0)
          
        inv_L=torch.inverse(L)
        S=torch.matmul(inv_L,Z)

        return S[:n,:], S[ n:,:]

    def K_matrix(self,G, X):
        eps = 1e-9
        D2 = torch.pow(G[:,None, :] - X[ None, :, :], 2) 
        D2=torch.sum(D2,dim=-1) 
        K = D2 * torch.log(D2 + eps) 
        return K


    def P_matrix(self,M):
        n,k = M.shape[:2]
        device = M.device
        P = torch.ones( n, 3, device=device) 
        P[ :, 1:] = M 
        return P

mTPS_Model=TPS_Module()

def get_image_mapxy(image,pi,qi):
    t_image=torch.from_numpy(image).to(DEVICE).float()
    t_pi=torch.from_numpy(pi).to(DEVICE).float()
    t_qi=torch.from_numpy(qi).to(DEVICE).float()
    mapxy=mTPS_Model(t_image,t_pi,t_qi)
    mapxy=mapxy.cpu().numpy()  
    return mapxy

def save_as_onnx_models(img):
    
    pi = np.array([
        [155, 30], [155, 125], [155, 225],
        [235, 100], [235, 160], [295, 85], [293, 180]
    ])
    qi = np.array([
        [211, 42], [155, 125], [100, 235],
        [235, 80], [235, 140], [295, 85], [295, 180]
    ])
    model=TPS_Module();

    device="cuda"
    model=model.to(device)
    img=torch.from_numpy(img).float().to(device)
    pi=torch.from_numpy(pi).to(device)
    qi=torch.from_numpy(qi).to(device)
    input_names=["image","pi","qi"]
    output_names=["mapxy"]
    torch.onnx.export(model,(img,pi,qi),"D:/TpsTorchDeform32.onnx",opset_version=11,input_names=input_names,output_names=output_names,
                  dynamic_axes={"image":{0:"height",1:"width",2:"channel"},
                                "pi":{0:"NP"},
                                "qi":{0:"NQ"},
                                 "mapxy":{0:"height",1:"width"}},    );

def preview_tpss_deform(img):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    h,w,c=img.shape
    tps_model=TPS_Module(size=(h,w),device=DEVICE)
    for i in range(4):
        time_start=time.time()
        pi=np.array([[0,0],[0,h],[w,0],[w,h],[142,332],[452,342],[142,512],[452,512]])
        qi=np.array([[0,0],[0,h],[w,0],[w,h],[149,349],[412,332],[148,528],[402,510]])
  
        mapxy=get_image_mapxy(img,qi,pi)
        new_img = cv2.remap(img, mapxy[:, :, 0], mapxy[:, :, 1], borderMode=cv2.BORDER_REPLICATE, interpolation=cv2.INTER_LINEAR)
        time_end=time.time()
        print("处理时间：",time_end-time_start)


    for x,y in pi:
        cv2.circle(new_img,(int(x),int(y)),3,(0,255,0),3)
    for x,y in qi:
        cv2.circle(new_img,(int(x),int(y)),3,(0,0,255),3)

    cv2.imshow("img",np.hstack([img,new_img]))
    cv2.waitKey(0)


if __name__=="__main__":
    import cv2,numpy as np
    import time 
    import core.cv2ex as cv2ex
    img=cv2ex.cv2_imread("D:/yc.jpg")    
    #save_as_onnx_models(img)
    preview_tpss_deform(img)

    

    