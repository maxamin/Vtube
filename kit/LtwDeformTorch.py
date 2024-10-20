#coding=utf-8
# cython:language_level=3

import torch,time,numpy as np,cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

class LTW_Module(torch.nn.Module):
      
     
    def __init__(self, device=DEVICE):
        super().__init__() 
        self.device = device 
        self.V=None;
        

    def create_grid(self,img):
        h, w ,c= img.shape
        self.V = torch.ones( h, w, 2, device=self.device)
        self.V[ :, :, 0] = torch.arange(0,w)
        self.V[ :, :, 1] = torch.arange(0, h).unsqueeze(-1)
        #print("self.V:",self.V.shape)
        

    def forward(self, img,S, T,R):
        self.create_grid(img)
        V=self.V
        h, w ,c= self.V.shape 
        n=S.shape[0]
        VS=V.reshape(h,w,1,2)-S.reshape(1,1,n,2) #[h-w-n-2]
        ST=S-T #[n,2]
        VS2=torch.sum(VS*VS,dim=3)  #[h,w,n]
        #print("VS2 shape:",VS2.shape)
        R2=(R*R).reshape(1,1,n)  #[h,w,n]
        ST2=torch.sum(ST*ST,dim=1).reshape(1,1,n);
        R2_VS2=R2-VS2+0.000001
        #RR2_VS2=4*R2-VS2+0.000001
        B=torch.relu(R2_VS2)/(torch.abs(R2_VS2)+0.0001) #[h-w-n]
        r=R2_VS2/(R2_VS2+ST2+0.000001)
        W=r*r*B #[h,w,n]
        #print("W shape:",W.shape)
           
        M=torch.matmul(W,ST)
        print("坐标位移值维度：",M.shape)
        mapxy=V+M
        #print("M[480,480]",M[480,390])
        return mapxy

mLTW_Module=LTW_Module(device=DEVICE)
def get_image_mapxy(image,pi,qi,r):
    t_image=torch.from_numpy(image).to(DEVICE).float()
    t_pi=torch.from_numpy(pi).to(DEVICE).float()
    t_qi=torch.from_numpy(qi).to(DEVICE).float()
    t_R=torch.from_numpy(r).to(DEVICE).float()
    mapxy=mLTW_Module(t_image,t_pi,t_qi,t_R)
    mapxy=mapxy.cpu().numpy()  
    return mapxy

def preview_ltw_deform(img):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    h,w,c=img.shape 
    for i in range(4):
        time_start=time.time()
        S=np.array([[214,555],[142,484],[384,522],[418,428]])
        T=np.array([[245,533],[176,475],[368,513],[398,426]])
        R=np.array([20,20,6,6])
  
        mapxy=get_image_mapxy(img,S,T,R)
        print("GPU处理时间：",round(time.time()-time_start,3))
        new_img = cv2.remap(img, mapxy[:, :, 0], mapxy[:, :, 1], borderMode=cv2.BORDER_REPLICATE, interpolation=cv2.INTER_LINEAR)
        time_end=time.time()
        print("总处理时间：",round(time_end-time_start,3))


    for x,y in S:
        cv2.circle(new_img,(int(x),int(y)),3,(0,255,0),3)
    for x,y in T:
        cv2.circle(new_img,(int(x),int(y)),3,(0,0,255),3)

    cv2.imshow("img",np.hstack([img,new_img]))
    cv2.waitKey(0)


if __name__=="__main__":
    import core.cv2ex as cv2ex
    img=cv2ex.cv2_imread("D:/zjm.jpg")  
    preview_ltw_deform(img)

    #from PIL import Image
    #import matplotlib.pyplot as plt
    #img=Image.open ( 'D:/zjm.jpg')
    #plt.figure("Image")
    #plt.imshow(img)
    #plt.show()
    