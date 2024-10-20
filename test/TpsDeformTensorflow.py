import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import tensorflow as tf
 
DEVICE = "/gpu:0"

class TPS_Module():
      
    #Thin plate splines (TPS_Module) are a spline-based technique for datainterpolation and smoothing.
    #see: https://en.wikipedia.org/wiki/Thin_plate_spline
     
    def __init__(self, device):
        super().__init__() 
        self.device = device 
        self.grid=None;
        

    def create_grid(self,img):
        with  tf.device(self.device):
            height, width ,c= img.shape
            #self.grid = tf.ones( (h, w, 2))
            #self.grid[ :, :, 0] = tf.range(0,w)
            #self.grid[ :, :, 1] = tf.range(0, h).unsqueeze(-1)

            pcth = np.repeat(np.arange(height).reshape(height, 1), [width], axis=1)
            pctw = np.repeat(np.arange(width).reshape(width, 1), [height], axis=1).T
            img_coordinate = np.swapaxes(np.array([pcth,pctw]), 1, 2).T
            self.grid= tf.convert_to_tensor(img_coordinate,dtype=tf.float32)
            print("self.grid:",self.grid.shape)

    def forward(self, img,p, q):
        with  tf.device(self.device):
            self.create_grid(img)
            h, w ,c= self.grid.shape
            self.grid = tf.reshape( self.grid,(h*w, 2))
        
            U = self.K_matrix(self.grid, p)  #--[hw*n]
            P = self.P_matrix(self.grid)     #--[hw*3]
            W, A = self.calc_WA(p, q)     #
            mapxy = P @ A + U @ W
            mapxy=tf.reshape(mapxy,( h, w, 2))
            return mapxy


    def calc_WA(self,X,Y):

        n, k = X.shape[:2] 

        Z = tf.zeros( (n + 3, 2),dtype=tf.float32)
        P = tf.ones( (n, 3),dtype=tf.float32)
        L = tf.zeros((n + 3, n + 3),dtype=tf.float32)
        K = self.K_matrix(X, X)

        P[ :, 1:] = X
        Z[ :n, :] = Y
        L[ :n, :n] = K
        L[ :n, n:] = P
        L[ n:, :n] = P.permute(1, 0)
          
        inv_L=tf.inverse(L)
        S=tf.matmul(inv_L,Z)

        return S[:n,:], S[ n:,:]

    def K_matrix(self,G, X):
        eps = 1e-9
        D2 = tf.pow(G[:,None, :] - X[ None, :, :], 2) 
        D2=tf.reduce_sum(D2,axis=-1) 
        K = D2 * tf.math.log(D2 + eps) 
        return K


    def P_matrix(self,M):
        n,k = M.shape[:2]
        P = tf.ones( (n, 3) )
        P[ :, 1:] = M 
        return P

mTPS_Model=TPS_Module(device=DEVICE)

def get_image_mapxy(image,pi,qi):
    #t_image=tf.from_numpy(image).to(DEVICE).float()
    #t_pi=tf.from_numpy(pi).to(DEVICE).float()
    #t_qi=tf.from_numpy(qi).to(DEVICE).float()
    mapxy=mTPS_Model.forward(image,pi,qi)
    mapxy=mapxy.cpu().numpy()  
    return mapxy
 

def preview_tpss_deform(img):
    h,w,c=img.shape
    
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

    

    