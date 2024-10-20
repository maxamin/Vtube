import numpy as np,time,cv2
import numba
import numpy as np
from skimage import io
import math


class MlsAffineWarping(object):
    def __init__(self, cp, cq, whether_color_q=False, point_size=3):
        self._cp = cp
        self._cq = cq
        self._whether_color_q = whether_color_q
        self._point_size = point_size
        self._num_cpoints = len(cp)
        self._maximum = 2**31-1

    def update_cp(self, cp):
        self._cp = cp

    def update_cq(self, cq):
        self._cq = cq

    def check_is_cq(self, x, y):
        for i in range(self._num_cpoints):
            if abs(x - self._cq[i][0]) <= self._point_size and abs(y - self._cq[i][1]) <= self._point_size:
                return True
        return False

 
    def get_weights(self, input_pixel):
        #Weights = np.zeros(self._num_cpoints)
        Weights = []
        for i in range(self._num_cpoints):
            cpx, cpy = self._cp[i][0], self._cp[i][1]
            x, y = input_pixel[1], input_pixel[0]
            if x != cpx or y != cpy:
                weight = 1 / ((cpx - x) * (cpx - x) + (cpy - y) * (cpy - y))
            else:
                weight = self._maximum

            #Weights[i] = weight
            Weights.append(weight)

        return Weights

    @numba.jit()
    def getPStar(self, Weights):
        #numerator = np.zeros(2)
        numerator=[0,0]
        denominator = 0
        for i in range(len(Weights)):
            numerator[0] += Weights[i] * self._cp[i][0]
            numerator[1] += Weights[i] * self._cp[i][1]
            denominator += Weights[i]

        return numerator / denominator


    def getQStar(self, Weights):
        numerator=[0,0]
        denominator = 0
        for i in range(len(Weights)):
            numerator[0] += Weights[i] * self._cq[i][0]
            numerator[1] += Weights[i] * self._cq[i][1]
            denominator += Weights[i]

        return numerator / denominator

    def getTransformMatrix(self, p_star, q_star, Weights):
        sum_pwp = np.zeros((2, 2))
        sum_wpq = np.zeros((2, 2))
        for i in range(self._num_cpoints):
            tmp_cp = (np.array(self._cp[i]) - np.array(p_star)).reshape(1, 2)
            tmp_cq = (np.array(self._cq[i]) - np.array(q_star)).reshape(1, 2)

            sum_pwp += np.matmul(tmp_cp.T*Weights[i], tmp_cp)
            sum_wpq += Weights[i] * np.matmul(tmp_cp.T, tmp_cq)

        try:
            inv_sum_pwp = np.linalg.inv(sum_pwp)
        except np.linalg.linalg.LinAlgError:
            if np.linalg.det(sum_pwp) < 1e-8:
                return np.identity(2)
            else:
                raise

        return inv_sum_pwp*sum_wpq


    def transfer(self, data):
        row, col, channel = data.shape
        res_data = np.zeros((row, col, channel), np.uint8)

        for j in range(col):
            for i in range(row):
                input_pixel = [i, j]
                Weights = self.get_weights(input_pixel)
                p_star = self.getPStar(Weights)
                q_star = self.getQStar(Weights)
                M = self.getTransformMatrix(p_star, q_star, Weights)

                ## 逆变换版本
                try:
                    inv_M = np.linalg.inv(M)
                except np.linalg.linalg.LinAlgError:
                    if np.linalg.det(M) < 1e-8:
                        inv_M = np.identity(2)
                    else:
                        raise

                pixel = np.matmul((np.array([input_pixel[1], input_pixel[0]]) - np.array(q_star)).reshape(1, 2),
                                  inv_M) + np.array(p_star).reshape(1, 2)

                pixel_x = pixel[0][0]
                pixel_y = pixel[0][1]

                if math.isnan(pixel_x):
                    pixel_x = 0
                if math.isnan(pixel_y):
                    pixel_y = 0

                # pixel_x, pixel_y = max(min(int(pixel_x), row-1), 0), max(min(int(pixel_y), col-1), 0)
                pixel_x, pixel_y = max(min(int(pixel_x), col - 1), 0), max(min(int(pixel_y), row - 1), 0)

                if self._whether_color_q == True:
                    if self.check_is_cq(j, i):
                        res_data[i][j] = np.array([255, 0, 0]).astype(np.uint8)
                    else:
                        res_data[i][j] = data[pixel_y][pixel_x]
                else:
                    res_data[i][j] = data[pixel_y][pixel_x]

        return res_data


if __name__ == '__main__':
    # 里面输入你的图片位置，绝对位置和相对位置都可以
    from core import cv2ex
    from core.DFLIMG.DFLJPG import DFLJPG
    #img = cv2ex.cv2_imread(r'F:\AvVideo\真人原始图片\异常有边缘的脸型\Ha_Yeon_Soo_w.jpg')
    img_path=r'F:\TrainFaceLib\布丁\c13_0.jpg'
    img_path=r'F:\Ai_VideoImage\云飞\切脸\f61_0.jpg'
    img_path=r'D\yc.jpg'
    #img = cv2ex.cv2_imread(img_path)
    dfl=DFLJPG.load(img_path)
    img=dfl.get_img()
    landmarks=dfl.get_landmarks()
    print("img.shape",img.shape)
    print("landmarks.shape",landmarks.shape)
    #pi = np.array([[208, 536], [540, 347], [555, 513], [624, 608], [956, 489]])
    #qi = np.array([[208, 536], [560, 347], [580, 513], [684, 608], [956, 489]])


    face_widen_out_ratio=0.1
    face_thin_chin_ratio=0.2
    face_long_ratio=0.1
    mouth_small_ratio=0.3

    face_out_idx=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    mouth_out_idx=[48,49,50,51,52,53,54,55,56,57,58,59]
    mouth_out_idx=[48,54]
    deform_idx=face_out_idx+mouth_out_idx
    pi=landmarks[deform_idx].astype(int)
    qi=pi.copy()

    #---- 宽窄脸部
    for i in range(17):
        distance=landmarks[66][0]-landmarks[i][0]
        nPixel=(face_widen_out_ratio*distance).astype(int)
        qi[i][0]+=nPixel
    #---下巴
    for i in range(4,13):
        distance=landmarks[33]-landmarks[i]
        nPixel=(face_thin_chin_ratio*distance).astype(int)
        qi[i]+=nPixel
    #---嘴巴调整
    for i in range(17,19):
        distance=landmarks[62]-pi[i]
        nPixel=(mouth_small_ratio*distance).astype(int)
        qi[i]+=nPixel

    pi=pi[-3:]
    qi=qi[-3:]


    print("pi shape:",pi.shape)
    print("qi shape:",qi.shape)

    time0=time.time()
    mls = MlsAffineWarping(pi, qi, False)
    time1=time.time() 
    print("创建pi：",time1-time0)
    deform_img = mls.transfer(img)
    time2=time.time()
    
    print("变形qi：",time2-time1)
    target_width=512 

    for idx in range(pi.shape[0]):
        x,y=int(pi[idx][0]),int(pi[idx][1])
        nx,ny=int(qi[idx][0]),int(qi[idx][1])         
        #cv2.putText(img,f'{idx}',(x, y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
        cv2.circle(img, (x, y), int(2), (0, 255, 0), -1)
        cv2.circle(img, (nx, ny), int(2), (0, 0, 255), -1)
        cv2.circle(deform_img, (x, y), int(2), (0, 0, 255), -1)
    #img=cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    preview=np.hstack([deform_img,img])
    cv2.imshow('preview', preview)
    
    cv2.waitKey(0)

