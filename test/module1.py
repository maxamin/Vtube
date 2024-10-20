import numpy as np,cv2,sys
import multiprocessing,time,threading
import core.cv2ex as cv2ex
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtWidgets import QApplication,QMainWindow,QColorDialog,QSplashScreen
from PyQt5.QtCore import Qt,QSize
import PyQt5.QtCore as QtCore;

 
app = QApplication(sys.argv)
img=cv2ex.cv2_imread(r"F:\AiVideoSample\真人原始图片\异常有边缘的脸型\cut_face.png")
#img=cv2ex.cv2_imread(r"F:\AiVideoSample\真人原始图片\异常有边缘的脸型\34534.png")

img_src = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.uint8)
temp_imgSrc = QImage(img_src, img_src.shape[1], img_src.shape[0], QImage.Format_RGB888)
h,w,c=img_src.shape
hh=(h//4)*4
ww=(w//4)*4
print(f"width:{ww},{hh}")
temp_imgSrc=temp_imgSrc.copy(0,0,ww,hh)
#temp_imgSrc=temp_imgSrc.scaled(size)
pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc)
cv2.imshow("img",img)
cv2.waitKey(0)
#a=np.arange(1,37).reshape(3,3,4)
#print(a[2,1,3])
#a=a.swapaxes(0,1)
#print(a[1,2,3])
#print(a)

#for i in range(5):

#    img=cv2ex.cv2_imread(r"F:/Ai_VideoImage\真人原始图片\异常有边缘的脸型\R-C.jpg")
#    time1=time.time()
#    img=img.astype(float)
#    img=img.astype(np.uint8)
#    time2=time.time()
#    print(f"img shape:{img.shape}, convert float time:",time2-time1)

#time1=time.time()
#aa=np.random.randn(1000,1000,3)
#time2=time.time()
#print(f"randn time:",time2-time1)

#time1=time.time()
#pcth = np.repeat(np.arange(1000).reshape(1000, 1), [1000], axis=1)
#aa.reshape(100,10,100,10,3,1)
#time2=time.time()
#print(f"reshape time:",time2-time1)
# 按列表a中元素的值进行排序，并返回元素对应索引序列
#loss_list = [1.2, 3.5, 4, 5, 2, 7, 9,0,0.5,99,66]
#print('loss_list:', loss_list)
#sorted_id = sorted(range(len(loss_list)), key=lambda k: loss_list[k], reverse=True)
#print('元素索引序列：', sorted_id)

#def get_sorted_idx_list(loss_list):
#    sorted_id = sorted(range(len(loss_list)), key=lambda k: loss_list[k], reverse=True)
#    return sorted_id