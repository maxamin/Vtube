import numpy as np,cv2
import multiprocessing,time,threading
import core.cv2ex as cv2ex
from core.DFLIMG import DFLJPG

dflimg=DFLJPG.load(r"F:/FaceSampleLib\dy钙钙\HD\c31_0.jpg");
img=dflimg.get_img()

xseg_mask=dflimg.get_xseg_mask()
kernel = np.ones((4, 4), np.uint8)
xseg_mask = cv2.dilate(xseg_mask, kernel, iterations = 1)
xseg_mask=cv2.resize(xseg_mask,img.shape[0:2])
xseg_mask=np.expand_dims(xseg_mask,2)

xseg_img = img.astype(np.float32)/255.0
xseg_overlay_mask = xseg_img*(1-xseg_mask)*0.5 + xseg_img*xseg_mask
cv2.imshow("merge",xseg_overlay_mask)
cv2.waitKey(0)