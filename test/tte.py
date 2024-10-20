import numpy as np,time,cv2


time1=time.time()
width, height = 1024,1024
pcth = np.repeat(np.arange(height).reshape(height, 1), [width], axis=1)
pctw = np.repeat(np.arange(width).reshape(width, 1), [height], axis=1).T
img_coordinate = np.swapaxes(np.array([pcth, pctw]), 1, 2).T
time2=time.time()
print(time2-time1)

#aa=np.random.rand(16,2)
#print(aa)
#bb=[1,2]
#d=np.multiply(aa,[10,100]).astype(np.int32);