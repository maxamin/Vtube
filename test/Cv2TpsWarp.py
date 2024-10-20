import numpy as np
import cv2

def WarpImage_TPS(source,target,img):
	tps = cv2.createThinPlateSplineShapeTransformer()

	source=source.reshape(-1,len(source),2)
	target=target.reshape(-1,len(target),2)

	matches=list()
	for i in range(0,len(source[0])):

		matches.append(cv2.DMatch(i,i,0))

	tps.estimateTransformation(target, source, matches)  # note it is target --> source

	new_img = tps.warpImage(img)

	# get the warp kps in for source and target
	tps.estimateTransformation(source, target, matches)  # note it is source --> target
	# there is a bug here, applyTransformation must receive np.float32 data type
	f32_pts = np.zeros(source.shape, dtype=np.float32)
	f32_pts[:] = source[:]
	transform_cost, new_pts1 = tps.applyTransformation(f32_pts)  # e.g., 1 x 4 x 2
	f32_pts = np.zeros(target.shape, dtype=np.float32)
	f32_pts[:] = target[:]
	transform_cost, new_pts2 = tps.applyTransformation(f32_pts)  # e.g., 1 x 4 x 2

	return new_img, new_pts1, new_pts2

def thin_plate_transform(x,y,offw,offh,imshape,shift_l=-0.05,shift_r=0.05,num_points=5,offsetMatrix=False):
	rand_p=np.random.choice(x.size,num_points,replace=False)
	movingPoints=np.zeros((1,num_points,2),dtype='float32')
	fixedPoints=np.zeros((1,num_points,2),dtype='float32')

	movingPoints[:,:,0]=x[rand_p]
	movingPoints[:,:,1]=y[rand_p]
	fixedPoints[:,:,0]=movingPoints[:,:,0]+offw*(np.random.rand(num_points)*(shift_r-shift_l)+shift_l)
	fixedPoints[:,:,1]=movingPoints[:,:,1]+offh*(np.random.rand(num_points)*(shift_r-shift_l)+shift_l)

	tps=cv2.createThinPlateSplineShapeTransformer()
	good_matches=[cv2.DMatch(i,i,0) for i in range(num_points)]
	tps.estimateTransformation(movingPoints,fixedPoints,good_matches)

	imh,imw=imshape
	x,y=np.meshgrid(np.arange(imw),np.arange(imh))
	x,y=x.astype('float32'),y.astype('float32')
	# there is a bug here, applyTransformation must receive np.float32 data type
	newxy=tps.applyTransformation(np.dstack((x.ravel(),y.ravel())))[1]
	newxy=newxy.reshape([imh,imw,2])

	if offsetMatrix:
		return newxy,newxy-np.dstack((x,y))
	else:
		return newxy

def getControlPoints(landmarks,narrow_face=0,jaw_thin=0,small_mouth=0,long_face=0):
    if landmarks is None:
        return None,None
    pi=landmarks.copy().astype(int)
    qi=landmarks.copy().astype(int)
     
    #---窄脸
    if narrow_face!=0:  
        pts=[2,14]
        for idx in pts:
            distance=landmarks[30][0]-pi[idx][0]
            nPixel=(narrow_face*distance).astype(int)
            qi[idx][0]+=nPixel 

    #---长短脸
    if long_face!=0:  
        pts=[8]
        for idx in pts:
            distance=landmarks[30][1]-pi[idx][1]
            nPixel=(long_face*distance).astype(int)
            qi[idx][1]+=nPixel 

    #---下巴
    if jaw_thin!=0:
        pts=[5,11]
        for idx in pts:
            distance=landmarks[33]-pi[idx]
            nPixel=(jaw_thin*distance).astype(int)
            qi[idx]+=nPixel 
    #小嘴
    if small_mouth!=0:
        pts=[48,54]
        for idx in pts:
            distance=landmarks[62]-pi[idx]
            nPixel=(small_mouth*distance).astype(int)
            qi[idx]+=nPixel 

    deform_idx=[]
    for idx in range(len(pi)):
        if pi[idx][0]!=qi[idx][0] or pi[idx][1]!=qi[idx][1]:
            deform_idx.append(idx)
    return pi[deform_idx],qi[deform_idx]

# the correspondences need at least four points
Zp = np.array([[217, 39], [204, 95], [174, 223], [648, 402]]) # (x, y) in each row
Zs = np.array([[283, 54], [166, 101], [198, 250], [666, 372]])
r = 6

from core.DFLIMG.DFLJPG import DFLJPG
img_path=r'F:\Ai_VideoImage\切脸\f66_0.jpg'
dfl=DFLJPG.load(img_path)
im=dfl.get_img()
landmarks=dfl.get_landmarks()
print("im.shape",im.shape)
print("landmarks.shape",landmarks.shape)

narrow_face=0.1
jaw_thin=0.2
face_long=0.1
mouth_small=0.05

Zp,Zs=getControlPoints(landmarks,narrow_face,jaw_thin,mouth_small,face_long)

print("控制点pi shape:",Zp.shape)
print("变形点qi shape:",Zs.shape)

# draw parallel grids
for y in range(0, im.shape[0], 10):
		im[y, :, :] = 255
for x in range(0, im.shape[1], 10):
		im[:, x, :] = 255

new_im, new_pts1, new_pts2 = WarpImage_TPS(Zp, Zs, im)
new_pts1, new_pts2 = new_pts1.squeeze(), new_pts2.squeeze()
print(new_pts1, new_pts2)

# new_xy = thin_plate_transform(x=Zp[:, 0], y=Zp[:, 1], offw=3, offh=2, imshape=im.shape[0:2], num_points=4)

for p in Zp:
	cv2.circle(im, (p[0], p[1]), r, [0, 0, 255])
for p in Zs:
	cv2.circle(im, (p[0], p[1]), r, [255, 0, 0])
cv2.imshow('w', im)
cv2.waitKey(500)


for p in Zs:
	cv2.circle(new_im, (p[0], p[1]), r, [255, 0, 0])
for p in new_pts1:
	cv2.circle(new_im, (int(p[0]), int(p[1])), 3, [0, 0, 255])
cv2.imshow('w2', new_im)
cv2.waitKey(0)

