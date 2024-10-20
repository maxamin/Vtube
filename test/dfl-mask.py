import cv2,numpy as np
from  core.DFLIMG.DFLJPG import DFLJPG
imgpath = r"C:\Users\ps\Pictures\随机遮罩\032359_4.jpg"
dfl=DFLJPG.load(imgpath)

img=dfl.get_img()
#cv2.imshow("img",img)

seg_ie_polys=dfl.get_seg_ie_polys()
if seg_ie_polys.has_polys():
    print("polygons points count:",seg_ie_polys.get_pts_count())
    #for poly in seg_ie_polys.get_polys():
    #    print(poly.get_pts())
    mask=np.zeros_like(img)
    overlay_mask=seg_ie_polys.overlay_mask(mask)
    #cv2.imshow("overlay_mask",mask*255)
else:
    print("no polygons points ")


xseg_mask=dfl.get_xseg_mask()
xseg_mask=cv2.resize(xseg_mask,(img.shape[1],img.shape[0]))
if xseg_mask is not None:
    xseg_mask=255*xseg_mask.astype(np.uint8)    
    contours,hier=cv2.findContours(xseg_mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    seg_ie_polys=dfl.get_seg_ie_polys()
    if seg_ie_polys.has_polys():
        seg_ie_polys.polys.clear()
    for contour in contours:
        contour=contour[0:-1:3]
        poly=seg_ie_polys.add_poly(ie_poly_type=1)
        for pt in contour:
            poly.add_pt(pt[0][0],pt[0][1])
    dfl.set_seg_ie_polys(seg_ie_polys)
    #dfl.save()

    #---输出轮廓信息
    print("contours count:",len(contours))
    print("hier count:",len(hier[0]),hier[0])
    hier=hier[0]
    for  i  in range(len(contours)):     
        contour=contours[i]
        contour=contour[0:-1:5]
        print("coutour points num:",len(contour))
        h=hier[i][3]
        if h>=0:
            color=(0,255,0)
        else:
            color=(0,0,255)
        for pt in contour:
            cv2.circle(img,pt[0],1,color,-1)

    #-----预览展示
    #xseg_mask=np.expand_dims(xseg_mask,2)
    #xseg_mask=np.repeat(xseg_mask,3,axis=2)
    #preview=np.hstack([xseg_mask,img])
    cv2.imshow("xseg_mask",xseg_mask)
    cv2.imshow("img",img)
cv2.waitKey(0)