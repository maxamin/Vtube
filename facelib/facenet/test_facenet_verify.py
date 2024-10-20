from facelib.facenet import facenet
import cv2,time

def TestVerifyFolderImages():
    import core.pathex as pathex
    img_list=pathex.get_image_paths(r"F:\FaceDataSet\FFHQ_ASIAN\Man5K\Align_wf",[".jpg",".png"])
    facenet.set_standard_face_img(r"F:\FaceDataSet\test\ttt_enhanced\91_0.jpg")
    for file_name in img_list:
        #print(file_name)
        time1=time.time();
        ok,distance,face_img=facenet.get_verify_result_from_standard(file_name,0.85)

        print(file_name,"|",distance,"|时间：",time.time()-time1);
        cv2.putText(face_img,str(ok),(30,30),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2,cv2.LINE_AA)
        cv2.putText(face_img,str(round(distance,3)),(30,130),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),1,cv2.LINE_AA)

        sleep_time=2000 if ok  else 100;
        cv2.imshow("diff",face_img)
        cv2.waitKey(sleep_time)
        
TestVerifyFolderImages()