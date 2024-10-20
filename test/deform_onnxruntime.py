import onnxruntime as ort
import cv2,time
import numpy as np

def test_by_onnxruntime():
    #session=ort.InferenceSession("D:/MlsPytorchDeform.onnx",providers=['CPUExecutionProvider','CUDAExecutionProvider'])
    device_ep="CUDAExecutionProvider"
    ep_flags={'device_id':0}
    session=ort.InferenceSession("D:/MlsPytorchDeform32.onnx",providers=[(device_ep,ep_flags)])
    input_names=session.get_inputs()
    output_names=session.get_outputs()
    for input in input_names:
        print("输入参数:",input.name)
    print(" ort.get_device:", ort.get_device()  )

    img=cv2.imread("D:/mj.jpg").astype(float)/255
    img=cv2.resize(img,(0,0),fx=0.5,fy=0.5)
    cv2.circle(img,(100,100),40,(0,0,1),thickness=-1)
    cv2.circle(img,(200,300),40,(0,1,0),thickness=-1)
    cv2.circle(img,(300,300),40,(1,0,0),thickness=-1)
    img=img.astype(np.float32)
    print("输入精度:",img.dtype)
    for i in range(5):
        time_start=time.time()
        pi=np.array([[3,5],[213,18],[22,266],[599,298],[221,266],[199,298],[221,266]])
        qi=np.array([[3,15],[203,18],[29,216],[239,315],[225,266],[199,298],[221,266]])

        inputs={"image":img,"pi":pi,"qi":qi}
        mapxy=session.run([output_names[0].name] ,inputs)[0]
        print(mapxy.shape)
        time_end=time.time()
        print("time use:",time_end-time_start)
        #return
        new_img = cv2.remap(img, mapxy[:, :, 0], mapxy[:, :, 1], borderMode=cv2.BORDER_REPLICATE, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("new_img",new_img) 
        cv2.waitKey(10)

        
    cv2.waitKey(0)


def test_dflive_onnx():
    img=cv2.imread("D:/yc.jpg").astype(float)/255

    device_ep="CUDAExecutionProvider"
    ep_flags={'device_id':0}
    session=ort.InferenceSession("E:\\VtubeKit_Dist\\Model_Lib\\景甜-LIAE256.dfm",providers=[(device_ep,ep_flags)])
    inputs=session.get_inputs()
    for input in inputs:
        print("输入参数:",input.name)
    print(" ort.get_device:", ort.get_device()  )
    
    shapes=inputs[0].shape
    w=shapes[1]
    print("输入维度:", shapes  )

    img=cv2.resize(img,(w,w))
    img=np.expand_dims(img,axis=0).astype(np.float32)
    print(img.shape,img.dtype)

    for i in range(5):
        time_start=time.time()
        out_face_mask, out_celeb, out_celeb_mask =session.run(None, {'in_face:0': img})
        cv2.imshow("new_img",out_celeb[0]) 
        time_end=time.time()
        print("推理时间:",time_end-time_start)
        cv2.waitKey(10)
    cv2.waitKey(0)

if __name__=="__main__":
    test_by_onnxruntime()
    test_dflive_onnx()