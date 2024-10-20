from pathlib import Path
from core.xlib.image import ImageProcessor
import numpy as np;
import cv2,time;
from core.xlib.onnxruntime import InferenceSession_with_device, ORTDeviceInfo,get_available_devices_info
from core import cv2ex

class facenet:
#embedding=mInferSession.run(None, {mInputName: img})
    mInferSession = None; mInputName =None;

    @staticmethod
    def _confirm_onnx_model_loaded(): 
        if facenet.mInferSession is None:
            #facenet_onnx_path=Path(__file__).parent / 'facenet-20180402-114759-vggface2.onnx'
            facenet_onnx_path = Path('../weights/facenet-20180402-vggface2.onnx') 
            device_info_list=get_available_devices_info(include_cpu=False)
            device_info=device_info_list[0]
            facenet.mInferSession = InferenceSession_with_device(str(facenet_onnx_path), device_info)
            facenet.mInputName = facenet.mInferSession.get_inputs()[0].name
            print("加载人脸验证模型完成")

    @staticmethod
    def _get_image_for_facenet_input(current_img,target_size=(160,160)):
        h,w=current_img.shape[0:2]
        if (h,w)!=target_size :
            current_img = cv2.resize(current_img, target_size)
        return current_img
    @staticmethod
    def getImageEmbed(img):
        global mInputName;
        facenet._confirm_onnx_model_loaded();
        img=facenet._get_image_for_facenet_input(img)
        ip = ImageProcessor(img)
        img = ip.ch(3).swap_ch().to_ufloat32().get_image('NCHW')
        embedding=facenet.mInferSession.run(None, {facenet.mInputName: img})[0][0]
        return embedding;

    @staticmethod
    def getEmbedDistance(embed1,embed2):
        if embed1 is None or embed2 is None:
            return 22.0;
        return np.linalg.norm((embed1-embed2),2)

    #-----------  设置参考标准人脸-------------
    standard_face_embedding=None;
    ref_face_images=[]
    ref_face_embeds=[]

    @staticmethod
    def set_standard_face_img(img,person_idx=0,angle_idx=0): 
        if isinstance(img,str):
            img=cv2ex.cv2_imread(img);
        img_input=facenet._get_image_for_facenet_input(img)
        facenet.standard_face_embedding=facenet.getImageEmbed(img)

    @staticmethod
    def add_ref_face_img(img):
        img=cv2.resize(img,(256,256))
        embed=facenet.getImageEmbed(img)
        facenet.ref_face_images.append(img)
        facenet.ref_face_embeds.append(embed)

    @staticmethod
    def del_last_face_img():
        if len(facenet.ref_face_images)>=1:
            facenet.ref_face_images.pop(len(facenet.ref_face_images)-1);
            facenet.ref_face_embeds.pop(len(facenet.ref_face_embeds)-1);

    @staticmethod
    def getRefImagesMerge():
        if len(facenet.ref_face_images)==0:
            return np.zeros([128,128,3],dtype=np.uint8);
        merge=np.concatenate(facenet.ref_face_images,axis=1)
        return merge;

    @staticmethod
    def clear_standard_face():
        facenet.standard_face_embedding=None;
        facenet.ref_face_images.clear();
        facenet.ref_face_embeds.clear();

    @staticmethod
    def getVerifyResult(img,threshold=1.0):
        if len(facenet.ref_face_embeds)==0:
            return True,100
        img_face_embedding=facenet.getImageEmbed(img)
        for ref_embed in facenet.ref_face_embeds:
            distance=facenet.getEmbedDistance(img_face_embedding,ref_embed)
            if distance<threshold:
                return True,distance
        return False,5.0

    @staticmethod
    def get_verify_result_from_standard(img,threshold=1.0):
        if facenet.standard_face_embedding is None:
            return True,100.0,None
        if isinstance(img,str):
            img=cv2ex.cv2_imread(img);
        img_face_embedding=facenet.getImageEmbed(img)
        distance=facenet.getEmbedDistance(img_face_embedding,facenet.standard_face_embedding)
        ok=True if distance<=threshold else False
        return ok,distance,img
    
    @staticmethod
    def getImageFaceSimilarity(image1,image2):
        if image1 is None or image2 is None:
            return 30.0;
        try:
            embed_1=facenet.getImageEmbed(image1)
            embed_2=facenet.getImageEmbed(image2)
            dist=facenet.getEmbedDistance(embed_1,embed_2)
            return dist;
        except :
            return 20.0;
    




