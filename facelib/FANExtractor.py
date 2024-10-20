import os
import traceback
from pathlib import Path

import cv2,time
import numpy as np
from numpy import linalg as npla

from facelib import FaceType, LandmarksProcessor
from core.leras import nn

from typing import List
from core.xlib.image import ImageProcessor
from core.xlib.onnxruntime import (InferenceSession_with_device, ORTDeviceInfo,
                              get_available_devices_info)

"""
ported from https://github.com/1adrianb/face-alignment
"""
class FANExtractor(object):

    @staticmethod
    def get_available_devices() -> List[ORTDeviceInfo]:
        return get_available_devices_info()

    def __init__ (self, landmarks_3D=False, place_model_on_cpu=False):
        

        path = Path(__file__).parent / '2DFAN.onnx'
        if not path.exists():
            raise FileNotFoundError(f'{path} not found')
     
        tick_load_start=time.time()
        self._sess  = InferenceSession_with_device(str(path), FANExtractor.get_available_devices()[0])
        self._input_name = self._sess.get_inputs()[0].name
        self._input_width = 256
        self._input_height = 256

        tick_load_end=time.time()
        print(f"finish load FAN onnx,time:{tick_load_end-tick_load_start}")
        

    def extract (self, input_image, rects, second_pass_extractor=None, is_bgr=True, multi_sample=False):
        if len(rects) == 0:
            return []

        if is_bgr:
            input_image = input_image[:,:,::-1]
            is_bgr = False

        landmarks = []
        for (left, top, right, bottom) in rects:
            scale = (right - left + bottom - top) / 195.0
            center = np.array( [ (left + right) / 2.0, (top + bottom) / 2.0] )
             
            input_img=self.crop(input_image, center, scale) 
            input_img = input_img.astype(np.float32) / 255.0
            input_img= input_img[None,...] 
            predicted=self._sess.run(None, {self._input_name: input_img })[0]
            #print(predicted.shape)
            pts_img=self.get_pts_from_predict ( predicted[0], center, scale)
            #print("ptss:",pts_img.shape)
            landmarks.append (pts_img)

        return landmarks

    def transform(self, point, center, scale, resolution):
        pt = np.array ( [point[0], point[1], 1.0] )
        h = 200.0 * scale
        m = np.eye(3)
        m[0,0] = resolution / h
        m[1,1] = resolution / h
        m[0,2] = resolution * ( -center[0] / h + 0.5 )
        m[1,2] = resolution * ( -center[1] / h + 0.5 )
        m = np.linalg.inv(m)
        return np.matmul (m, pt)[0:2]

    def crop(self, image, center, scale, resolution=256.0):
        ul = self.transform([1, 1], center, scale, resolution).astype( np.int )
        br = self.transform([resolution, resolution], center, scale, resolution).astype( np.int )

        if image.ndim > 2:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0], image.shape[2]], dtype=np.int32)
            newImg = np.zeros(newDim, dtype=np.uint8)
        else:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
            newImg = np.zeros(newDim, dtype=np.uint8)
        ht = image.shape[0]
        wd = image.shape[1]
        newX = np.array([max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
        newY = np.array([max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
        oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
        oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
        newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1] ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]

        newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)), interpolation=cv2.INTER_LINEAR)
        return newImg

    def get_pts_from_predict(self, a, center, scale):
        a_ch, a_h, a_w = a.shape

        b = a.reshape ( (a_ch, a_h*a_w) )
        c = b.argmax(1).reshape ( (a_ch, 1) ).repeat(2, axis=1).astype(np.float)
        c[:,0] %= a_w
        c[:,1] = np.apply_along_axis ( lambda x: np.floor(x / a_w), 0, c[:,1] )

        for i in range(a_ch):
            pX, pY = int(c[i,0]), int(c[i,1])
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = np.array ( [a[i,pY,pX+1]-a[i,pY,pX-1], a[i,pY+1,pX]-a[i,pY-1,pX]] )
                c[i] += np.sign(diff)*0.25

        c += 0.5

        return np.array( [ self.transform (c[i], center, scale, a_w) for i in range(a_ch) ] )


    def export_fan_onnx(self):
        output_path="F:/AiML/FAN_2.onnx"
        tf = nn.tf
        output_graph_def=tf.get_default_graph().as_graph_def()
        tensor_name_list = [tensor.name for tensor in output_graph_def.node]
        for tensor in output_graph_def.node:
            if 'land' in tensor.name:
                print(tensor.name)
        #return
        output_graph_def = tf.graph_util.convert_variables_to_constants( nn.tf_sess,tf.get_default_graph().as_graph_def(),['landmarks'])
        import tf2onnx
        with tf.device("/CPU:0"):
            model_proto, _ = tf2onnx.convert._convert_common(
                output_graph_def,
                name='SAEHD',
                input_names=['in_face:0'],
                output_names=['landmarks:0'],
                opset=12,
                output_path=output_path)


            
if __name__ == '__main__':           
    nn.initialize_main_env()
    mFANExtractor=FANExtractor()
    mFANExtractor.export_fan_onnx()