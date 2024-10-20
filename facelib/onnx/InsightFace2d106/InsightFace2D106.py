from pathlib import Path
from typing import List
import numpy as np,cv2
from core.xlib.image import ImageProcessor
from core.xlib.onnxruntime import (InferenceSession_with_device, ORTDeviceInfo,
                              get_available_devices_info)

class InsightFace2D106:
    """
    arguments

     device_info    ORTDeviceInfo

        use InsightFace2D106.get_available_devices()
        to determine a list of avaliable devices accepted by model

    raises
     Exception
    """

    @staticmethod
    def get_available_devices() -> List[ORTDeviceInfo]:
        return get_available_devices_info()

    def __init__(self, device_info : ORTDeviceInfo):
        if device_info not in InsightFace2D106.get_available_devices():
            raise Exception(f'device_info {device_info} is not in available devices for InsightFace2D106')

        #path = Path(__file__).parent / 'InsightFace2D106.onnx'
        path = Path('../weights/InsightFace2D106.onnx') 
        if not path.exists():
            raise FileNotFoundError(f'{path} not found')
            
        self._sess = sess = InferenceSession_with_device(str(path), device_info)
        self._input_name = sess.get_inputs()[0].name
        self._input_width = 192
        self._input_height = 192

    def extract_lmks_68(self, img):
        """
        arguments
         img    np.ndarray      HW,HWC,NHWC uint8/float32
        returns (N,106,2)
        """
        ip = ImageProcessor(img)
        N,H,W,_ = ip.get_dims()

        if ip._img.dtype == np.float32:
            ip._img *= 255.0

        feed_img = ip.resize( (self._input_width, self._input_height) ).swap_ch().as_float32().ch(3).get_image('NCHW')

        uni_lmrks = self._sess.run(None, {self._input_name: feed_img})[0]
        uni_lmrks = uni_lmrks.reshape( (106,2))
        uni_lmrks /= 2.0
        uni_lmrks += (0.5, 0.5)
        uni_lmrks = uni_lmrks[self.lmrks_106_to_68_mean_pairs]
        uni_lmrks = uni_lmrks.reshape( (68,2,2)).mean(1)
        lmrks= uni_lmrks* (W, H)
        return uni_lmrks,lmrks

    def extract_lmks_106(self, img):
        """
        arguments
         img    np.ndarray      HW,HWC,NHWC uint8/float32
        returns (N,106,2)
        """
        ip = ImageProcessor(img)
        N,H,W,_ = ip.get_dims()

        if ip._img.dtype == np.float32:
            ip._img *= 255.0

        h_scale = H / self._input_height
        w_scale = W / self._input_width

        feed_img = ip.resize( (self._input_width, self._input_height) ).swap_ch().as_float32().ch(3).get_image('NCHW')

        uni_lmrks = self._sess.run(None, {self._input_name: feed_img})[0]
        uni_lmrks = uni_lmrks.reshape( (N,106,2))
        uni_lmrks /= 2.0
        uni_lmrks += (0.5, 0.5)
        uni_lmrks *= (w_scale, h_scale) 
        uni_lmrks = uni_lmrks*(W, H)
        return uni_lmrks
  
    lmrks_106_to_68_mean_pairs = [1,9, 10,11, 12,13, 14,15, 16,2, 3,4, 5,6, 7,8, 0,0, 24,23, 22,21, 20,19, 18,32, 31,30, 29,28, 27,26,25,17,
                   43,43, 48,44, 49,45, 51,47, 50,46,
                   102,97, 103,98, 104,99, 105,100, 101,101,
                   72,72, 73,73, 74,74, 86,86, 77,78, 78,79, 80,80, 85,84, 84,83,
                   35,35, 41,40, 40,42, 39,39, 37,33, 33,36,
                   89,89, 95,94, 94,96, 93,93, 91,87, 87,90,
                52,52, 64,64, 63,63, 71,71, 67,67, 68,68, 61,61, 58,58, 59,59, 53,53, 56,56, 55,55, 65,65, 66,66, 62,62, 70,70, 69,69, 57,57, 60,60, 54,54]
