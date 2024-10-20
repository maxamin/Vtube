from pathlib import Path
from typing import List

from core.xlib.image import ImageProcessor
from core.xlib.onnxruntime import (InferenceSession_with_device, ORTDeviceInfo,
                              get_available_devices_info)


class XSegOnnx:
    """
    Google XSegOnnx detection model.
    
    arguments

     device_info    ORTDeviceInfo

        use XSegOnnx.get_available_devices()
        to determine a list of avaliable devices accepted by model

    raises
     Exception
    """

    @staticmethod
    def get_available_devices() -> List[ORTDeviceInfo]:
        return get_available_devices_info()

    def __init__(self, device_info : ORTDeviceInfo):
        if device_info not in XSegOnnx.get_available_devices():
            raise Exception(f'device_info {device_info} is not in available devices for XSegOnnx')

        path = Path(__file__).parent / 'XSeg_model.onnx'
        if not path.exists():
            raise FileNotFoundError(f'{path} not found')
            
        self._sess = sess = InferenceSession_with_device(str(path), device_info)
        self._input_name = sess.get_inputs()[0].name
        self._input_width = 256
        self._input_height = 256

    def get_resolution(self):
        return self._input_width

    def extract(self, img):
        
        ip = ImageProcessor(img)
        N,H,W,_ = ip.get_dims()

        h_scale = H / self._input_height
        w_scale = W / self._input_width

        feed_img = ip.resize( (self._input_width, self._input_height) ).to_ufloat32().ch(3).get_image('NHWC')

        mask = self._sess.run(None, {self._input_name: feed_img})[0] 

        return mask
