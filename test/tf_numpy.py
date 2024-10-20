
import numpy as np
import time
import set_env

set_env.set_env()
import tensorflow as tf

with tf.device("gpu:0"):

    frame_img=np.random.randn(1920,1080,3)
    mask_img=np.random.randn(1920,1080,3)
    swapped_img=np.random.randn(1920,1080,3)

    start=time.time()
    tensor_frame_img=tf.convert_to_tensor(frame_img,dtype="float32",)
    tensor_mask_img=tf.convert_to_tensor(mask_img,dtype="float32")
    tensor_swapped_img=tf.convert_to_tensor(swapped_img,dtype="float32")
    end=time.time()
    print("numpy 转tensor time:",round(end-start,5))

    start=time.time()
    t=tensor_frame_img*tensor_mask_img
    end=time.time()
    print("tf tensor time:",round(end-start,5))

    start=time.time()
    t=frame_img*mask_img
    end=time.time()
    print("numpy  cpu time:",round(end-start,5))

gpus = tf.config.experimental.list_physical_devices('CPU')
for gpu in gpus:
    print(gpu.name)


