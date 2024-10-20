import tensorflow as tf
import keras


class Downscale(keras.Model):
    def __init__(self,in_ch,out_ch,kernel_size=5):
        super.__init__()
        self.in_ch=in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.den=tf.nn.de
    def call(self,x):
        #keras.layers.Dense()
        #keras.layers.Conv2D()
        pass


class DeepFakeArchi:
    def __init__():
        pass
