import os
import pickle
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from core.interact import interact as io
from core.leras import nn
#from core.leras.models.XSegArchi import XSegArchi

class XSegNet(object):
    VERSION = 1

    def __init__ (self, name, 
                        resolution=256, 
                        load_weights=True, 
                        weights_file_root=None, 
                        training=False, 
                        place_model_on_cpu=False, 
                        run_on_cpu=False, 
                        optimizer=None, 
                        data_format="NHWC",
                        raise_on_no_model_files=False):
                
        self.resolution = resolution
        self.weights_file_root = Path(weights_file_root) if weights_file_root is not None else Path(__file__).parent
        
        nn.initialize(data_format=data_format)
        tf = nn.tf
        
        model_name = f'{name}_{resolution}'
        self.model_filename_list = []
        
        with tf.device ('/CPU:0'):
            #Place holders on CPU
            self.input_t  = tf.placeholder (nn.floatx, nn.get4Dshape(resolution,resolution,3) )
            self.target_t = tf.placeholder (nn.floatx, nn.get4Dshape(resolution,resolution,1) )

        # Initializing model classes
        with tf.device ('/CPU:0' if place_model_on_cpu else nn.tf_default_device_name):

            #self.input_t  = tf.placeholder (nn.floatx, nn.get4Dshape(resolution,resolution,3) )
            #self.target_t = tf.placeholder (nn.floatx, nn.get4Dshape(resolution,resolution,1) )

            self.xseg_archi = nn.XSegArchi(3, 32, 1, name=name)
            self.model_weights = self.xseg_archi.get_weights()
            if training:
                if optimizer is None:
                    raise ValueError("Optimizer should be provided for training mode.")                
                self.opt = optimizer              
                self.opt.initialize_variables (self.model_weights, vars_on_cpu=place_model_on_cpu)                    
                self.model_filename_list += [ [self.opt, f'{model_name}_opt.npy' ] ]
                
        
        self.model_filename_list += [ [self.xseg_archi, f'{model_name}.npy'] ]

        if not training:
            with tf.device ('/CPU:0' if run_on_cpu else nn.tf_default_device_name):
                _, pred = self.xseg_archi(self.input_t) 
            def net_run(input_np):
                
                return nn.tf_sess.run ( [pred], feed_dict={self.input_t :input_np})[0]
            self.net_run = net_run


        self.initialized = True
        # Loading/initializing all models/optimizers weights
        for model, filename in self.model_filename_list:
            if load_weights is True:
                model_file_path = self.weights_file_root / filename
                load_weights_succeed = model.load_weights(f"{model_file_path}" )
                if load_weights_succeed is False:
                    if raise_on_no_model_files:
                        raise Exception(f'load model data {model_file_path} fail')
                    if not training:
                        self.initialized = False
                        break

            if load_weights is False:
                model.init_weights()
        
    def get_resolution(self):
        return self.resolution
        
    def flow(self, x, pretrain=False):
        return self.xseg_archi(x, pretrain=pretrain)

    def get_weights(self):
        return self.model_weights

    def save_weights(self):
        for model, filename in io.progress_bar_generator(self.model_filename_list, "Saving", leave=False):
            model.save_weights( self.weights_file_root / filename )

    def extract (self, input_image):
        if not self.initialized:
            return 0.5*np.ones ( (self.resolution, self.resolution, 1), nn.floatx.as_numpy_dtype )
        
        resolution=self.get_resolution()
        h,w,c=input_image.shape
        if h!=resolution:
            input_image=cv2.resize(input_image,(resolution,resolution)) 

        input_shape_len = len(input_image.shape)            
        if input_shape_len == 3:
            input_image = input_image[None,...]

        result = np.clip ( self.net_run(input_image), 0, 1.0 )
        result[result < 0.1] = 0 #get rid of noise

        if input_shape_len == 3:
            result = result[0]
        if h!=resolution:
            result=cv2.resize(result,(h,h)) 
        return result

    def export_onnx(self):
        print("bengin export xsegnet onnx ")