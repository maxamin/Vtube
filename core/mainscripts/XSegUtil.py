import json,time
import shutil
import traceback
from pathlib import Path

import numpy as np

from core import pathex
from core.cv2ex import *
from core.interact import interact as io
from core.leras import nn
from core.DFLIMG import *
from facelib import XSegNet, LandmarksProcessor, FaceType
import pickle

def apply_xseg(input_path, model_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} not found. Please ensure it exists.')
    if not model_path.exists():
        raise ValueError(f'{model_path} not found. Please ensure it exists.') 

    nn.close_session()
    io.log_info(f'从{model_path}目录创建XSegNet')
    xseg = XSegNet(name='XSeg',load_weights=True,weights_file_root=model_path,data_format=nn.data_format,raise_on_no_model_files=True)
    xseg_res = xseg.get_resolution()
    
    io.log_info(f'向{input_path} 目录写入分割遮罩')
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath} is not a DFLIMG')
            continue
        
        img = cv2_imread(filepath).astype(np.float32) / 255.0
        h,w,c = img.shape
        
        if w != xseg_res:
            img = cv2.resize( img, (xseg_res,xseg_res), interpolation=cv2.INTER_LANCZOS4 ) 
        if len(img.shape) == 2: img = img[...,None]   
        mask = xseg.extract(img)
        mask[mask < 0.5]=0
        mask[mask >= 0.5]=1    
        dflimg.set_xseg_mask(mask)
        dflimg.save()
        


def export_xseg_onnx(model_path,output_onnx_path):
    xseg = XSegNet(name='XSeg',load_weights=True,weights_file_root=model_path,data_format=nn.data_format,raise_on_no_model_files=True)
    xseg_res = xseg.get_resolution()
    output_graph_def = tf.graph_util.convert_variables_to_constants( nn.tf_sess,tf.get_default_graph().as_graph_def(),['landmarks'])
    import tf2onnx
    with tf.device("/CPU:0"):
        model_proto, _ = tf2onnx.convert._convert_common(
            output_graph_def,name='XSeg',input_names=['in_face:0'],output_names=['landmarks:0'],
            opset=12,output_path=output_onnx_path)
    print("export xseg onnx finish")



def fetch_xseg(input_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} not found. Please ensure it exists.')
    
    output_path = input_path.parent / (input_path.name + '_xseg')
    output_path.mkdir(exist_ok=True, parents=True)
    
    io.log_info(f'Copying faces containing XSeg polygons to {output_path.name}/ folder.')
    
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    
    files_copied = []
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath} is not a DFLIMG')
            continue
        
        ie_polys = dflimg.get_seg_ie_polys()

        if ie_polys.has_polys():
            files_copied.append(filepath)
            shutil.copy ( str(filepath), str(output_path / filepath.name) )
    
    io.log_info(f'Files copied: {len(files_copied)}')
    
    #is_delete = io.input_bool (f"\r\nDelete original files?", True)
    #if is_delete:
    #    for filepath in files_copied:
    #        Path(filepath).unlink()
            
    
def remove_xseg(input_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} not found. Please ensure it exists.')
    
    io.log_info(f'Processing folder {input_path}')
    io.log_info('!!! WARNING : APPLIED XSEG MASKS WILL BE REMOVED FROM THE FRAMES !!!')
    io.log_info('!!! WARNING : APPLIED XSEG MASKS WILL BE REMOVED FROM THE FRAMES !!!')
    io.log_info('!!! WARNING : APPLIED XSEG MASKS WILL BE REMOVED FROM THE FRAMES !!!')
    io.input_str('Press enter to continue.')
                               
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    files_processed = 0
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath} is not a DFLIMG')
            continue
        
        if dflimg.has_xseg_mask():
            dflimg.set_xseg_mask(None)
            dflimg.save()
            files_processed += 1
    io.log_info(f'Files processed: {files_processed}')
    
def remove_xseg_labels(input_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} not found. Please ensure it exists.')
    
    io.log_info(f'Processing folder {input_path}')
    io.log_info('!!! WARNING : LABELED XSEG POLYGONS WILL BE REMOVED FROM THE FRAMES !!!')
    io.log_info('!!! WARNING : LABELED XSEG POLYGONS WILL BE REMOVED FROM THE FRAMES !!!')
    io.log_info('!!! WARNING : LABELED XSEG POLYGONS WILL BE REMOVED FROM THE FRAMES !!!')
    io.input_str('Press enter to continue.')
    
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    files_processed = 0
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath} is not a DFLIMG')
            continue

        if dflimg.has_seg_ie_polys():
            dflimg.set_seg_ie_polys(None)
            dflimg.save()            
            files_processed += 1
            
    io.log_info(f'Files processed: {files_processed}')