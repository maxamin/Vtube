#coding=utf-8
# cython:language_level=3
from enum import IntEnum
from pathlib import Path

import cv2,threading,time,queue
import numpy as np
from core.joblib import SubprocessGenerator
from core.cv2ex import *
from facelib import LandmarksProcessor
from facelib.FaceType import FaceType
from core import imagelib,pathex
from core.imagelib import SegIEPolys
from core.mplib.MPSharedList import MPSharedList
from vtmodel.Sample import SampleType,Sample
from vtmodel.SampleProcessor import *
from core.samplelib.SampleLoader import SampleLoader
from core.DFLIMG import *

class FaceSampleGenerator:
    def __init__(self,data_path, *args, **kwargs):
        self.initialized = False
        self.data_path=data_path
        self.prefetch_sample_queue=queue.Queue(maxsize=3)
        self.shuffle_sample_idx=0;
        self.shuffle_sample_idx_list=[];
        self.sample_len=0;
        self.face_samples_list=None
        self.src_face_samples_list=None
        self.name=kwargs.get("name","smplelib")
        self.output_sample_types=kwargs.get("output_sample_types",[])
        self.sample_process_options=kwargs.get("sample_process_options",None)
        self.batch_size=kwargs.get("batch_size",4)
        self.subdirs=kwargs.get("subdirs",False)
        
        self.initialized = True 
        self.epoch=1;
        self.sample_mode="random"
        self.hloss_sample_idx=0;
        self.hloss_sample_idx_list=[];

        self.load_samples(self.data_path,self.subdirs)

   
    def is_initialized(self):
        return self.initialized


    def load_face_samples(self):        
        self.face_samples_list=[]
        print(f"[S]开始加载训练素材{self.name}")
        img_count=len(self.face_samples_pathstr_list)
        idx=0;
        for filename in self.face_samples_pathstr_list:
            idx=idx+1;
            print(f"[P]加载素材中{idx}/{img_count}")
            dflimg = DFLIMG.load (Path(filename))
            if dflimg is None or not dflimg.has_data():
                print (f"load sample{self.name}: {filename} is not a dfl image file.will ignore this image")
                data = None
            else:
                sample=Sample(filename=filename,sample_type=SampleType.FACE, shape=dflimg.get_shape(), landmarks=dflimg.get_landmarks(),
                              seg_ie_polys=dflimg.get_seg_ie_polys(), xseg_mask_compressed=dflimg.get_xseg_mask_compressed(),
                              eyebrows_expand_mod=dflimg.get_eyebrows_expand_mod()  )
                self.face_samples_list.append(sample)

    def load_samples(self,sample_data_path,subdirs): 
        self.face_samples_pathstr_list=pathex.get_image_paths( sample_data_path, subdirs=self.subdirs);
        #self.load_face_samples( )
        self.face_samples_list=SampleLoader.load_face_samples(self.face_samples_pathstr_list,sample_name=self.name)
        self.sample_len=len(self.face_samples_list)
        self.shuffle_sample_idx_list=[*range(self.sample_len)]
        self.sample_loss_list=np.ones_like(self.shuffle_sample_idx_list,dtype=float)
        np.random.shuffle(self.shuffle_sample_idx_list)

        th=threading.Thread(target=self.prefetch_samples_thread_exec)
        th.isDaemo=True;
        th.start()

    def get_batch_idx(self):
        if self.sample_mode=="random":
            return self.get_batch_indexes_from_shuffle_idx_list()
        else:
            return self.get_batch_indexes_from_high_loss_idx_list()

         ###获取本批次的素材序号
    def get_batch_indexes_from_shuffle_idx_list(self): 
        start_idx=self.shuffle_sample_idx;
        end_idx=start_idx+self.batch_size; 
        if(end_idx>=self.sample_len):
            start_idx=self.sample_len-self.batch_size;
            end_idx=self.sample_len;
        else:
            self.shuffle_sample_idx=end_idx;
        batch_sample_index_nums=self.shuffle_sample_idx_list[start_idx:end_idx]
        if end_idx==self.sample_len:   #训练完一个epoch
            np.random.shuffle(self.shuffle_sample_idx_list)
            self.shuffle_sample_idx=0;
            self.epoch+=1;
            if self.epoch % 3==0:
                self.sample_mode="h_loss"
                self.hloss_sample_idx=0
                self.hloss_sample_idx_list= sorted(range(len(self.sample_loss_list)), key=lambda k: self.sample_loss_list[k], reverse=True)
                self.hloss_sample_idx_list=self.hloss_sample_idx_list[0:self.sample_len//2]
                #print(self.sample_loss_list)
                #print(self.hloss_sample_idx_list)
                #print("go to high loss sample train")
        return batch_sample_index_nums

    def get_batch_indexes_from_high_loss_idx_list(self):
        start_idx=self.hloss_sample_idx;
        end_idx=start_idx+self.batch_size; 
        hloss_sample_len=len(self.hloss_sample_idx_list)
        if(end_idx>=hloss_sample_len):
            start_idx=max(hloss_sample_len-self.batch_size,0);
            end_idx=hloss_sample_len;
        else:
            self.hloss_sample_idx=end_idx;
        
        if end_idx==hloss_sample_len: 
            self.sample_mode="random"
            #print("back to random train")
        batch_sample_index_nums=self.shuffle_sample_idx_list[start_idx:end_idx]
        return batch_sample_index_nums
    
    def prefetch_samples_thread_exec(self):   #持续提取处理好的样本数据存入队列
        while(True):
            batch_sample=self.get_batch_sample_data()
            self.prefetch_sample_queue.put(batch_sample)
            #print("put sample")
           

    def get_batch_sample_data(self,bs=4):        
            batches=None;
            batch_sample_index_nums=self.get_batch_idx()
            #print(batch_sample_index_nums)
            for batch_index in range(bs):
                if batch_index==len(batch_sample_index_nums):
                    break;
                sample_idx=batch_sample_index_nums[batch_index]
                sample=self.face_samples_list[sample_idx]
                x,= SampleProcessor.process ([sample], self.sample_process_options, self.output_sample_types, debug=False, ct_sample=None)
                type_count=len(x)
                if batches is None:
                    batches = [ [] for _ in range(type_count+1) ]
                for type_idx in range(type_count):
                    batches[type_idx].append ( x[type_idx] )
                sample_info=[sample_idx,sample.filename]
                batches[type_count].append(sample_info)
            #return [ np.array(batch) for batch in batches]
            return batches
             
    def generate_next(self):
        return self.prefetch_sample_queue.get()
  
    def record_sample_loss(self,losses,info):
        if len(losses)!=len(info):
            print("损失值和输入样本数不一致")
            return
        for bi in range(len(losses)):
            idx,name=info[bi]
            loss=losses[bi]
            #print(f"[序号:{idx}],{name}:[{loss}]")
            self.sample_loss_list[idx]=loss
        #print(self.name,self.sample_loss_list)

if __name__ == '__main__':
    training_data_src_path="F:\TrainFaceLib\dy憨憨铁"
    training_data_dst_path="F:\TrainFaceLib\开发"
    random_warp=False;ct_mode="none";random_hsv_power=False;face_type=FaceType.WHOLE_FACE;data_format="NHWC";resolution=320;batch_size=1;

    generator_list=[FaceSampleGenerator(data_path=training_data_src_path, batch_size=batch_size,
                        sample_process_options=SampleProcessor.Options(scale_range=[-0.15, 0.15], random_flip=False),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':random_warp, 'transform':True,'channel_type' : SampleProcessor.ChannelType.BGR, 'ct_mode': ct_mode, 'random_hsv_shift_amount' : random_hsv_power,  'face_type':face_type, 'data_format':data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'ct_mode': ct_mode,                           'face_type':face_type, 'data_format':data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE, 'face_type':face_type, 'data_format':data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.EYES_MOUTH, 'face_type':face_type, 'data_format':data_format, 'resolution': resolution},
                                              ],
                        uniform_yaw_distribution=False),
                FaceSampleGenerator(data_path=training_data_dst_path, batch_size=batch_size,
                    sample_process_options=SampleProcessor.Options(scale_range=[-0.15, 0.15], random_flip=False),
                    output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':random_warp, 'transform':True,'channel_type' : SampleProcessor.ChannelType.BGR, 'ct_mode': ct_mode, 'random_hsv_shift_amount' : random_hsv_power,  'face_type':face_type, 'data_format':data_format, 'resolution': resolution},
                                            {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'ct_mode': ct_mode,                           'face_type':face_type, 'data_format':data_format, 'resolution': resolution},
                                            {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE, 'face_type':face_type, 'data_format':data_format, 'resolution': resolution},
                                            {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.EYES_MOUTH, 'face_type':face_type, 'data_format':data_format, 'resolution': resolution},
                                            ],
                    uniform_yaw_distribution=False)]

    import time
    time.sleep(1)
    for i in range(200):
        #samples=next(generator_dst.generate_shuffle_next(bs=3))
        start=time.time()
        
        sample_src=generator_list[0].generate_next()
        sample_dst=generator_list[1].generate_next()
        stack_sample=np.hstack((sample_src[0][0],sample_src[1][0],sample_dst[0][0],sample_dst[1][0])) 

        end=time.time()
        print("time:",round(end-start,3))
        cv2.imshow("stack_sample",stack_sample)
        #cv2.imshow("dst",dst_sample)
        #mask=Sample.get_xseg_mask()
        #if mask is not None: cv2.imshow("mask",mask)
        cv2.waitKey(3000)