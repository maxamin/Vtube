#coding=utf-8
# cython:language_level=3

from core.leras import nn
from core.interact import interact as io
import time,os,shutil,cv2
import pickle,numpy as np
import core.pathex as pathex
from pathlib import Path
from vtmodel.SampleProcessor  import *  
from vtmodel.SampleLoader  import *  
from vtmodel.Sample import SampleType
from vtmodel.FaceSampleGenerator import FaceSampleGenerator 
from facelib.FaceType import FaceType


class FaceSwapModelSAE:
    def __init__(self,saved_models_dir, kwargs):
        #super().__init__(self, **kwargs)
        self.options={}
        self.model_filename_list=[]
        self.iter=0
        self.loss_history=[]
        #--model archi
        self.saved_models_dir=saved_models_dir
        self.training_data_src_path=kwargs.get("training_data_src_path","")
        self.training_data_dst_path=kwargs.get("training_data_dst_path","")

        self.options["archi"]=kwargs.get("archi","df-ud")
        self.options["resolution"]=kwargs.get("resolution",256)
        self.options["ae_dims"]=kwargs.get("ae_dims",256)
        self.options["e_dims"]=kwargs.get("e_dims",64)
        self.options["d_dims"]=kwargs.get("d_dims",64)
        self.options["d_mask_dims"]=kwargs.get("d_mask_dims",22)

        archi_split = self.options["archi"].split('-')
        if len(archi_split) == 2:
            archi_type, archi_opts = archi_split
        elif len(archi_split) == 1:
            archi_type, archi_opts = archi_split[0], None
        self.options["archi_type"]=archi_type;
        self.options["archi_opts"]=archi_opts;

        #--train options
        self.options["learning_rate"]=kwargs.get("learning_rate",0.00005)
        self.options["masked_training"]=kwargs.get("masked_training",True)
        self.options["gpu_training"]=kwargs.get("gpu_training",True)
        self.options["target_iter"]=kwargs.get("target_iter",0)
        self.options["eyes_mouth_prio"]=kwargs.get("eyes_mouth_prio",True)
        self.options["blur_out_mask"]=kwargs.get("blur_out_mask",False)
        self.options["adabelief"]=kwargs.get("adabelief",False)
        self.options["lr_dropout"]=kwargs.get("lr_dropout",False)
        self.options["clipgrad"]=kwargs.get("clipgrad",True)
        self.options["true_face_power"]=kwargs.get("true_face_power",0)
        self.options["face_style_power"]=kwargs.get("face_style_power",0)
        self.options["bg_style_power"]=kwargs.get("bg_style_power",0)
        self.options["gan_power"]=kwargs.get("gan_power",0)
        self.options["gan_patch_size"]=kwargs.get("gan_patch_size",64)
        self.options["gan_dims"]=kwargs.get("gan_dims",16)

        self.options["ssim_power"]=kwargs.get("ssim_power",10.0)
        self.options["mse_power"]=kwargs.get("mse_power",10.0)
        self.options["emp_power"]=kwargs.get("emp_power",10.0)
        self.options["mask_power"]=kwargs.get("mask_power",10.0)
        self.options["hloss_cycle"]=kwargs.get("hloss_cycle",3)
        #---sample-process-options
        self.options["src_data_subdir"]=kwargs.get("src_data_subdir",False)
        self.options["dst_data_subdir"]=kwargs.get("dst_data_subdir",False)
        self.options["batch_size"]=kwargs.get("batch_size",4)
        self.options["random_flip_src"]=kwargs.get("random_flip_src",True)
        self.options["random_flip_dst"]=kwargs.get("random_flip_dst",True)
        self.options["uniform_yaw"]=kwargs.get("uniform_yaw",True)
        self.options["random_warp"]=kwargs.get("random_warp",False)
        self.options["random_hsv_shift"]=kwargs.get("random_hsv_shift",False)
        self.options["random_hsv_power"]=0.01 if self.options["random_hsv_shift"] else 0;
        self.options["ct_mode"]=kwargs.get("ct_mode","none") 

        self.config_full_path=self.get_strpath_storage_for_file("_data.dat")        
        self.model_data_path = Path( self.config_full_path )

        #-- init nn and tensorflow 
        nn.initialize_main_env()
        self.device_config = nn.getCurrentDeviceConfig()
        devices = self.device_config.devices

        self.model_data_format = "NCHW"
        #self.model_data_format = "NCHW" if len(devices) != 0  else "NHWC"
        
        nn.initialize(data_format=self.model_data_format)
        self.models_opt_device = nn.tf_default_device_name
        self.tf = nn.tf

        #print("devices:",len(devices))
        #print("model_data_format:",self.model_data_format)
        #print("models_opt_device:",self.models_opt_device)

    def load_archi_config_from_dat(self,model_path=None):
        if model_path is None:
            model_path=self.saved_models_dir
        if model_path is None:
            model_path=self.saved_models_dir
         #--step01:load dat file
        

        print(f"加载配置: {self.config_full_path}")
        if self.config_full_path is None:
            print(f"错误：未发现模型配置文件_data.dat")
            return

        #--- load nn archi and dims
        
        model_data = pickle.loads ( self.model_data_path.read_bytes() )
        self.iter =self.options["iter"]= model_data.get('iter',0)
        options=model_data.get("options",None)
        
        archi=self.options["archi"]=options.get('archi',"liae-ud")
        resolution =self.options['resolution']= options['resolution']
        ae_dims = self.options['ae_dims']=options['ae_dims']
        e_dims = self.options['e_dims']=options['e_dims']
        d_dims = self.options['d_dims']=options['d_dims']
        d_mask_dims = self.options['d_mask_dims']=options['d_mask_dims']
        adabelief = self.options['adabelief']=options['adabelief']        

        archi_split = archi.split('-')
        if len(archi_split) == 2:
            archi_type, archi_opts = archi_split
        elif len(archi_split) == 1:
            archi_type, archi_opts = archi_split[0], None
        self.options["archi_type"]=archi_type;
        self.options["archi_opts"]=archi_opts;


        #----- 创建模型的各个架构（DF: encoder,inter,decoder_src,decoder_dst) （LIAE: encoder,inter_AB,inter_B,decoder)
    def create_model_nn_archi(self):

        #print(self.options["archi"],self.options["archi_type"])
        model_archi = nn.DeepFakeArchi(self.options["resolution"], use_fp16=False, opts=self.options.get("archi_opts",""))
        archi=self.options["archi"]
        resolution =self.options['resolution']
        ae_dims = self.options['ae_dims']
        e_dims = self.options['e_dims']
        d_dims = self.options['d_dims']
        d_mask_dims = self.options['d_mask_dims']

        #--- 创建模型神经网络的各个分结构（Encoder，Inter，Decoder），生成参数文件列表
        with self.tf.device (self.models_opt_device):
            if 'df' in self.options["archi_type"]:
                self.encoder = model_archi.Encoder(in_ch=3, e_ch=e_dims, name='encoder')
                encoder_out_ch = self.encoder.get_out_ch()*self.encoder.get_out_res(resolution)**2
                self.inter = model_archi.Inter (in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims, name='inter')
                inter_out_ch = self.inter.get_out_ch()
                self.decoder_src = model_archi.Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder_src')
                self.decoder_dst = model_archi.Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder_dst')

                self.model_filename_list += [ [self.encoder,'encoder.npy'],[self.inter,'inter.npy'],
                                              [self.decoder_src, 'decoder_src.npy'],[self.decoder_dst, 'decoder_dst.npy']  ]
                if self.options['true_face_power'] != 0:
                        self.trueface_code_discriminator = nn.CodeDiscriminator(ae_dims, code_res=self.inter.get_out_res(), name='dis' )
                        self.model_filename_list += [ [self.trueface_code_discriminator, 'code_discriminator.npy'] ]

            elif 'liae' in self.options["archi_type"]:
                self.encoder = model_archi.Encoder(in_ch=3, e_ch=e_dims, name='encoder')
                encoder_out_ch = self.encoder.get_out_ch()*self.encoder.get_out_res(resolution)**2

                self.inter_AB = model_archi.Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims*2, name='inter_AB')
                self.inter_B  = model_archi.Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims*2, name='inter_B')

                inter_out_ch = self.inter_AB.get_out_ch()
                inters_out_ch = inter_out_ch*2
                self.decoder = model_archi.Decoder(in_ch=inters_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder')

                self.model_filename_list += [ [self.encoder,  'encoder.npy'],
                                              [self.inter_AB, 'inter_AB.npy'],
                                              [self.inter_B , 'inter_B.npy'],
                                              [self.decoder , 'decoder.npy'] ]

            if self.options['gan_power'] != 0:
                    self.D_src = nn.UNetPatchDiscriminator(patch_size=self.options['gan_patch_size'], in_ch=3, base_ch=self.options['gan_dims'], name="D_src")
                    self.model_filename_list += [ [self.D_src, 'GAN.npy'] ]
                     
            if 'df' in self.options["archi_type"]:
                self.src_dst_saveable_weights = self.encoder.get_weights() + self.inter.get_weights() + self.decoder_src.get_weights() + self.decoder_dst.get_weights()
                self.src_dst_trainable_weights = self.src_dst_saveable_weights
            elif 'liae' in self.options["archi_type"]:
                self.src_dst_saveable_weights = self.encoder.get_weights() + self.inter_AB.get_weights() + self.inter_B.get_weights() + self.decoder.get_weights()
                if self.options["random_warp"]:
                    self.src_dst_trainable_weights = self.src_dst_saveable_weights
                else:
                    self.src_dst_trainable_weights = self.encoder.get_weights() + self.inter_B.get_weights() + self.decoder.get_weights()

                #--- optimizer (创建优化器）
            lr=self.options.get("learning_rate",0.00005)
            clipnorm = 1.0 if self.options['clipgrad'] else 0.0
            lr_dropout = 0.3 if self.options['lr_dropout'] else 1.0
            lr_cos = 500 if self.options['lr_dropout'] else 1.0

            OptimizerClass = nn.AdaBelief if self.options["adabelief"] else nn.RMSprop

            self.src_dst_optimizer = OptimizerClass(lr=lr, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name='src_dst_opt')
            self.src_dst_optimizer.initialize_variables (self.src_dst_saveable_weights, vars_on_cpu=False, lr_dropout_on_cpu=self.options['lr_dropout']=='cpu')
            self.model_filename_list += [ (self.src_dst_optimizer, 'src_dst_opt.npy') ]

            if self.options['true_face_power'] != 0:
                optimizer_vars_on_cpu=True;
                self.true_face_D_code_optimizer = OptimizerClass(lr=lr, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name='D_code_opt')
                self.true_face_D_code_optimizer.initialize_variables ( self.trueface_code_discriminator.get_weights(), vars_on_cpu=optimizer_vars_on_cpu, lr_dropout_on_cpu=self.options['lr_dropout']=='cpu')
                self.model_filename_list += [ (self.true_face_D_code_optimizer, 'D_code_opt.npy') ]

            if self.options['gan_power'] != 0:
                gan_optimizer_vars_on_cpu=True;
                self.gan_patch_D_code_optimizer = OptimizerClass(lr=lr, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name='GAN_opt')
                self.gan_patch_D_code_optimizer.initialize_variables ( self.D_src.get_weights(), vars_on_cpu=gan_optimizer_vars_on_cpu, lr_dropout_on_cpu=self.options['lr_dropout']=='cpu')#+self.D_src_x2.get_weights()
                self.model_filename_list += [ (self.gan_patch_D_code_optimizer, 'GAN_opt.npy') ]

        #self.tf.global_variables_initializer()

    #--- 新建模型文件（create model files)
    def load_model_from_dir(self,model_path=None):
        
        self.load_archi_config_from_dat()
        self.create_model_nn_archi()
                
        # ----- Loading  weights files（加载权重和优化器文件）
        for model, filename in io.progress_bar_generator(self.model_filename_list, "[S]加载权重文件(Loading models)"):
            full_path=self.get_strpath_storage_for_file(filename) 
            do_init = not model.load_weights(full_path )
            filename=os.path.basename(full_path)
            print(f"加载: {filename}")
        #print(f"[S]模型文件加载完成")


        #---- 优化器和损失函数的定义
    def define_tf_output_and_loss(self,define_loss=True):

        tf = nn.tf
        resolution=self.options["resolution"]
        bgr_shape = self.bgr_shape = nn.get4Dshape(resolution,resolution,3)
        mask_shape = nn.get4Dshape(resolution,resolution,1)
        archi_type=self.options["archi_type"]

        with self.tf.device ('/CPU:0'):  #输入占位变量
             
            self.warped_src = tf.placeholder (nn.floatx, bgr_shape, name='warped_src')
            self.warped_dst = tf.placeholder (nn.floatx, bgr_shape, name='warped_dst')
            self.target_src = tf.placeholder (nn.floatx, bgr_shape, name='target_src')
            self.target_dst = tf.placeholder (nn.floatx, bgr_shape, name='target_dst')

            self.target_srcm    = tf.placeholder (nn.floatx, mask_shape, name='target_srcm')
            self.target_srcm_em = tf.placeholder (nn.floatx, mask_shape, name='target_srcm_em')
            self.target_dstm    = tf.placeholder (nn.floatx, mask_shape, name='target_dstm')
            self.target_dstm_em = tf.placeholder (nn.floatx, mask_shape, name='target_dstm_em')
        
        with self.tf.device(nn.tf_default_device_name):  #转GPU
            gpu_warped_src      = self.warped_src 
            gpu_warped_dst      = self.warped_dst 
            gpu_target_src      = self.target_src 
            gpu_target_dst      = self.target_dst 
            gpu_target_srcm     = self.target_srcm
            gpu_target_srcm_em  = self.target_srcm_em
            gpu_target_dstm     = self.target_dstm
            gpu_target_dstm_em  = self.target_dstm_em

            gpu_target_srcm_anti = 1-gpu_target_srcm
            gpu_target_dstm_anti = 1-gpu_target_dstm

            if 'df' in archi_type:
                gpu_src_code     = self.inter(self.encoder(gpu_warped_src))
                gpu_dst_code     = self.inter(self.encoder(gpu_warped_dst))
                gpu_pred_src_src, gpu_pred_src_srcm = self.decoder_src(gpu_src_code)
                gpu_pred_dst_dst, gpu_pred_dst_dstm = self.decoder_dst(gpu_dst_code)
                gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(gpu_dst_code)
                gpu_pred_src_dst_no_code_grad, _ = self.decoder_src(tf.stop_gradient(gpu_dst_code))

            elif 'liae' in archi_type:
                    gpu_src_code = self.encoder (gpu_warped_src)
                    gpu_src_inter_AB_code = self.inter_AB (gpu_src_code)
                    gpu_src_code = tf.concat([gpu_src_inter_AB_code,gpu_src_inter_AB_code], nn.conv2d_ch_axis  )
                    gpu_dst_code = self.encoder (gpu_warped_dst)
                    gpu_dst_inter_B_code = self.inter_B (gpu_dst_code)
                    gpu_dst_inter_AB_code = self.inter_AB (gpu_dst_code)
                    gpu_dst_code = tf.concat([gpu_dst_inter_B_code,gpu_dst_inter_AB_code], nn.conv2d_ch_axis )
                    gpu_src_dst_code = tf.concat([gpu_dst_inter_AB_code,gpu_dst_inter_AB_code], nn.conv2d_ch_axis )

                    gpu_pred_src_src, gpu_pred_src_srcm = self.decoder(gpu_src_code)
                    gpu_pred_dst_dst, gpu_pred_dst_dstm = self.decoder(gpu_dst_code)
                    gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder(gpu_src_dst_code)
                    gpu_pred_src_dst_no_code_grad, _ = self.decoder(tf.stop_gradient(gpu_src_dst_code))

            #--- src原始遮罩,高斯模糊 
            gpu_target_srcm_blur = nn.gaussian_blur(gpu_target_srcm,  max(1, resolution // 32) )
            gpu_target_srcm_blur = tf.clip_by_value(gpu_target_srcm_blur, 0, 0.5) * 2
            gpu_target_srcm_anti_blur = 1.0-gpu_target_srcm_blur

            #--- dst原始遮罩,高斯模糊 
            gpu_target_dstm_blur = nn.gaussian_blur(gpu_target_dstm,  max(1, resolution // 32) )
            gpu_target_dstm_blur = tf.clip_by_value(gpu_target_dstm_blur, 0, 0.5) * 2

            #-- src，dst预测遮罩相乘，高斯模糊
            gpu_style_mask_blur = nn.gaussian_blur(gpu_pred_src_dstm*gpu_pred_dst_dstm,  max(1, resolution // 32) )
            gpu_style_mask_blur = tf.stop_gradient(tf.clip_by_value(gpu_target_srcm_blur, 0, 1.0))
            gpu_style_mask_anti_blur = 1.0 - gpu_style_mask_blur


            gpu_target_src_anti_masked = gpu_target_src*gpu_target_srcm_anti_blur
            gpu_pred_src_src_anti_masked = gpu_pred_src_src*gpu_target_srcm_anti_blur

            masked_training=self.options["masked_training"]
            eyes_mouth_prio=self.options["eyes_mouth_prio"]
            gpu_target_src_masked_face  = gpu_target_src*gpu_target_srcm_blur if masked_training else gpu_target_src
            gpu_target_dst_masked_face  = gpu_target_dst*gpu_target_dstm_blur if masked_training else gpu_target_dst
            gpu_pred_src_src_masked_face = gpu_pred_src_src*gpu_target_srcm_blur if masked_training else gpu_pred_src_src
            gpu_pred_dst_dst_masked_face = gpu_pred_dst_dst*gpu_target_dstm_blur if masked_training else gpu_pred_dst_dst


            ssim_power=self.options["ssim_power"]
            half_ssim_power=ssim_power/2;
            mse_power=self.options["mse_power"]
            emp_power=self.options["emp_power"]
            mask_power=self.options["mask_power"]
            #--定义常规训练src损失函数( ssim,pixel_mse,eye_mouth,mask)
            if resolution < 256:
                gpu_src_loss =  tf.reduce_mean ( ssim_power*nn.dssim(gpu_target_src_masked_face, gpu_pred_src_src_masked_face, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
            else:
                gpu_src_loss =  tf.reduce_mean ( half_ssim_power*nn.dssim(gpu_target_src_masked_face, gpu_pred_src_src_masked_face, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
                gpu_src_loss += tf.reduce_mean ( half_ssim_power*nn.dssim(gpu_target_src_masked_face, gpu_pred_src_src_masked_face, max_val=1.0, filter_size=int(resolution/23.2)), axis=[1])
            gpu_src_loss += tf.reduce_mean ( mse_power *tf.square ( gpu_target_src_masked_face - gpu_pred_src_src_masked_face ), axis=[1,2,3])
            if eyes_mouth_prio:
                gpu_src_loss += tf.reduce_mean ( emp_power*tf.abs ( gpu_target_src*gpu_target_srcm_em - gpu_pred_src_src*gpu_target_srcm_em ), axis=[1,2,3])
            gpu_src_loss += tf.reduce_mean ( mask_power*tf.square( gpu_target_srcm - gpu_pred_src_srcm ),axis=[1,2,3] )

            #--定义常规训练dst损失函数( ssim,pixel_mse,eye_mouth,mask)
            if resolution < 256:
                gpu_dst_loss = tf.reduce_mean ( ssim_power*nn.dssim(gpu_target_dst_masked_face, gpu_pred_dst_dst_masked_face, max_val=1.0, filter_size=int(resolution/11.6) ), axis=[1])
            else:
                gpu_dst_loss = tf.reduce_mean ( half_ssim_power*nn.dssim(gpu_target_dst_masked_face, gpu_pred_dst_dst_masked_face, max_val=1.0, filter_size=int(resolution/11.6) ), axis=[1])
                gpu_dst_loss += tf.reduce_mean ( half_ssim_power*nn.dssim(gpu_target_dst_masked_face, gpu_pred_dst_dst_masked_face, max_val=1.0, filter_size=int(resolution/23.2) ), axis=[1])
            gpu_dst_loss += tf.reduce_mean ( mse_power*tf.square(  gpu_target_dst_masked_face- gpu_pred_dst_dst_masked_face ), axis=[1,2,3])
            if eyes_mouth_prio:
                gpu_dst_loss += tf.reduce_mean ( emp_power*tf.abs ( gpu_target_dst*gpu_target_dstm_em - gpu_pred_dst_dst*gpu_target_dstm_em ), axis=[1,2,3])
            gpu_dst_loss += tf.reduce_mean ( mask_power*tf.square( gpu_target_dstm - gpu_pred_dst_dstm ),axis=[1,2,3] )
             
            #--总体损失函数
            gpu_Gross_loss = gpu_src_loss + gpu_dst_loss


            #--- 判别器
            def DLoss(labels,logits):
                        return tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=[1,2,3])
            
            #--- True_Face 判别器计算定义，损失函数定义
            if self.options['true_face_power'] != 0:
                    gpu_src_code_d = self.trueface_code_discriminator( gpu_src_code )
                    gpu_src_code_d_ones  = tf.ones_like (gpu_src_code_d)
                    gpu_src_code_d_zeros = tf.zeros_like(gpu_src_code_d)
                    gpu_dst_code_d = self.trueface_code_discriminator( gpu_dst_code )
                    gpu_dst_code_d_ones = tf.ones_like(gpu_dst_code_d)

                    gpu_Gross_loss += self.options['true_face_power']*DLoss(gpu_src_code_d_ones, gpu_src_code_d)
                    gpu_trueface_D_code_loss = (DLoss(gpu_dst_code_d_ones , gpu_dst_code_d) + DLoss(gpu_src_code_d_zeros, gpu_src_code_d) ) * 0.5
                    gpu_trueface_D_code_loss_gv = nn.gradients (gpu_trueface_D_code_loss, self.trueface_code_discriminator.get_weights() ) 
                    true_face_loss_gv_op = self.true_face_D_code_optimizer.get_update_op (gpu_trueface_D_code_loss_gv)
            
            #-- GanPatch 算子及损失函数
            if self.options['gan_power'] != 0:
                gpu_pred_src_src_d,  gpu_pred_src_src_d2 = self.D_src(gpu_pred_src_src_masked_face)
                gpu_pred_src_src_d_ones  = tf.ones_like (gpu_pred_src_src_d)
                gpu_pred_src_src_d_zeros = tf.zeros_like(gpu_pred_src_src_d)
                gpu_pred_src_src_d2_ones  = tf.ones_like (gpu_pred_src_src_d2)
                gpu_pred_src_src_d2_zeros = tf.zeros_like(gpu_pred_src_src_d2)

                gpu_target_src_d, gpu_target_src_d2= self.D_src(gpu_target_src_masked_face)
                gpu_target_src_d_ones    = tf.ones_like(gpu_target_src_d)
                gpu_target_src_d2_ones    = tf.ones_like(gpu_target_src_d2)
                gpu_D_src_dst_loss = (DLoss(gpu_target_src_d_ones      , gpu_target_src_d) + \
                                              DLoss(gpu_pred_src_src_d_zeros   , gpu_pred_src_src_d) ) * 0.5 + \
                                             (DLoss(gpu_target_src_d2_ones      , gpu_target_src_d2) + \
                                              DLoss(gpu_pred_src_src_d2_zeros   , gpu_pred_src_src_d2) ) * 0.5
                gpu_D_src_dst_loss_gv=nn.gradients (gpu_D_src_dst_loss, self.D_src.get_weights() )
                gpu_Gross_loss += self.options['gan_power']*(DLoss(gpu_pred_src_src_d_ones, gpu_pred_src_src_d)  + \
                                                 DLoss(gpu_pred_src_src_d2_ones, gpu_pred_src_src_d2))
                src_D_src_dst_loss_gv_op = self.gan_patch_D_code_optimizer.get_update_op (gpu_D_src_dst_loss_gv)

            gpu_G_loss_gv=nn.gradients ( gpu_Gross_loss, self.src_dst_trainable_weights )
            src_dst_loss_gv_op = self.src_dst_optimizer.get_update_op (gpu_G_loss_gv)




        #定义src-dst训练过程
        def src_dst_train(warped_src, target_src, target_srcm, target_srcm_em,  warped_dst, target_dst, target_dstm, target_dstm_em, ):
            s, d = nn.tf_sess.run ( [ gpu_src_loss, gpu_dst_loss, src_dst_loss_gv_op],
                                        feed_dict={self.warped_src :warped_src,
                                                    self.target_src :target_src,
                                                    self.target_srcm:target_srcm,
                                                    self.target_srcm_em:target_srcm_em,
                                                    self.warped_dst :warped_dst,
                                                    self.target_dst :target_dst,
                                                    self.target_dstm:target_dstm,
                                                    self.target_dstm_em:target_dstm_em,
                                                    })[:2]
            return s, d
        self.src_dst_train = src_dst_train

        #定义trueface训练过程
        if self.options['true_face_power'] != 0:
                def trueface_Discriminator_train(warped_src, warped_dst):
                    nn.tf_sess.run ([true_face_loss_gv_op], feed_dict={self.warped_src: warped_src, self.warped_dst: warped_dst})
                self.trueface_Discriminator_train = trueface_Discriminator_train

        #定义gan_patch训练过程
        if self.options['gan_power'] != 0:
            def gan_patch_src_dst_train(warped_src, target_src, target_srcm, target_srcm_em,  \
                                warped_dst, target_dst, target_dstm, target_dstm_em, ):
                nn.tf_sess.run ([src_D_src_dst_loss_gv_op], feed_dict={self.warped_src :warped_src,
                                                                        self.target_src :target_src,
                                                                        self.target_srcm:target_srcm,
                                                                        self.target_srcm_em:target_srcm_em,
                                                                        self.warped_dst :warped_dst,
                                                                        self.target_dst :target_dst,
                                                                        self.target_dstm:target_dstm,
                                                                        self.target_dstm_em:target_dstm_em})
            self.gan_patch_src_dst_train = gan_patch_src_dst_train


        def AE_view(warped_src, warped_dst):
            with self.tf.device(f'/CPU:0'):
                pred_src_src  = gpu_pred_src_src
                pred_dst_dst  = gpu_pred_dst_dst
                pred_src_dst  = gpu_pred_src_dst
                pred_src_srcm = gpu_pred_src_srcm
                pred_dst_dstm = gpu_pred_dst_dstm
                pred_src_dstm = gpu_pred_src_dstm
            return nn.tf_sess.run ( [pred_src_src, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm],
                                        feed_dict={self.warped_src:warped_src,
                                                self.warped_dst:warped_dst})
        self.AE_view = AE_view

        def Test_view(warped_dst):
            with self.tf.device(f'/CPU:0'):
                pred_dst_dst  = gpu_pred_dst_dst
                pred_src_dst  = gpu_pred_src_dst
                pred_dst_dstm = gpu_pred_dst_dstm
                pred_src_dstm = gpu_pred_src_dstm
            return nn.tf_sess.run([ pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm],
                                feed_dict={self.warped_dst:warped_dst}  )
        self.Test_view=Test_view


    #----- 加载训练素材
    def load_train_samples(self,training_data_src_path=None,training_data_dst_path=None):
        if training_data_src_path is None:
            training_data_src_path=self.training_data_src_path
        if training_data_dst_path is None:
            training_data_dst_path=self.training_data_dst_path
        face_type=FaceType.WHOLE_FACE
        self.generator_src=FaceSampleGenerator(training_data_src_path,  batch_size=self.options["batch_size"],name="src",subdirs=self.options["src_data_subdir"],
                        sample_process_options=SampleProcessor.Options(scale_range=[-0.15, 0.15], random_flip=False),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':self.options["random_warp"], 'transform':True,'channel_type' : SampleProcessor.ChannelType.BGR, 'ct_mode': self.options["ct_mode"], 'random_hsv_shift_amount' : self.options["random_hsv_power"], 'data_format':nn.data_format, 'resolution': self.options["resolution"]},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'ct_mode': self.options["ct_mode"],                           'face_type':face_type, 'data_format':nn.data_format, 'resolution': self.options["resolution"]},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE, 'face_type':face_type, 'data_format':nn.data_format, 'resolution': self.options["resolution"]},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.EYES_MOUTH, 'face_type':face_type, 'data_format':nn.data_format, 'resolution': self.options["resolution"]},
                                              ],
                        uniform_yaw_distribution=self.options['uniform_yaw']  )

        self.generator_dst=FaceSampleGenerator(training_data_dst_path,batch_size=self.options["batch_size"],name="dst",subdirs=self.options["dst_data_subdir"],
                            sample_process_options=SampleProcessor.Options(scale_range=[-0.15, 0.15], random_flip=False),
            output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':self.options["random_warp"], 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,'face_type':face_type, 'data_format':nn.data_format, 'resolution': self.options["resolution"]},
                                    {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,'face_type':face_type, 'data_format':nn.data_format, 'resolution': self.options["resolution"]},
                                    {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE, 'face_type':face_type, 'data_format':nn.data_format, 'resolution': self.options["resolution"]},
                                    {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.EYES_MOUTH, 'face_type':face_type, 'data_format':nn.data_format, 'resolution': self.options["resolution"]},
                                    ],
            uniform_yaw_distribution=self.options['uniform_yaw']  )
                             

    def generate_next_samples(self):
        sample = []
        sample.append ( self.generator_src.generate_next(options=self.options) )
        sample.append ( self.generator_dst.generate_next(options=self.options) )
        self.last_sample = sample
        return sample


    
    #---- 获取训练预览图片
    def get_previews(self,typeIdx=0):
        (warped_src, target_src, target_srcm, target_srcm_em, info_src)  =self.generator_src.generate_next()
        (warped_dst, target_dst, target_dstm, target_dstm_em, info_dst) = self.generator_dst.generate_next()

        src_loss, dst_loss = self.src_dst_train (warped_src, target_src, target_srcm, target_srcm_em, warped_dst, target_dst, target_dstm, target_dstm_em)
        self.generator_src.record_sample_loss(src_loss,info_src)
        self.generator_dst.record_sample_loss(dst_loss,info_dst)

        S, D, SS, DD, DDM, SD, SDM = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([target_src,target_dst] + self.AE_view (target_src, target_dst) ) ]
        DDM, SDM, = [ np.repeat (x, (3,), -1) for x in [DDM, SDM] ]
        target_srcm, target_dstm = [ nn.to_data_format(x,"NHWC", self.model_data_format) for x in ([target_srcm, target_dstm] )]
        n_samples = min(4, self.options["batch_size"],1000 // self.options["resolution"],len(S) )
        
         
        st = []
        for i in range(n_samples):
            src_file_name=os.path.basename(info_src[i][1])
            loss_src=round(src_loss[i],3)    
            dst_file_name=os.path.basename(info_dst[i][1])
            loss_dst=round(dst_loss[i] ,3)

            Si, SSi, Di, DDi, SDi =S[i].copy(), SS[i].copy(), D[i].copy(), DD[i].copy(), SD[i].copy()
            cv2.putText(Si,src_file_name,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            cv2.putText(SSi,str(loss_src),(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            cv2.putText(Di,str(dst_file_name),(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            cv2.putText(DDi,str(loss_dst),(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            ar=Si, SSi, Di, DDi, SDi 
            merge=np.hstack( ar)
            st.append (merge)
        faces=np.concatenate (st, axis=0 )
               
        st_m = []
        for i in range(n_samples):
            SD_mask = DDM[i]*SDM[i] 
            ar = S[i]*target_srcm[i], SS[i]*target_srcm[i], D[i]*target_dstm[i], DD[i]*DDM[i], SD[i]*SD_mask
            merge=np.concatenate(ar, axis=1) 
            st_m.append (merge)
        faces_mask= np.concatenate (st_m, axis=0 )
        return faces,faces_mask

        #---获取测试预览图
    def get_test_preview(self,test_align_face):
        
        test_align_face=np.transpose(test_align_face, (2,0,1))
        test_align_face=np.expand_dims(test_align_face,axis=0)
        
        pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm=self.Test_view(test_align_face)

        return pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm


    def get_strpath_storage_for_file(self,end_name,model_path=None):
        if model_path is None:
            model_path=self.saved_models_dir
        fullpath=""
        file_paths=pathex.get_file_paths(model_path)
        for filepath in file_paths:
            filepath_name = filepath.name
            if  filepath_name.endswith(end_name):
                fullpath= model_path+"\\"+filepath_name
        if fullpath=="":
            fullpath= model_path+"\\default_SAEHD_"+end_name
        return fullpath

    #--- 获取训练样本集状态
    def get_train_status(self):
        status=""
        if (self.generator_src is not None) and (self.generator_dst is not None) :
            status=f"src train:{self.generator_src.sample_mode},dst train:{self.generator_dst.sample_mode}"
        return status



    #--- 新建模型文件（create model files)
    def create_new_models_files(self,model_path,model_name):
        ae_dims = self.options['ae_dims']
        e_dims = self.options['e_dims']
        d_dims = self.options['d_dims']
        d_mask_dims = self.options['d_mask_dims']
        resolution = self.options['resolution']

        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices
        self.model_data_format = "NCHW" if len(devices) != 0  else "NHWC"
        nn.initialize(data_format=self.model_data_format)
        tf = nn.tf

        archi_split = self.options['archi'].split('-')
        if len(archi_split) == 2:
            archi_type, archi_opts = archi_split
        elif len(archi_split) == 1:
            archi_type, archi_opts = archi_split[0], None
        model_archi = nn.DeepFakeArchi(resolution, use_fp16=False, opts=archi_opts)
        self.models_opt_device = nn.tf_default_device_name


    def empty_run_nn_init(self):
        init = self.tf.global_variables_initializer()
        with self.tf.Session() as session:
            session.run(init) 

    #----单个训练迭代
    def  onTrainOneIter(self):
        (warped_src, target_src, target_srcm, target_srcm_em,info_src)  =self.generator_src.generate_next()
        (warped_dst, target_dst, target_dstm, target_dstm_em,info_dst) = self.generator_dst.generate_next()

        src_loss, dst_loss = self.src_dst_train (warped_src, target_src, target_srcm, target_srcm_em, warped_dst, target_dst, target_dstm, target_dstm_em)
        self.generator_src.record_sample_loss(src_loss,info_src)
        self.generator_dst.record_sample_loss(dst_loss,info_dst)
     

        if self.options['true_face_power'] != 0 :
            self.trueface_Discriminator_train (warped_src, warped_dst)

        if self.options['gan_power'] != 0:
            self.gan_patch_src_dst_train (warped_src, target_src, target_srcm, target_srcm_em, warped_dst, target_dst, target_dstm, target_dstm_em)

        return ( ('src_loss', np.mean(src_loss) ), ('dst_loss', np.mean(dst_loss) ), )
    
    
    def train_one_iter(self):
        iter_time = time.time()
        losses = self.onTrainOneIter()
        iter_time = time.time() - iter_time
        self.iter +=1
        self.loss_history.append ( [float(loss[1]) for loss in losses] )
        return self.iter,iter_time
    
    

    def get_strpath_for_file(self,postfix,full_path=True):
        file_name= self.model_name+postfix;
        if full_path:
            file_name=self.saved_models_dir+"/"+file_name
        return file_name;

    def get_model_filename_list(self):
        return self.model_filename_list
       
    def get_summary_text(self):
        visible_options = self.options.copy()
        
        ###Generate text summary of model hyperparameters
        #Find the longest key name and value string. Used as column widths.
        width_name = max([len(k) for k in visible_options.keys()] + [17]) + 1 # Single space buffer to left edge. Minimum of 17, the length of the longest static string used "Current iteration"
        width_value = max([len(str(x)) for x in visible_options.values()]) + 1 # Single space buffer to right edge
        if len(self.device_config.devices) != 0: #Check length of GPU names
            width_value = max([len(device.name)+1 for device in self.device_config.devices] + [width_value])
        width_total = width_name + width_value + 2 #Plus 2 for ": "

        summary_text = []
        summary_text += [f'=={" Model Summary ":=^{width_total}}=='] # Model/status summary
        summary_text += [f'Current iteration: {str(self.iter)}'] # Iter

        for key in visible_options.keys():
            summary_text += [f'{key}: {str(visible_options[key])}'] # visible_options key/value pairs
        

        summary_text += [f'===== Running Device ===== '] # Training hardware info
        
        if len(self.device_config.devices) == 0:
            summary_text += [f'Using device:CPU'] # cpu_only
        else:
            for device in self.device_config.devices:
                summary_text += [f'Device index:  {device.index}'] # GPU hardware device index
                summary_text += [f'Device Name: {device.name}'] # GPU name
                vram_str = f'{device.total_mem_gb:.2f}GB' # GPU VRAM - Formated as #.## (or ##.##)
                summary_text += [f'Device VRAM: {vram_str}']

        summary_text = "\n".join (summary_text)
        return summary_text
    
    #保存模型
    def save(self,init=False):

        #--- (第一次创建初始化模型）
        if init: 
            nn.tf_sess.run(nn.tf.global_variables_initializer())

        for model, filename in io.progress_bar_generator(self.get_model_filename_list(), "[S]Saving models", leave=False):
            model.save_weights ( self.get_strpath_storage_for_file(filename) )
            print(f"[P]saving model file: {filename}")

        model_data = {
            'iter': self.iter,
            'options': self.options,
            'loss_history': self.loss_history,
            'sample_for_preview' : None,
            'choosed_gpu_indexes' : [0],
        }
        pathex.write_bytes_safe (self.model_data_path, pickle.dumps(model_data) )


    def create_backup(self):
        
        time_str = time.strftime("%m%d-%H%M%S\\")
        autobackups_path=f"{self.saved_models_dir}\\backup\\"
        
        if os.path.exists( autobackups_path) is False:
            os.mkdir(autobackups_path)
        this_autobackups_path=f"{autobackups_path}\\{time_str}"
        os.mkdir(this_autobackups_path)

        file_paths=pathex.get_file_paths(self.saved_models_dir)
        for filepath in file_paths:
            origin_filepath = self.saved_models_dir+"\\"+filepath.name            
            backup_filepath=this_autobackups_path+"\\"+filepath.name  
            #print(origin_filepath,"\n",backup_filepath,"-------------")
            shutil.copy ( origin_filepath, backup_filepath )

    def export_dfm(self,output_path=""):
        self.load_model_from_dir()
        print("[S]开始导出dfm...")
        with self.tf.device (nn.tf_default_device_name):
            warped_dst = self.tf.placeholder (nn.floatx, (None, self.options["resolution"], self.options["resolution"], 3), name='in_face')
            warped_dst = self.tf.transpose(warped_dst, (0,3,1,2))

            if 'df' in self.options["archi_type"]:
                gpu_dst_code     = self.inter(self.encoder(warped_dst))
                gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(gpu_dst_code)
                _, gpu_pred_dst_dstm = self.decoder_dst(gpu_dst_code)

            elif 'liae' in self.options["archi_type"]:
                gpu_dst_code = self.encoder (warped_dst)
                gpu_dst_inter_B_code = self.inter_B (gpu_dst_code)
                gpu_dst_inter_AB_code = self.inter_AB (gpu_dst_code)
                gpu_dst_code = self.tf.concat([gpu_dst_inter_B_code,gpu_dst_inter_AB_code], nn.conv2d_ch_axis)
                gpu_src_dst_code = self.tf.concat([gpu_dst_inter_AB_code,gpu_dst_inter_AB_code], nn.conv2d_ch_axis)

                gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder(gpu_src_dst_code)
                _, gpu_pred_dst_dstm = self.decoder(gpu_dst_code)

            gpu_pred_src_dst = self.tf.transpose(gpu_pred_src_dst, (0,2,3,1))
            gpu_pred_dst_dstm = self.tf.transpose(gpu_pred_dst_dstm, (0,2,3,1))
            gpu_pred_src_dstm = self.tf.transpose(gpu_pred_src_dstm, (0,2,3,1))

        self.tf.identity(gpu_pred_dst_dstm, name='out_face_mask')
        self.tf.identity(gpu_pred_src_dst, name='out_celeb_face')
        self.tf.identity(gpu_pred_src_dstm, name='out_celeb_face_mask')

        output_graph_def = self.tf.graph_util.convert_variables_to_constants(
            nn.tf_sess,
            self.tf.get_default_graph().as_graph_def(),
            ['out_face_mask','out_celeb_face','out_celeb_face_mask']
        )

        import tf2onnx
        with self.tf.device("/CPU:0"):
            model_proto, _ = tf2onnx.convert._convert_common(
                output_graph_def,
                name='SAEHD',
                input_names=['in_face:0'],
                output_names=['out_face_mask:0','out_celeb_face:0','out_celeb_face_mask:0'],
                opset=12,
                output_path=output_path)
        print("[S]导出完成")



    def is_reached_iter_goal(self):
        if self.iter>0 and self.options["target_iter"]==self.iter:
            return True
        return False
  
    def finalize(self):
        nn.close_session()
 
