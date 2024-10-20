#coding=utf-8
# cython:language_level=3
from vtmodel.FaceSwapModelSAE import FaceSwapModelSAE
import cv2,numpy as np,time,sys
from vtmodel.ModelTrainer import TrainerThread 
from core.leras import nn
Trainer=None;

from kit.FaceEngines import FaceEngines
from core.xlib.face.FRect import FRect
import kit.screen_cap as cap

def PreviewModelTest(model,raw_img):
    time_start=time.time()
    raw_img=raw_img.astype(np.float32)/255.0
    H,W,C=raw_img.shape
    rects = FaceEngines.getYoloEngineOnnx().extract (raw_img, 0.5,fixed_window=256)[0]
    if  len(rects)==0:
        print("No Face Detected");
        return
    l,t,r,b= rects[0]
    ul,ut,ur,ub=l/W,t/H,r/W,b/H
    face_urect=FRect.from_ltrb((ul,ut,ur,ub))
    face_rect_img, face_uni_mat=face_urect.cut(raw_img,1.4,256) 

    #face_align_img, src_to_align_uni_mat = sd.face_ulmrks_inSrc.cut(frame_image_src_float, coverage=sd.AlignCoverage, 
    #                    output_size=sd.AlignResolution,exclude_moving_parts=True)
    #aligned_to_source_uni_mat = src_to_align_uni_mat.invert()

    test_align_face=face_rect_img
    h,w,c=test_align_face.shape;
    resolution=model.options.get("resolution",256)
    if h !=resolution:
        test_align_face=cv2.resize(test_align_face,(resolution,resolution))
    if test_align_face.dtype == np.uint8:
            test_align_face = test_align_face.astype(np.float32)
            test_align_face /= 255.0
    pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm=model.get_test_preview(test_align_face)
    print("pred_dst_dstm:",pred_dst_dstm.shape)
    out_dst_dst=pred_dst_dst.transpose((0,2,3,1))[0]
    out_dst_dstm=pred_dst_dstm.transpose((0,2,3,1))[0]
    out_dst_src=pred_src_dst.transpose((0,2,3,1))[0]
    out_dst_srcm=pred_src_dstm.transpose((0,2,3,1))[0]
    out_dst_dst=out_dst_dst*out_dst_dstm+(1-out_dst_dstm)*test_align_face
    out_dst_src=out_dst_src*out_dst_srcm+(1-out_dst_srcm)*test_align_face
    preview=np.hstack([test_align_face,out_dst_dst,out_dst_src])
    print("out_dst_src:",out_dst_src.shape)
    time_end=time.time()
    print(time_end-time_start)
    cv2.imshow("align_face",preview)
    cv2.waitKey(0)


if __name__ == '__main__':

    import set_env
    set_env.set_env()
    nn.initialize_main_env()
    nn.initialize(data_format="NCHW")
    #----------------sample load test
    kwargs={"masked_training":True,"gpu_training":True}
    model=FaceSwapModelSAE(saved_models_dir=r"F:\DeepModels\NaZha256-liaeRTM",kwargs=kwargs)
    model.load_model_from_dir()
    model.define_tf_output_and_loss()
    print(model.get_summary_text())
    #model.load_train_samples(training_data_src_path=r"F:\TrainFaceLib\开发\李兰迪",training_data_dst_path=r"F:\TrainFaceLib\开发\男")
    import core.cv2ex as cv2ex
    from core.xlib.image.ImageProcessor import ImageProcessor 
    raw_img=cv2ex.cv2_imread(r"F:\AvVideo\真人原始图片\低颜值生tu\中年妇女.jpg"); 
    PreviewModelTest(model,raw_img)
     
    #------------读取dat文件
    #import core.pathex as pathex
    #dat_files=pathex.get_first_file_end_with(r"F:\DeepModels\aa-XiaoSiSi-L256RTM","data.dat",fullpath=False) 
    #print(dat_files)


    #----------------export dfm mdoels test
    #kwargs={"saved_models_dir":r"F:\DeepModels\aa-XiaoSiSi-L256RTM","masked_training":True,"gpu_training":True,"models_opt_on_gpu":True}
    #model=FaceSwapModelSAE(saved_models_dir=r"F:\DeepModels\aa-XiaoSiSi-L256RTM"，kwargs)
    #model.export_dfm(r"F:\DeepModels\aa-XiaoSiSi-L256RTM\sisi.dfm") 

    #----------------backup mdoels test
    #kwargs={"saved_models_dir":r"F:\DeepModels\aa-XiaoSiSi-L256RTM","masked_training":True,"gpu_training":True}
    #model=FaceSwapModelSAE(kwargs)
    #model.create_backup()

    #---- create new model 
    #kwargs={"archi":archi,"resolution":resolution,"ae_dims":ae_dim,"e_dims":e_dim,"d_dims":d_dim,"d_mask_dims":mask_dim}
    #model=FaceSwapModelSAE(saved_models_dir=models_path,kwargs=kwargs)
    #model.create_model_nn_archi()
    #model.define_tf_output_and_loss()
    #model.empty_run_nn_init()
    #model.save(init=True)

    
    
