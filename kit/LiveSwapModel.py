import numpy as np,time,cv2
import onnx,json
import onnxruntime as rt
from core.xlib.image import ImageProcessor
import kit.Auth as auth

class LiveSwapModel:
    Model_File_Path=None;
    ModelMetaData={};
    OnnxModelBytes=None;
    ortSession=None;
    input_width,input_height=256,256
    live_model_type=1;

    def __init__(self,Model_File_Path,device="cuda"):
        self.Model_File_Path=Model_File_Path
        self.read_model_weights_data(Model_File_Path,device)
        #--- 加载模型文件字节流

    #-----创建---OnnxRuntime-Inference--Session
    def create_ort_session(self,model_bytes,device,model_file_path):
        if "cuda" in device:            
            device_ep="CUDAExecutionProvider"
            ep_flags={'device_id':0}
        else:
            device_ep="CPUExecutionProvider"
            ep_flags={}
        sess_options = rt.SessionOptions()
        sess_options.log_severity_level = 4
        sess_options.log_verbosity_level = -1
        self.ortSession=rt.InferenceSession(model_bytes,providers=[(device_ep,ep_flags)],sess_options=sess_options)

        if self.ortSession is not None:
            self.input_height ,self.input_width=self.ortSession.get_inputs()[0].shape[1:3];
            inputs = self.ortSession.get_inputs()

            if len(inputs) == 0:
                print(f'Invalid model {model_file_path}')
                self.ortSession=None;
                return
            elif len(inputs) == 1:
                self.live_model_type = 1
                if 'in_face' not in inputs[0].name:
                    print(f'Invalid live model {model_file_path},input[0] is not inface')
                    self.ortSession=None;
                    return
            elif len(inputs) == 2:
                if 'morph_value' not in inputs[1].name:
                    print(f'Invalid live model {model_file_path},morph_value not in input[1]')
                    self.ortSession=None;
                    return 
                self.live_model_type = 2
            elif len(inputs) > 2:
                print(f'Invalid live model {model_file_path},input more than 2')
                self.ortSession=None;
                return 


    def read_model_weights_data(self,model_file_path,device):
        if model_file_path.lower().endswith(".dfm"):
            data_bytes=onnx._load_bytes(model_file_path);
            self.create_ort_session(data_bytes,device,model_file_path)
        if model_file_path.lower().endswith(".vtfm"):
            data_bytes=onnx._load_bytes(model_file_path)
            meta_bytes=data_bytes[0:1024].decode(encoding="utf-8").replace('#','')
            #print(meta_bytes)
            self.ModelMetaData=json.loads(meta_bytes)
            data_bytes=data_bytes[1024:]
            barr=bytearray(data_bytes)
            tmp=barr[0:100]
            barr[0:100]=barr[100:200]
            barr[100:200]=tmp
            data_bytes=bytes(barr)
            self.create_ort_session(data_bytes,device,model_file_path)

        

    #-----转换图像------------
    def convert(self,img,morph_factor=1.0):
        ip = ImageProcessor(img)
        N,H,W,C = ip.get_dims()
        dtype = ip.get_dtype()
        img = ip.resize( (self.input_width,self.input_height) ).ch(3).to_ufloat32().get_image('NHWC')
        if self.ortSession is None:
            print("FaceSwap加载失败，LiveSwap模型为空，可能模型文件损坏");
            out_celeb_mask=out_celeb=out_face_mask=np.zeros_like(img,dtype=np.float32)

        if self.ortSession is not None:
            if self.live_model_type==1:
                out_face_mask, out_celeb, out_celeb_mask = self.ortSession.run(None, {'in_face:0': img})
            if self.live_model_type==2:
                factor=np.float32([morph_factor])
                out_face_mask, out_celeb, out_celeb_mask = self.ortSession.run(None, {'in_face:0': img,"morph_value:0":factor})

        out_celeb      = ImageProcessor(out_celeb).resize((W,H)).ch(3).to_ufloat32().get_image('NHWC')
        out_celeb_mask = ImageProcessor(out_celeb_mask).resize((W,H)).ch(1).to_ufloat32().get_image('NHWC')
        out_face_mask  = ImageProcessor(out_face_mask).resize((W,H)).ch(1).to_ufloat32().get_image('NHWC')

        return out_celeb, out_celeb_mask, out_face_mask

  

    @staticmethod
    def conver_dfm_to_vtfm(dfm_path,vtfm_path,dict_data):  
        #--- 创建文件
        fvtfm=open(vtfm_path,'wb')
        #--- 写入文件头（模式、机器码、验证令牌） 
        str_bytes=json.dumps(dict_data).ljust(1024,'#').encode(encoding="utf-8")
        fvtfm.write(str_bytes)

        #---- 写入原onnx数据(前200个字符分别+1）
        dfm_bytes=onnx._load_bytes(dfm_path)
        barr=bytearray(dfm_bytes)
        tmp=barr[0:100]
        barr[0:100]=barr[100:200]
        barr[100:200]=tmp
        dfm_bytes=bytes(barr)
        fvtfm.write(dfm_bytes)
        fvtfm.close()
    #conver_vtfm_from_dfm(dfm_file_path,vtfm_file_path)

    @staticmethod
    def read_model_metadata(model_path): 
        metadata={"mode":"free","format":"dfm"}
        if model_path.endswith(".vtfm"):
            file=open(model_path,"rb")
            if file is not None:
                meta_bytes=file.read(1024).decode(encoding="utf-8").replace('#','')
                metadata=json.loads(meta_bytes)  
            file.close()
        return metadata

    @staticmethod 
    def check_model_authorized(model_path):
        if model_path.endswith(".dfm"):
            return True,"free","-","-";
        metadata=LiveSwapModel.read_model_metadata(model_path)
        mode=metadata.get("auth_mode","-")
        if "machine" in mode:  
            cpu_sn=auth.getMachineSn(False);
            model_id=metadata.get("model_id","vtfm")
            request_code=cpu_sn[-18:]+"-"+model_id
            right_token=auth.get_md5_hash_str(f"{request_code}{request_code}-zzz")[0:8]
            token=metadata.get("auth_token","--")
            if right_token==token:
                return True,mode,request_code,right_token
            else:
                return False,mode,request_code,right_token
        if "pwd" in mode:
            right_token=metadata.get("pwd_hash","-")
            return False,mode,"",right_token
     
        return False,mode,"free","free"

    @staticmethod
    def write_model_metadata(model_path,key,value):
        if model_path.endswith(".dfm"):
            return
        metadata=LiveSwapModel.read_model_metadata(model_path)
        metadata[key]=value;
        str_bytes=json.dumps(metadata).ljust(1024,'#').encode(encoding="utf-8")
        with open(model_path,"rb+") as file:
            file.write(str_bytes)

if __name__=="__main__":            
    from core import cv2ex
    import set_env
    set_env.set_env()
    input_face=cv2ex.cv2_imread(r"F:\TrainFaceLib\dy你的星\z_c10_0.jpg")
    live_model=LiveSwapModel(r"F:\VtubeKit\Model_Lib\娜扎-DF256ud.vtfm","cuda")
    out_face,mask_c,mask_src=live_model.convert(input_face)
    cv2.imshow("input_face",input_face)
    cv2.imshow("out_face",out_face[0])
    cv2.waitKey(15000);