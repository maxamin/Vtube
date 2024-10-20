from facelib.onnx.YoloV5Face.YoloV5Face import YoloV5Face;
from facelib.onnx.S3FD.S3FD import S3FD

from facelib.onnx.FaceMesh.FaceMesh import FaceMesh
from facelib.onnx.InsightFace2d106.InsightFace2D106 import InsightFace2D106

from facelib.xseg.XSegNet import XSegNet
from  core.leras import nn
from facelib.onnx.Xseg.XSegOnnx import XSegOnnx;
import os

class FaceEngines:

    CenterFaceEngineOnnx=None; S3fdFaceEngineOnnx=None; YoloEngineOnnx=None;
    FaceInsightEngineOnnx=None;   FaceMeshAlignEngineOnnx=None;
    XSegModelEngineTF=None;    XSegEngineOnnx=None; CurrXsegPath="";

    @staticmethod
    def getXSegModelEngineOnnx():
        if FaceEngines.XSegEngineOnnx is None:
            FaceEngines.XSegEngineOnnx=XSegOnnx(XSegOnnx.get_available_devices()[0])
        return FaceEngines.XSegEngineOnnx

    @staticmethod
    def getXSegModelEngineTF(xseg_model_dir=None):
        
        if xseg_model_dir is None:
            xseg_model_dir=os.path.abspath("..\\XSegModels\\default_xseg")
        else:
            xseg_model_dir=os.path.abspath(xseg_model_dir)

        
        if(xseg_model_dir!=FaceEngines.CurrXsegPath):
            nn.initialize_main_env()
            nn.initialize() 
            #nn.tf.reset_default_graph()

            FaceEngines.XSegModelEngineTF = XSegNet(name='XSeg',load_weights=True,weights_file_root=xseg_model_dir,training=False)
            FaceEngines.CurrXsegPath=xseg_model_dir
            print(f"加载遮罩分割引擎结束,路径为{xseg_model_dir}")
        
        return FaceEngines.XSegModelEngineTF

    @staticmethod
    def getFaceMeshAlignEngineOnnx():
        if FaceEngines.FaceMeshAlignEngineOnnx is None:
            FaceEngines.FaceMeshAlignEngineOnnx=FaceMesh(FaceMesh.get_available_devices()[0])
            print("成功加载FaceMesh人脸特征点标记引擎")
        return FaceEngines.FaceMeshAlignEngineOnnx



    @staticmethod
    def getS3fdFaceEngineOnnx():
        if FaceEngines.S3fdFaceEngineOnnx is None:
            FaceEngines.S3fdFaceEngineOnnx=S3FD(S3FD.get_available_devices()[0]);
            print("成功加载S3FD人脸检测引擎")
        return FaceEngines.S3fdFaceEngineOnnx

    @staticmethod
    def getFaceInsightEngineOnnx():
        if FaceEngines.FaceInsightEngineOnnx is None:
            FaceEngines.FaceInsightEngineOnnx=InsightFace2D106(InsightFace2D106.get_available_devices()[0]);
            print("成功加载FaceInsight人脸特征点标记引擎")
        return FaceEngines.FaceInsightEngineOnnx

    @staticmethod
    def getYoloEngineOnnx():
        if FaceEngines.YoloEngineOnnx is None:
            FaceEngines.YoloEngineOnnx=YoloV5Face(YoloV5Face.get_available_devices()[0]);
            print("成功加载Yolo人脸检测引擎")
        return FaceEngines.YoloEngineOnnx
#print(FaceEngines.getAA())

if __name__=="__main__":
    import set_env
    set_env.set_env()
    FaceEngines.getXSegModelEngineTF() 