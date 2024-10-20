#coding=utf-8
# cython:language_level=3

from facelib import FaceType, XSegNet
from core.leras import nn
from pathlib import Path

def export_xseg_onnx():
    nn.initialize_main_env()
    nn.initialize()
    saved_models_path=Path(r"F:\DeepFaceKit\SrcCode\XSegModels\wuhen_tongue_xseg")
    from core.swapmodels.Model_XSeg.Model import XSegModel
    model=XSegModel(is_training=False,is_exporting=True,saved_models_path=saved_models_path,cpu_only=True,
                    debug=False,silent_start=True)
    print("export onnx....")
    model.export_dfm();
    print("export onnx finish")


if __name__ == '__main__':
    export_xseg_onnx()