
import torch
import torch.nn
import onnx
from facelib.facenet.inception_resnet_v1  import  InceptionResnetV1

pretrain_dict = torch.load(r'F:\AiML\facenet-20180402-114759-vggface2.pt')
print("load pretrain_dict pt file finish ")

resnet = InceptionResnetV1(pretrained='vggface2').eval()
resnet.load_state_dict(pretrain_dict)

onnx_path=r'F:\AiML\facenet-vggface2-55.onnx'
  
x = torch.randn(1,3,160,160)
print("开始导出onnx")
torch.onnx.export(resnet, x, onnx_path)
print("完成导出")