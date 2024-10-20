#coding=utf-8
# cython:language_level=3
import os,sys;

SetRun=False;

def set_env():

    #---- 避免重复设置-----
    global SetRun;
    if  SetRun is True:
        return

    curr_file=os.path.abspath(__file__)
    curr_dir=os.path.dirname(curr_file)
    os.chdir(curr_dir)
    #print(f"chdir {curr_dir} ")
    #print(f"os.getcwd:{os.getcwd()}")
     
    parent_dir=os.path.dirname(curr_dir) 
    cuda_torch_v116_folder=f"{parent_dir}\\\Cuda_Cudnn_dlls\\Cuda_Torch_2.0_Cuda_11.6"
    os.environ["path"]=os.environ["path"]+";"+cuda_torch_v116_folder
    os.environ["cuda_for_torch"]=cuda_torch_v116_folder
    

    #--- 删除Path环境变量中原来的Cuda路径
    path_list=os.environ["Path"].split(";")
    for path in path_list: 
        #print(path)
        if  "NVIDIA GPU Computing Toolkit" in path:
            print(f"已经跳过系统CUDA，加载自带CUDA")  
            os.environ["path"]=os.environ["path"].replace(path,"D:\\asa3")

    #---- 修改【CUDA_PATH】环境变量
    if "CUDA_PATH" in os.environ:
        cuda_path=os.environ["CUDA_PATH"]
        os.environ["CUDA_PATH"]="--"

    #----- 屏蔽Tensorflow的警告信息
    os.environ["TF_MIN_REQ_CAP"]="30"
    os.environ["TF_CPP_MIN_LOG_LEVEL"]="3" 

    SetRun=True;

if __name__=="__main__":
    set_env() 