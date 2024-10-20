
import os,shutil;
import argparse

parser = argparse.ArgumentParser(description='参数')
parser.add_argument('-dst_dir', type=str,default="../SrcDist", help='复制的目标文件夹')
args = parser.parse_args()

src_root_folder="../SrcCode/"
dst_root_folder="../SrcDist"
dst_root_folder=args.dst_dir
#dst_root_folder="E:/VtubeKit_Dist/SrcDist"

print("开始移动编译好的pyd文件");
file_list=["LabUiHandler","gui_main_live","gui_main_train","","","","",""]
exclude_list=["build.py","copy_pyd.py","UpdateListCreate.py"]
copy_py_list=["FaceDeformLtw.py"]
folders=["","kit","vvc","vtmodel","facelib/XSegEditor","matting"]
copy_from_lib_dir="../build_files/build_lib";
for folder in folders:
    print(f"-------{folder}--------------")
    abs_folder=src_root_folder+folder;
    for file in os.listdir(abs_folder):
        name,ext=os.path.splitext(file)
        if ext != ".py":
            continue;
        if file  in exclude_list:
            continue
        
        lib_pyd_path=f"{copy_from_lib_dir}/{name}.cp37-win_amd64.pyd"
        lic_cpy_dst_path=f"{dst_root_folder}/{folder}/{name}.pyd"
        if file in copy_py_list:
            lib_pyd_path=f"{src_root_folder}/{folder}/{file}"
            lic_cpy_dst_path=f"{dst_root_folder}/{folder}/{file}"
        try:
            shutil.copy(lib_pyd_path,lic_cpy_dst_path);
            print(lib_pyd_path,"成功复制到：",lic_cpy_dst_path);
        except Exception as ex :
            print(f"xx-复制文件 {lic_cpy_dst_path}发生错误: {ex}")

print("文件复制结束！！！！")