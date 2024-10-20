#coding=utf-8

from distutils.core import setup
from Cython.Build import cythonize
from collections import namedtuple
import os

if __name__=="__main__":
    curr_path=os.getcwd()
    parent=os.path.dirname(curr_path) 
    
    build_root_dir=f"{parent}\\build_files"
    lib_dir=f"{build_root_dir}\\build_lib";
    print("build_root_dir:",build_root_dir)

    build_files=True;
    if build_files:
        exclude_files=["FaceDeformLtw.py","UpdateListCreate.py.py","build.py","main.py","copy_pyd.py","start.py"]

        root_ext_modules=cythonize(module_list=["*.py"],exclude=["build.py","main.py","copy_pyd.py"],build_dir=f"{build_root_dir}\\c")    
        setup(name='root', ext_modules=root_ext_modules  )

        kit_ext_modules=cythonize(module_list=["kit/*.py","vvc/*.py","matting/*.py"],exclude=exclude_files,build_dir=f"{build_root_dir}\\c")
        setup(name='kit', ext_modules=kit_ext_modules  )

        vtmodel_ext_modules=cythonize(module_list=["vtmodel/*.py"],exclude=exclude_files,build_dir=f"{build_root_dir}\\c")
        setup(name='vtmodel', ext_modules=vtmodel_ext_modules  )

        #--build all python
        xseg_ext_modules=cythonize(module_list=["facelib/XSegEditor/*.py"],exclude=["build.py",],build_dir=f"{build_root_dir}\\c")
        setup(name='xseg', ext_modules=xseg_ext_modules  )

    #---参数的含义
    #   libraries=['gdbm', 'readline']  libraries=['X11', 'Xt']

    #---修改文件名
    #print("lib_dir:",lib_dir)
    #for root,dirs,files in os.walk(lib_dir):
    #    for file in files:
    #        full_path=f"{root}\\{file}"
    #        rename_path=full_path.replace("cp37-win_amd64.","")
    #        os.rename(full_path,rename_path)
            #print(full_path,rename_path)
        