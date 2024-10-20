#coding=utf-8
# cython:language_level=3
import os,time,shutil
import requests

def DownLoadFiles(root_dir,download_temp_dir,server_root,DownLoad=True,MoveFile=True):
    if os.path.exists(download_temp_dir) is False:
        os.mkdir(download_temp_dir)
    updatelist_save_path=download_temp_dir+"\\UpdateList.txt"

    print("准备下载UpdateList.txt")
    try:
        url="http://d.vtubekit.com/UpdateList.txt"
        myfile=requests.get(url)
        f=open(updatelist_save_path,'wb')
        f.write(myfile.content)
        f.close()
        print("完成下载UpdateList.txt")
        print("---------------------------")
    except  Exception as ex:
        print(f"下载UpdateList.txt发生错误,错误原因:\n{ex}\n\n")
        return

    #Step2:分析updatelist，获得需要更新的文件列表
    print("开始分析updatelist和本地文件比对")
    file_count=0
    need_update_files=[]
    file_info_list_str=""
    f=open(updatelist_save_path,'r',encoding='utf-8')
    print(f.readline(),end="")
    for line in f.readlines(): 
            file_count=file_count+1;
            #print(line)
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            rel_path,last_mtime_str_server,file_size=line.split("|")        
            local_file_fullpath=root_dir+rel_path
            if os.path.exists(local_file_fullpath) is   False:
                print("新增文件:"+rel_path);
                need_update_files.append(rel_path)
            if os.path.exists(local_file_fullpath):
                last_mtime_local=float(os.path.getmtime(local_file_fullpath))
                #last_mtime_str_local=time.strftime('%Y-%m-%d %H:%M:%S',last_mtime_local)
                last_mtime_server=time.mktime(time.strptime(last_mtime_str_server, '%Y-%m-%d %H:%M:%S'))
                if last_mtime_local<last_mtime_server:
                    need_update_files.append(rel_path)
                    print(f"过期文件:{rel_path}")
                if last_mtime_local>=last_mtime_server:
                  #print(f"{rel_path},文件是最新的。本地修改时间:{last_mtime_local},服务器:{last_mtime_server}")
                  pass
    f.close()
    print(f"完成比对{file_count}文件，需要更新的文件总数:{len(need_update_files)}")
    print("---------------------------")

    if len(need_update_files)==0:
        print("！！！！系统无需更新\n\n\n")
        return
    
    #Step3:下载需要更新的文件列表（存放到临时目录）
    print("开始下载文件")
    idx=0;
    download_count=len(need_update_files)
    for rel_path  in need_update_files:
        if DownLoad is False:
            continue
        idx=idx+1;
        file_name=os.path.basename(rel_path)
        url=server_root+rel_path
        save_path=download_temp_dir+file_name
        print(f"下载进度[{idx}/{download_count}],下载文件{rel_path}".ljust(100,' '),end='\n')
        myfile=requests.get(url)
        f=open(save_path,'wb')
        f.write(myfile.content)
        f.close()
    
        time.sleep(0.2)

    print("\n文件下载结束")
    print("---------------------------")

    #Step4:从临时目录移动到正确的目录
    print("开始移动归位下载的文件")
    for rel_path  in need_update_files:
        file_name=os.path.basename(rel_path)
        save_path=download_temp_dir+file_name
        moveto_path=root_dir+rel_path;
        moveto_dir=os.path.dirname(moveto_path)
        if os.path.exists(moveto_dir) is False:
            os.mkdir(moveto_dir)
        if os.path.exists(save_path) is False:
            print(f"错误：{save_path}文件不存在")
            continue;
        shutil.move(save_path,moveto_path)
    print("完成移动下载的文件，更新结束")
    print("---------------------------")

def UpdateSystem():
    root_dir=r"E:/VtubeKit_Dist/"
    root_dir=os.path.dirname(__file__)+"\\"
    download_temp_dir=root_dir+"/update/"
    server_root="http://d.vtubekit.com/SrcDist/"
    DownLoad=True; MoveFile=True;
    DownLoadFiles(root_dir,download_temp_dir,server_root)

if __name__=="__main__":  
    UpdateSystem()
    

    

    
