#coding=utf-8
# cython:language_level=3

import hashlib
import pickle,os
import datetime
import configparser
import kit.ShareData as sd

def get_md5_hash_str(text): 
    hl=hashlib.md5()
    hl.update(text.encode(encoding='utf8'))
    sha1=hl.hexdigest()
    return sha1

def get_dict_sign(dict):
    dict_copy=dict.copy()
    if "sign" in dict_copy:
        dict_copy.pop("sign")
    dict_copy_sorted_keys=sorted(dict_copy)
    merge="ttki0098op000"
    for key in dict_copy_sorted_keys:
        merge=merge+str(dict_copy[key])
    return get_md5_hash_str(merge)

def sign_dict(dict):
    if(len(dict)==0):
        return
    sign=get_dict_sign(dict)
    dict["sign"]=sign
    return dict

def check_dict_sign_valid(dict):
    sign_correct=get_dict_sign(dict)
    sign_record=dict.get("sign","none")
    print(sign_correct,sign_record)
    return sign_correct==sign_record

def getMachineSn(print_sn=True):
    try:
        import wmi
        w = wmi.WMI()    
        cpu_id=w.Win32_Processor()[0].ProcessorId
        baseboard_sn=w.Win32_BaseBoard()[0].SerialNumber
        os_sn=w.Win32_OperatingSystem()[0].SerialNumber
        mergeInfo=cpu_id+baseboard_sn +os_sn
        machineSN=get_md5_hash_str(mergeInfo).upper()[0:18]
        
        SN_sign=get_md5_hash_str(f"{machineSN}-vtksn")
        config=configparser.ConfigParser()
        config.add_section('info')
        config.set("info","machine_sn",machineSN)
        config.set("info","sn_sign",SN_sign)
        config.set("info","version",sd.Version)
        with open("../info.ini","w+") as f:
            config.write(f)   
        if print_sn:
            print(f"本机机器码: {machineSN}")
    except Exception as e:
        import socket 
        machineSN="PC_"+socket.gethostname() 
         
    return machineSN;
 
if __name__=="__main__":
    dict={"mode":"free","sn":"kkkkk","pwd":"pwd","name":"shenteng","author":"ggyycc"}
    print(dict)
    dict=sign_dict(dict)
    print(dict)
    dict["mode"]="ppp"
    valid=check_dict_sign_valid(dict)
    print("sign valid:",valid)
