#coding=utf-8
# cython:language_level=3

from PyQt5.QtWidgets import QFileDialog,QMessageBox;
from PyQt5.QtCore import QThread, pyqtSignal
import core.interact.interact as io
import time,threading
from  configparser import ConfigParser  
from urllib.request import urlopen
import json,os
import kit.ShareData as sd
import kit.Auth as auth
class ServerRequestThread(QThread):
    mAlertSignal=pyqtSignal();
    app="live"
    delay=5.0;
    cpu_sn=None;
    interval=20;
    continue_thread=True;

    def __init__(self,app="live",delay=5.0,mode="auth_check",interval=20):
        super(ServerRequestThread, self).__init__()
        self.cpu_sn=auth.getMachineSn();
        self.app=app;
        self.delay=delay; 
        self.continue_thread=True;
        self.mode=mode
        self.interval=interval;

   

    def setConfig(self,app,delay):
        self.app=app;
        self.delay=delay; 

    def run(self):
        time.sleep(self.delay);
        while(self.continue_thread):

            try:
                if self.mode=="auth_check":
                    self.app_auth_check();
                if self.mode=="create_model":
                    self.request_model_create();
            
            except Exception as ex:
                timestr= time.strftime('%H:%M:%S')
                msg=f"[{timestr}]网络请求发生异常：{ex}"  
                #print(msg)

            time.sleep(self.interval) 

    def app_auth_check(self):
        myURL = urlopen(f"http://v.vtubekit.com/KitAPI/kitauth.aspx?m=check&sn={self.cpu_sn}&app={self.app}&v={sd.Version}")
        result=myURL.read()
        self.parse_server_response(result)

    def request_model_create(self):
        myURL = urlopen(f"http://v.vtubekit.com/KitAPI/ModelDeal.aspx?m=create_model&sn={self.cpu_sn}&app={self.app}&v={sd.Version}")
        result=myURL.read()
        self.parse_server_response(result)

    def parse_server_response(self,str):
        data=json.loads(str)
        app_enable=data.get("app_enable","no");
        alert=data.get("alert","-")
        msg=data.get("msg","-")
        print_info=data.get("print_info","-")
        open_web=data.get("open_web","")
        url=data.get("url","")
        end_check=data.get("end_check","")
        exit_app=data.get("exit_app","")
        show_warn=data.get("show_warn","")
        server_date=data.get("date","")
        receive_token=data.get("token","")
        token_seed=f"{alert}{exit_app}{show_warn}{open_web}{end_check}{server_date}-ttt"
        token_valid=auth.get_md5_hash_str(token_seed)
        #print(f"接受的：{receive_token}\ntoken_valid:{token_valid}")

        if receive_token!=token_valid:
            sd.ShowAlert=True;
            sd.AlertMsg="非法篡改信息，即将退出程序";
            sd.ExitApp=True; 
            self.mAlertSignal.emit();
            return

        if app_enable=="yes":
            sd.AppEnable=True;

        if exit_app=="yes":
            sd.ExitApp=True; 

        if alert =='yes': 
            sd.ShowAlert=True;
            sd.AlertMsg=msg;
                    
        if print_info=="yes":
            print(f"提示信息:{msg}",end='\r')
                
        if open_web=='yes':
            sd.OpenWeb=True;
            sd.WebUrl=url;
        else:
            sd.OpenWeb=False;

        if show_warn=="no":
            sd.ShowWarn=False;

        if sd.SysDate<server_date:
            sd.ShowAlert=True;
            sd.AlertMsg="您的系统时间错误，将退出程序";
            sd.ExitApp=True; 

        self.mAlertSignal.emit();
        if end_check=="yes":
            #print("网络请求进程结束退出");
            self.continue_thread=False;


def run_oo(self,app="live",delay=2.0):
    sn=self.getMachineSn();
     

#start_check_auth();