import os,threading,time,random
from pathlib import Path


align_face=1
seg_mask_result=1
th_main=None;   th_seg=None;
main_ctrl_event=threading.Event()
seg_ctrl_event=threading.Event()


def seg_thread_run():
    global align_face,seg_mask_result,main_ctrl_event,seg_ctrl_event
    while True: 
        seg_ctrl_event.wait()
        time.sleep(0.03)
        seg_mask_result=align_face
        
        main_ctrl_event.set()

def main_thread_run():
    global align_face,seg_mask_result,main_ctrl_event,seg_ctrl_event
    print("主线程运行")
    while True:
        
        align_face=random.random();
        seg_ctrl_event.set()

        time.sleep(0.035)
        main_ctrl_event.wait()
        check=seg_mask_result-align_face
        if check!=0:
            print(f"---发生错误---align_face:{align_face},mask:{seg_mask_result}")
        if check==0:
            print(f"ok,align_face:{align_face},mask:{seg_mask_result}")
        


if __name__=="__main__":
    #global th_main,th_seg;
    th_main=threading.Thread(target=main_thread_run)
    th_main.start()

    th_seg=threading.Thread(target=seg_thread_run)
    th_seg.start()