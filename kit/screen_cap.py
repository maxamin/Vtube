#coding=utf-8
# cython:language_level=3
import win32gui,win32ui,win32con
from PIL import Image
import cv2,numpy,time

hWnd=None; hWndDC=None;mfcDC=None;saveDC=None;saveBitMap=None;
window_left, window_top, window_right, window_bot,window_width,window_height=0,0,0,0,0,0

def get_all_windows():
    hWnd_info_list = []; hWnd_list=None;
    hWnd_info_list.append((win32gui.GetDesktopWindow(),"DesktopClass","Desktop_桌面屏幕"))
    def get_wnd_info(hwnd, param):
        title = win32gui.GetWindowText(hwnd)
        clasname = win32gui.GetClassName(hwnd)
        left, top, right, bot = win32gui.GetWindowRect(hwnd)
        width=right-left
        if win32gui.IsWindowVisible(hwnd)==True and width>100 and len(title)>0:
            hWnd_info_list.append((hwnd,clasname,title))
    win32gui.EnumWindows(get_wnd_info,hWnd_list)
    return hWnd_info_list
    #print(hWnd_info_list)

def set_cap_desktop():
    cap_init("DesktopClass","Desktop_桌面屏幕",win32gui.GetDesktopWindow())

def calc_window_size():
    global saveBitMap,saveDC,mfcDC,hWnd
    global left, top, right, bot,width,height
    left, top, right, bot = win32gui.GetWindowRect(hWnd)
    width = right - left
    height = bot - top

#--- 初始化截屏系统
def cap_init(wnd_class=None,wnd_name=None,hwnd=None):
    #print("init win32gui cap")
    #cap_end_release()
    global saveBitMap,saveDC,mfcDC,hWnd,hWndDC
    global window_left, window_top, window_right, window_bot,window_width,window_height
    if hwnd is not None:
        hWnd=hwnd;
    if hwnd is  None:
        hWnd=win32gui.FindWindow(wnd_class,wnd_name)
    #print(hWnd)
    if hWnd is None :
        print("发生错误：窗体未找到")
        return False;
    hWndDC=win32gui.GetWindowDC(hWnd)
    mfcDC = win32ui.CreateDCFromHandle(hWndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    #获取句柄窗口的大小信息
    window_left, window_top, window_right, window_bot = win32gui.GetWindowRect(hWnd)
    window_width = window_right - window_left
    window_height = window_bot - window_top

    return True;


def capture_one_frame(rect=None,clip_mode=0):
    time_start=time.time()
    global hWnd,mfcDC,saveDC,window_width,window_height;
     
    if saveDC is None:
        return None;

    #获取句柄窗口的大小信息
    window_left, window_top, window_right, window_bot = win32gui.GetWindowRect(hWnd)
    window_width = window_right - window_left
    window_height = window_bot - window_top

    left=0;top=0;width=window_width;height=window_height;

    if rect is not None:
        left,top,width,height=rect;
        left=min(left,window_width)
        right=min(left+width,window_width)
        top=min(top,window_height)
        bottom=min(top+height,window_height)

        if clip_mode==1:
            # center mode
            left=(window_width//2-width//2)
            top=(window_height//2-height//2)

    if clip_mode==2:
        left=0;top=0;
        width=window_width;
        height=window_height;

    saveBitMap = win32ui.CreateBitmap()
    #为bitmap开辟存储空间
    saveBitMap.CreateCompatibleBitmap(mfcDC,width,height)
    #将截图保存到saveBitMap中
    saveDC.SelectObject(saveBitMap)
    #保存bitmap到内存设备描述表
    saveDC.BitBlt((0,0), (width,height), mfcDC, (left, top), win32con.SRCCOPY)

    signedIntsArray = saveBitMap.GetBitmapBits(True)
    im_opencv = numpy.frombuffer(signedIntsArray, dtype = 'uint8')
    im_opencv.shape = ( height, width,4)
    im_opencv=cv2.cvtColor(im_opencv, cv2.COLOR_BGRA2BGR)
    h,w,c=im_opencv.shape
    time_use=time.time()-time_start
    #cv2.putText(im_opencv,f'{round(time_use*1000)}ms,{w}x{h}x{c}',(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1) 

    win32gui.DeleteObject(saveBitMap.GetHandle())
    return im_opencv
    
def cap_end_release():
    #内存释放
    global saveBitMap,saveDC,mfcDC,hWnd,hWndDC
    if saveBitMap is not None:
        win32gui.DeleteObject(saveBitMap.GetHandle())
    if saveDC is not None:
        saveDC.DeleteDC()
    if mfcDC is not None:
        mfcDC.DeleteDC()
    if hWndDC is not None:
       win32gui.ReleaseDC(hWnd,hWndDC)

def show_test_cap_window():
    for x in range(0,650):
            im_opencv=capture_one_frame()
            cv2.imshow("im_opencv",im_opencv) #显示 
            cv2.waitKey(30)

#-----------主程序测试
if __name__ == '__main__':  
    hWnd_info_list=get_all_windows()
    for hwnd_info in hWnd_info_list: 
            print('窗口类名:' ,(hwnd_info[1]),",标题:",hwnd_info[2],",hWnd:",hwnd_info[0])
    succed_init=cap_init("QyWindow.GroupClass.BasicWindow",None,18224134)
    cv2.namedWindow('im_opencv') #命名窗口

    if succed_init:
        for x in range(0,1650):
            im_opencv=capture_one_frame()
            cv2.imshow("im_opencv",im_opencv) #显示 
            cv2.waitKey(30)
            #time.sleep(0.2)
    cap_end_release()






