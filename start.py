import argparse
from facelib.XSegEditor import XSegEditorNew

parser = argparse.ArgumentParser(description='the app name to run')
parser.add_argument('-app', type=str,default="live", help='the app name to run')
parser.add_argument('-path', type=str,default="", help='path')
#parser.add_argument('token', type=str,default="token000", help='validate token')
args = parser.parse_args()
#print(args.app)

if __name__ == '__main__':
    if "live" in args.app:
        import gui_main_live
        gui_main_live.main()
    #print(args.filename, args.count, args.verbose)
    if "train" in args.app:
        import gui_main_train
        gui_main_train.main()
    if "extract" in args.app or "lab" in args.app:
        import gui_main_lab
        gui_main_lab.main()
    if "xseg_editor" in args.app:        
        from pathlib import Path
        path=Path(args.path)
        XSegEditorNew.start (path)
    if "update" in args.app:
        import UpdateListDownload
        UpdateListDownload.UpdateSystem()

 