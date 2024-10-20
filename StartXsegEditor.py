from facelib.XSegEditor import XSegEditorNew
from pathlib import Path
import argparse

if __name__=="__main__":  
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--folder', type=str, help='face folder',default=r"C:\Users\ps\Pictures\随机遮罩")
    args = parser.parse_args()
    path=Path(args.folder)
    exit_code = XSegEditorNew.start (path)

