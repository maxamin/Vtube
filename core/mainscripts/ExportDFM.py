import os
import sys
import traceback
import queue
import threading
import time
import numpy as np
import itertools
from pathlib import Path
from core import pathex
from core import imagelib
import cv2
from core.swapmodels.Model_SAEHD.SAEHDModel import SAEHDModel
from core.interact import interact as io


def main(model_class_name, models_path,save_dir):
    model = SAEHDModel(
                        is_exporting=True,
                        saved_models_path=models_path,
                        cpu_only=True)
    model.export_dfm (save_dir) 
    print(f"完成实时转换模型的导出{saved_models_path}",)
