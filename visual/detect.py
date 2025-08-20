import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .utils import Load, RunInference
import time

def build_emo_detector(URL=None):
    model, device, half, imgsz, stride, opt = Load()

    return URL, model, device, half, imgsz, stride, opt

def Detect(url, model, device, half, imgsz, stride, opt):

    # print("[visual]Running inference on the image...")
    pred_cat, pred_cont = RunInference(url, model, device, half, imgsz, stride, opt)
    time.sleep(1)  # Sleep for a while before the next iteration

    return  pred_cat, pred_cont