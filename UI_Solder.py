import os
import sys
from datetime import date, datetime
from pathlib import Path
from configupdater import ConfigUpdater
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import PhotoImage
from PIL import ImageFont, ImageDraw, Image

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode

result = 'NG'
sn_name = "1234567"
num = '23'
fonttext = "./Oswald-Bold.ttf"
borderType = cv2.BORDER_CONSTANT
fontmodel = ImageFont.truetype(fonttext, 30)
font_result = ImageFont.truetype(fonttext, 60)
sn_result = ImageFont.truetype(fonttext, 30)
w1, h1 = 300, 80
w2, h2 = 300, 180
shape1 = [(20, 20), (w1 - 10, h1 - 10)] 
shape2 = [(20, 70), (w2 - 10, h2 - 10)]

w3, h3 = 510, 80
w4, h4 = 510, 180
shape3 = [(295, 20),(w3 - 10, h3 - 10)] 
shape4 = [(295, 70),(w4 - 10, h4 - 10)] 

if result == "NG":
    path_image_NG = "C:/Users/5022820537/yolov5/runs/detect/14_02_2024/98765432/NG/4-0-{1_2_-1_-32}.jpg"
    img_result = cv2.imread(path_image_NG)
    img_resize = cv2.resize(img_result, (480,300), interpolation = cv2.INTER_LINEAR)
    img_result = cv2.copyMakeBorder( img_resize, 180, 20, 20, 20, borderType, None, (255, 255, 255))


    img_pil = Image.fromarray(img_result)
    draw = ImageDraw.Draw(img_pil)
    
    draw.rectangle(shape1, fill="#000000")
    draw.text((110, 22), "RESULT", font=fontmodel, fill=(255, 255, 255))

    draw.rectangle(shape3, fill="#000000")
    draw.text((335, 22), "NG COUNT", font=fontmodel, fill=(255, 255, 255))

    draw.rectangle(shape4, fill=(217, 217, 217))
    draw.text((365, 75), num, font=font_result, fill=(74, 74, 74))

    draw.rectangle(shape2, fill=(0, 0, 255))
    draw.text((120, 75), result, font=font_result, fill=(255, 255, 255))
    
    draw.text((30, 430), "Serial: " + sn_name + "-" + result, font= sn_result, fill=(255, 255, 255))

    img_result = np.array(img_pil)

    cv2.imshow("AI_Inspection_PC Result", img_result)  
elif result == "OK":
    img_OK = "C:/Users/5022820537/yolov5/runs/detect/14_02_2024/98765432/NG/4-0-{1_2_-1_-32}.jpg"  
    img_result = cv2.imread(img_OK)
    img_resize = cv2.resize(img_result, (480,300), interpolation = cv2.INTER_LINEAR)
    img_result = cv2.copyMakeBorder( img_resize, 180, 20, 20, 20, borderType, None, (255, 255, 255))
    
    img_pil = Image.fromarray(img_result)
    draw = ImageDraw.Draw(img_pil)
    
    draw.rectangle(shape1, fill="#000000")
    draw.text((210, 22), "RESULT", font=fontmodel, fill=(255, 255, 255))
    draw.rectangle(shape2, fill=(41, 225, 0))
    draw.text((220, 75), result, font=font_result, fill=(255, 255, 255))
    draw.text((30, 430), "Serial: " + sn_name + "-" + result, font= sn_result, fill=(255, 255, 255), stroke_width=2, stroke_fill='black')

    img_result = np.array(img_pil)

    cv2.imshow("AI_Inspection_PC Result", img_result) 

elif result == "wait":
    img_OK = "C:/Users/5022820537/yolov5/runs/detect/14_02_2024/98765432/NG/4-0-{1_2_-1_-32}.jpg"  
    img_result = cv2.imread(img_OK)
    img_resize = cv2.resize(img_result, (480,300), interpolation = cv2.INTER_LINEAR)
    img_result = cv2.copyMakeBorder( img_resize, 180, 20, 20, 20, borderType, None, (255, 255, 255))

    img_pil = Image.fromarray(img_result)
    draw = ImageDraw.Draw(img_pil)
    
    draw.rectangle(shape1, fill="#000000")
    draw.text((210, 22), "RESULT", font=fontmodel, fill=(255, 255, 255))
    draw.rectangle(shape2, fill=(217, 217, 217))
    draw.text((110, 80), "Waiting Serial...", font=font_result, fill=(74, 74, 74))

    img_result = np.array(img_pil)

    cv2.imshow("AI_Inspection_PC Result", img_result) 

cv2.waitKey(0)
cv2.destroyAllWindows()
