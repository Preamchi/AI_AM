import argparse
import os
import platform
import csv
import sys
import time
import random
from datetime import date, datetime, timedelta
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

@smart_inference_mode()
def run(
        config_file="C:/yolov5/sol_config.ini",
        weights=ROOT /'AImodels/solder.pt',
        source= '',  # file/dir/URL/glob/screen/0(webcam)
        data='',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.90,  # confidence threshold
        iou_thres=0.50,  # NMS IOU threshold
        max_det=1,  # maximum detections per image
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=True,  # save results in CSV format
        save_conf=True,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='',  # save results to project/name
        name='',  # save results to project/name
        exist_ok=True,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1, # video frame-rate stride
       
):
    config.read(config_file)
    config['AI_Check']['Last_folder'] = str(latest_fol)
    config['AI_Check']['Serial'] = str(new_sn)

    config['AI_Check']['Result'] = ''
    result = ''

    with open('C:/yolov5/sol_config.ini', 'w') as cf:   
       config.write(cf)

    sn_fol = config.get("AI_Check","Last_folder").value
    sn_name = config.get('AI_Check', 'Serial').value
   
    source = sn_fol 
    current_timestamp = datetime.now().strftime('%d_%m_%Y')
    
# ------------------------------------------------------------------------------------------------------
    # Detect Code (All)
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download 

    NG = "NG"
    OK = "OK"

    # Directories
    save_ok_dir = increment_path(Path(project)/(current_timestamp) / sn_name / OK, exist_ok=True)  # increment run
    (save_ok_dir / 'labels' if save_txt else save_ok_dir).mkdir(parents=True, exist_ok=True)  # make dir

    save_ng_dir = increment_path(Path(project)/(current_timestamp) / sn_name / NG , exist_ok=True)  # increment run
    (save_ng_dir / 'labels' if save_txt else save_ng_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    detected_files = []

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # 10 ภาพ loop 10 รอบ

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_ok_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)


        # # Define the path for the CSV file
        # csv_path = save_ok_dir / 'predictions.csv'

        # # Create or append to the CSV file
        # def write_to_csv(image_name, prediction, confidence):
        #     data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
        #     with open(csv_path, mode='a', newline='') as f:
        #         writer = csv.DictWriter(f, fieldnames=data.keys())
        #         if not csv_path.is_file():
        #             writer.writeheader()
        #         writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_ok = str(save_ok_dir / p.name)  # im.jpg
            save_ng = str(save_ng_dir / p.name)
            txt_path = str(save_ok_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            # detected_files = []
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                detected_files.append(p.name.split('.')[0])
                result = "NG"

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'

                    # if save_csv:
                    #     write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                       
                        
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_ok_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                    if save_img:
                        if dataset.mode == 'image':
                            cv2.imwrite(save_ng, im0)
            else:
                if result != "NG":
                    result = "OK"
                
                if save_img:
                        if dataset.mode == 'image':
                            cv2.imwrite(save_ok, im0)

            
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if not len(det) and save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_ok, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_ng:  # new video
                        vid_path[i] = save_ng
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_ng = str(Path(save_ng).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_ng, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
            
        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results 1 tab
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_ok_dir.glob('labels/*.txt')))} labels saved to {save_ok_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_ok_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

# set Image_Path ----------------------------------------------------------------------------------------------------------
   
    if result == 'NG':
        set_img = save_ng_dir
    else:
        set_img = save_ok_dir

    files = os.listdir(set_img)
    select_image = random.choice(files)
    image_path = os.path.join(set_img,  select_image )

    config['AI_Check']['Result'] = str(result)
    config['AI_Check']['Image_Path'] = str(image_path)
    config['AI_Check']['Last_folder'] = str('')


    with open('C:/yolov5/sol_config.ini', 'w') as f:   
        config.write(f)

# csv file ----------------------------------------------------------------------------------------------------------
    name_today = datetime.now().strftime('%Y-%m-%d')
    csv_file = f"D:/AI_Result/{name_today}.csv"

    file_exists = os.path.isfile(csv_file)
    data = {'Serial Number': sn_name, 'Result': result}

    if not file_exists:

        with open(csv_file, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames= data.keys())
            writer.writeheader()
            writer.writerow(data)         
    else:

        with open(csv_file, mode='r') as f:
            reader = csv.DictReader(f)
            sn_exist = {row['Serial Number'].strip() for row in reader}

        if data['Serial Number'].strip() not in sn_exist:

            with open(csv_file, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                writer.writerow(data)
                print("Data Write In CSV. Now")
        else:
            print("Serial Is Exist!")

# end csv file -----------------------------------------------------------------------------------------------------------    
          
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'AImodels/solder.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default='', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.90, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    # check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))    
    run(**vars(opt))

if __name__ == "__main__":

    config = ConfigUpdater()
    config_file = "C:/yolov5/sol_config.ini" 
    config.read(config_file)
    model = config.get("Setting","Model_Path").value
    today = date.today()
    date_check = today.strftime("%Y%m%d")


    config['AI_Check']['Serial'] = str('')
    config['AI_Check']['Result'] = str('')
    config['AI_Check']['Image_Path'] = str('')
    config['AI_Check']['Last_folder'] = str('')
    config['AI_Check']['Check_date'] = str(date_check)

    with open('C:/yolov5/sol_config.ini', 'w') as f:   
        config.write(f)
    

    new_sn = ""
    date_fol = ""

    while True:   
        config.read(config_file)
        if (config["AI_Check"]["start"].value == "1"):

            fonttext = "./Oswald-Bold.ttf"
            borderType = cv2.BORDER_CONSTANT
            fontmodel = ImageFont.truetype(fonttext, 30)
            font_result = ImageFont.truetype(fonttext, 60)
            sn_result = ImageFont.truetype(fonttext, 30)
            w1, h1 = 508, 80
            w2, h2 = 508, 180
            shape1 = [(20, 20), (w1 - 10, h1 - 10)] 
            shape2 = [(20, 70), (w2 - 10, h2 - 10)]

            for file in os.listdir(model):
                old_sn = config.get("AI_Check","Serial").value
                run_date = config.get("AI_Check","Check_date").value
                
                if file.endswith(run_date):
                    date_fol = file
                
                if date_fol == run_date:
                    is_path = os.path.join(model, date_fol)
                    files = [file for file in os.listdir(is_path) if not file.endswith('.csv')]

                    if files:
                        new_sn = max(files, key=lambda x: os.path.getmtime(os.path.join(is_path, x)))
                        latest_fol = os.path.join(model, date_fol, new_sn)
                        # print("new:", new_sn)
                        # print("old:", old_sn)

                        result = config.get("AI_Check", "Result").value
                        img_path = config.get("AI_Check", "Image_Path").value
                        sn_name = config.get("AI_Check", "Serial").value

                        if (new_sn != old_sn or old_sn == ""):
                            img = "C:/yolov5/data/images/frameimage.png"  
                            img_result = cv2.imread(img)
                            img_resize = cv2.resize(img_result, (480,300), interpolation = cv2.INTER_LINEAR)
                            img_result = cv2.copyMakeBorder( img_resize, 180, 20, 20, 20, borderType, None, (255, 255, 255))

                            img_pil = Image.fromarray(img_result)
                            draw = ImageDraw.Draw(img_pil)
                            
                            draw.rectangle(shape1, fill="#000000")
                            draw.text((210, 22), "RESULT", font=fontmodel, fill=(255, 255, 255))
                            draw.rectangle(shape2, fill=(217, 217, 217))
                            draw.text((120, 70), "Processing...", font=font_result, fill=(74, 74, 74))

                            img_result = np.array(img_pil)
                            cv2.imshow("AI_Inspection_PC Result", img_result)
                            
                            
                            opt = parse_opt()
                            main(opt)
                            
                            time.sleep(0.3)

                        else:
                            print("Waiting New...")
                            time.sleep(0.3)

                            today = date.today()
                            current_date_check = today.strftime("%Y%m%d")

                            if current_date_check != run_date:
                                config['AI_Check']['Check_date'] = str(current_date_check)
                                
                                with open('C:/yolov5/sol_config.ini', 'w') as f:   
                                    config.write(f)
         
                            if result == "NG":
                                img_result = cv2.imread(img_path)
                                img_resize = cv2.resize(img_result, (480,300), interpolation = cv2.INTER_LINEAR)
                                img_result = cv2.copyMakeBorder( img_resize, 180, 20, 20, 20, borderType, None, (255, 255, 255))
                                
                                img_pil = Image.fromarray(img_result)
                                draw = ImageDraw.Draw(img_pil)
                                
                                draw.rectangle(shape1, fill="#000000")
                                draw.text((210, 22), "RESULT", font=fontmodel, fill=(255, 255, 255))
                                draw.rectangle(shape2, fill=(0, 0, 255))
                                draw.text((220, 75), result, font=font_result, fill=(255, 255, 255))
                                draw.text((30, 430), "Serial: " + sn_name + "-" + result, font= sn_result, fill=(255, 255, 255),stroke_width=2, stroke_fill='black')

                                img_result = np.array(img_pil)
                                cv2.imshow("AI_Inspection_PC Result", img_result)

                            elif result == "OK":
                                img_result = cv2.imread(img_path)
                                img_resize = cv2.resize(img_result, (480,300), interpolation = cv2.INTER_LINEAR)
                                img_result = cv2.copyMakeBorder( img_resize, 180, 20, 20, 20, borderType, None, (255, 255, 255))
                                
                                img_pil = Image.fromarray(img_result)
                                draw = ImageDraw.Draw(img_pil)
                                
                                draw.rectangle(shape1, fill="#000000")
                                draw.text((210, 22), "RESULT", font=fontmodel, fill=(255, 255, 255))
                                draw.rectangle(shape2, fill=(41, 225, 0))
                                draw.text((220, 75), result, font=font_result, fill=(255, 255, 255))
                                draw.text((30, 430), "Serial: " + sn_name + "-" + result, font= sn_result, fill=(255, 255, 255),stroke_width=2, stroke_fill='black')

                                img_result = np.array(img_pil)
                                cv2.imshow("AI_Inspection_PC Result", img_result)

                            elif result == "wait":
                                img = "C:/yolov5/data/images/frameimage.png"  
                                img_result = cv2.imread(img)
                                img_resize = cv2.resize(img_result, (480,300), interpolation = cv2.INTER_LINEAR)
                                img_result = cv2.copyMakeBorder( img_resize, 180, 20, 20, 20, borderType, None, (255, 255, 255))

                                img_pil = Image.fromarray(img_result)
                                draw = ImageDraw.Draw(img_pil)
                                
                                draw.rectangle(shape1, fill="#000000")
                                draw.text((210, 22), "RESULT", font=fontmodel, fill=(255, 255, 255))
                                draw.rectangle(shape2, fill=(217, 217, 217))
                                draw.text((80, 70), "Waiting New...", font=font_result, fill=(74, 74, 74))

                                img_result = np.array(img_pil)
                                cv2.imshow("AI_Inspection_PC Result", img_result)
                                    
                            elif cv2.waitKey(1) & 0xFF == ord('q'):  # Check if 'q' is pressed
                                break  # Break the loop if 'q' is pressed

                    else:
                        print("Wait Serial...")
                        time.sleep(0.3)

                        img = "C:/yolov5/data/images/frameimage.png"  
                        img_result = cv2.imread(img)
                        img_resize = cv2.resize(img_result, (480,300), interpolation = cv2.INTER_LINEAR)
                        img_result = cv2.copyMakeBorder( img_resize, 180, 20, 20, 20, borderType, None, (255, 255, 255))

                        img_pil = Image.fromarray(img_result)
                        draw = ImageDraw.Draw(img_pil)
                        
                        draw.rectangle(shape1, fill="#000000")
                        draw.text((210, 22), "RESULT", font=fontmodel, fill=(255, 255, 255))
                        draw.rectangle(shape2, fill=(217, 217, 217))
                        draw.text((70, 70), "Waiting Serial...", font=font_result, fill=(74, 74, 74))

                        img_result = np.array(img_pil)
                        cv2.imshow("AI_Inspection_PC Result", img_result)
                     
                else:
                    print("Wait Folder...")
                    time.sleep(0.3)
                    img = "C:/yolov5/data/images/frameimage.png"  
                    img_result = cv2.imread(img)
                    img_resize = cv2.resize(img_result, (480,300), interpolation = cv2.INTER_LINEAR)
                    img_result = cv2.copyMakeBorder( img_resize, 180, 20, 20, 20, borderType, None, (255, 255, 255))

                    img_pil = Image.fromarray(img_result)
                    draw = ImageDraw.Draw(img_pil)
                    
                    draw.rectangle(shape1, fill="#000000")
                    draw.text((210, 22), "RESULT", font=fontmodel, fill=(255, 255, 255))
                    draw.rectangle(shape2, fill=(217, 217, 217))
                    draw.text((70, 70), "Waiting Folder...", font=font_result, fill=(74, 74, 74))

                    img_result = np.array(img_pil)
                    cv2.imshow("AI_Inspection_PC Result", img_result)
                   
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if 'q' is pressed
                    break  # Break the loop if 'q' is pressed
        else:
            print("IDK")
            time.sleep(0.3)
