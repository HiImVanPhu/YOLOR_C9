import cv2
import numpy as np
from zipfile import ZipFile

from utils.torch_utils import select_device
import socket
from models.models import *
from utils.datasets import *
from utils.general import *
import time
from datetime import datetime
import os
import shutil
# from wang_pakage.process_map import is_in_parking_line

import opt


#IP = "202.191.56.104"
#PORT = 5518

IP = "113.22.128.249"
PORT = 8080


file_name = 'main.zip'
send_check = False

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


out, source, weights, view_img, save_txt, imgsz, cfg, names = \
    opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names

# Initialize
device = select_device(opt.device)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = Darknet(cfg, imgsz)
model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
# model = attempt_load(weights, map_location=device)  # load FP32 model
# imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
model.to(device).eval()
if half:
    model.half()  # to FP16

# Get names and colors
names = load_classes(names)
colors = (0, 0, 255)


# Run inference
# img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
# _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once


def detect(img0):
    H, W, _ = img0.shape
    img0_copy = img0.copy()
    img = letterbox(img0, new_shape=imgsz, auto_size=64)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

    # Process detections
    center = []
    for i, det in enumerate(pred):  # detections per image

        # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                x_center = (x1 + x2) // 2
                y_center = (y1 + y2) // 2
                center.append([x_center, y_center])
                # if not is_in_parking_line(x_center, y_center):
                #     continue
                label = '%s %.2f' % (names[int(cls)], conf)
                # cv2.rectangle(img0, (x1, y1), (x2, y2), colors, 2)
                # cv2.putText(img0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                #             (0, 255, 0), 2)
    status = {}
    for key, value in points_dict_c9.items():
        x, y, w, h = value[0]
        cv2.rectangle(img0, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img0, str(key), (x + 10, y + 30), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 255, 0), 2)
        for [x_cen, y_cen] in center:
            if x <= x_cen <= x + w and y <= y_cen <= y + h:
                # cv2.fillPoly(img0, [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])], (0, 0, 255))
                cv2.rectangle(img0, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(img0, str(key), (x + 10, y + 30), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 0, 255), 2)
                status[key] = 1
                break
            else:
                # cv2.fillPoly(img0, np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]), (0, 255, 0))
                
                status[key] = 0
    # img0 = cv2.addWeighted(img0, 0.2, img0_copy, 1 - 0.2, 0)
    return img0, status


points_dict_c9 = {}
with open("slot_C9.txt", "r") as f:
    points = []
    count = 0
    for line in f.read().split():
        x, y, w, h = list(map(int, line.split(",")))
        points.append([x, y, w, h])
        count += 1
        points_dict_c9[int(count)] = points
        points = []

print(points_dict_c9)
if __name__ == '__main__':
    
    # fourcc = 'mp4v'  # output video codec
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # vid_writer = cv2.VideoWriter(r"D:\Lab IC\demo\ch16_C3 vào_sau 17h25 06012022.mp4", cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    check = 1
    rotate = 0
    frame = 0
    skip = 10  # seconds
    status = {}
    
    with torch.no_grad():
        t = time.time()
        while True:
            #if time.time() - t < 25:
                #continue
            today = f'{datetime.now().year}_{datetime.now().month}_{datetime.now().day}'
            timenow = f'{datetime.now().hour}_{datetime.now().minute}_{datetime.now().second}'            
            path = r"rtsp://admin:bk123456@192.168.0.55:554/Streaming/channels/1"
            cap = cv2.VideoCapture(path)
            t = time.time()
            frame += 1
            ret, img0 = cap.read()
            img0_copy = img0.copy()
            if not ret:
                print(f"Camera fail {today}")
                continue
            img0_copy = cv2.resize(img0_copy, (1280, 720))
            # img0 = cv2.imread('../Draw/C9_04_05_22.jpg')
            cv2.imwrite('C9_original.jpg', img0_copy)
            img0_detected, status = detect(img0)
           
                # img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
            img0_detected = cv2.resize(img0_detected, (1280, 720))
            cv2.imwrite('C9.jpg', img0_detected)
            
            if datetime.now().hour == 25 and not os.path.isdir(f'C9_Image/{today}__original'):
                os.mkdir(f'C9_Image/{today}__original')
                os.mkdir(f'C9_Image/{today}__detected')
            if 25 < datetime.now().hour < 26:
                cv2.imwrite(f'C9_Image/{today}__original/{timenow}_original.jpg', img0_copy)
                cv2.imwrite(f'C9_Image/{today}__detected/{timenow}_detected.jpg', img0_detected)
                print('saved')
                
            img0_detected = cv2.resize(img0_detected, (960, 640))
            cv2.imshow("Image", img0_detected)
            
            if datetime.now().hour == 25 and not os.path.isfile(f'C9_Image/C9__{today}__original.zip'):
                with ZipFile(f'C9_Image/C9__{today}__original.zip', 'w') as z:
                    lst_dir = os.listdir(f'C9_Image/{today}__original')
                    for i in lst_dir:
                        z.write(f'C9_Image/{today}__original/{i}')
                    #shutil.rmtree(f'C9_Image/{today}')
                        
                with ZipFile(f'C9_Image/C9__{today}__detected.zip', 'w') as z:
                    lst_dir = os.listdir(f'C9_Image/{today}__detected')
                    for i in lst_dir:
                        z.write(f'C9_Image/{today}__detected/{i}')
                        
                send_check = True
            
            with open('frame.txt', 'w+') as f:
                for key, value in status.items():
                    f.write(f"{str(key)} 70 {str(value)}\n")

            with ZipFile(file_name, 'w') as z:
                z.write('C9_original.jpg')
                z.write('C9.jpg')
                z.write('frame.txt')
                                      
            s = socket.socket()
            
            try:
                s.connect((IP, PORT))
                print('connected')
            except:
                print('connect failed')
                
            try:
                s.send(file_name.encode("utf-8"))
                with open(file_name, 'rb') as f:
                    data = f.read()
                    s.sendall(data)
                    print('sent main.zip')
                
                if send_check:
                    s.send(f'C9_Image/C9__{today}__original.zip'.encode("utf-8"))
                    with open(f'C9_Image/C9__{today}__original.zip', 'rb') as f:
                        data = f.read()
                        s.sendall(data)
                        
                    s.send(f'C9_Image/C9__{today}__detected.zip'.encode("utf-8"))
                    with open(f'C9_Image/C9__{today}__detected.zip', 'rb') as f:
                        data = f.read()
                        s.sendall(data)
                        
                    print('sent image')                           
                    send_check = False
            except:
                print('send failed')
            s.close()
            
            # vid_writer.write(img0)
            key = cv2.waitKey(1)
            print("FPS: ", 1 // (time.time() - t), '\n')
            if key == ord("q"):
                print(status)
                break
            if key == ord("c"):
                check = -check
            if key == ord("r"):
                rotate += 1
            if key == ord("n"):
                frame += skip * 25
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            elif key == ord("p") and frame > skip * 25:
                frame -= skip * 25
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            if key == 32:
                cv2.waitKey()

