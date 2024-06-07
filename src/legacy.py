import cv2
import torch
from time import time
import psutil
import os
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.torch_utils import select_device
from utils.plots import plot_one_box


device = select_device('')
model = attempt_load('yolov7-tiny.pt', map_location=device)
stride = int(model.stride.max())
img_size = 640
model.to(device).eval()


cap = cv2.VideoCapture(0)

prev_time = time()


pid = os.getpid()
ps = psutil.Process(pid)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 현재 시간 계산
    curr_time = time()
    fps = 1 / (curr_time - prev_time)  # FPS 계산
    prev_time = curr_time  # 현재 시간을 이전 시간으로 업데이트

    # 이미지 전처리
    img = letterbox(frame, img_size, stride=stride)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = torch.from_numpy(img.copy()).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # 객체 탐지
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.70, 0.45, classes=[0], agnostic=False)


    for i, det in enumerate(pred):
        if len(det):
            # Rescale boxes from img_size to frame size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

            # 결과를 프레임에 그리기
            for *xyxy, conf, cls in reversed(det):
                label = f'{model.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=(255, 0, 0), line_thickness=3)


    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    # memoryUse = ps.memory_info().rss / (1024 ** 2)
    # cv2.putText(frame, f'Memory Usage: {memoryUse:.2f} MB', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
/Users/hepheir/GitHub/liftvision/ppc-yolo-v7-tiny/requirements.txt /Users/hepheir/GitHub/liftvision/ppc-yolo-v7-tiny/.gitignore