import logging
import sys

import cv2

from src.models import ObjectDetectionModel
from src.utils import timer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(f"Python version: {sys.version}, {sys.version_info} ")


model = ObjectDetectionModel('models/yolov7-tiny.onnx')
capture = cv2.VideoCapture(0)


def main():
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        with timer:
            output = model.predict(frame)

        cv2.putText(output.image, f'{timer.fps:.2f} FPS', (16, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=2)
        cv2.imshow('frame', output.visualize())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

main()
