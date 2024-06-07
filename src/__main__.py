import logging
import sys
import typing

import cv2

from src.models import ObjectDetectionModel
from src.utils import timer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(f"Python version: {sys.version}, {sys.version_info} ")


def get_frames() -> typing.Generator[cv2.typing.MatLike, None, None]:
    source = '0'
    capture = cv2.VideoCapture(source)
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        yield frame


def main():
    model = ObjectDetectionModel('models/yolov7-tiny.onnx')
    for frame in get_frames():
        if cv2.waitKey(1) == -1:
            break
        output = model.predict(frame)
        cv2.putText(output.image, f'{timer.fps:.2f} FPS', (32, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [0, 255, 255], thickness=2)
        cv2.imshow('frame', output.visualize())


main()
