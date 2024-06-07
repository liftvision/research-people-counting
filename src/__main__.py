import logging
import sys

import cv2

from src.models import ObjectDetectionModel
from src.utils.capture import *
from src.utils.timer import Timer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(f"Python version: {sys.version}, {sys.version_info} ")


def get_capture() -> Capture:
    return CameraCapture()


def main():
    model_path = 'models/yolov7-tiny.onnx'
    model = ObjectDetectionModel(model_path)
    capture = get_capture()
    global_timer = Timer()
    inference_timer = Timer()
    CV2_PUT_TEXT_OPTIONS = {
        'fontFace': cv2.FONT_HERSHEY_SIMPLEX,
        'fontScale': 0.75,
        'color': (255, 255, 255),
        'thickness': 2,
    }
    while cv2.waitKey(1) == -1:
        with global_timer:
            frame = next(capture)
            with inference_timer:
                output = model.predict(frame)
            cv2.putText(output.image, f'{global_timer.frequency():.2f} FPS', (32, 32), **CV2_PUT_TEXT_OPTIONS)
            cv2.putText(output.image, f'{inference_timer.frequency():.2f} FPS (inference)', (32, 64), **CV2_PUT_TEXT_OPTIONS)
            cv2.imshow('frame', output.visualize())


main()
