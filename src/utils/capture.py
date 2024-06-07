import time

import cv2


__all__ = [
    'Capture',
    'CameraCapture',
    'FileCapture',
    'ScreenCapture',
]


class Capture:
    def __iter__(self):
        return self

    def __next__(self) -> cv2.typing.MatLike:
        raise NotImplementedError


class CameraCapture(Capture):
    def __init__(self):
        self.capture = cv2.VideoCapture(0)

    def __del__(self):
        self.capture.release()

    def __next__(self) -> cv2.typing.MatLike:
        ret, frame = self.capture.read()
        if not ret:
            raise StopIteration
        return frame


class FileCapture(Capture):
    def __init__(self, source: str):
        self.capture = cv2.VideoCapture(source)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.spf = 1 / self.fps
        self.last_read = time.time()

    def __del__(self):
        self.capture.release()

    def __next__(self) -> cv2.typing.MatLike:
        while (time.time()-self.last_read) < self.spf:
            time.sleep(0.001)
        ret, frame = self.capture.read()
        self.last_read = time.time()
        if not ret:
            raise StopIteration
        return frame


class ScreenCapture(Capture):
    def __init__(self, top: int, left: int, width: int, height: int):
        import mss
        self.sct = mss.mss()
        self.monitor = {
            'top': top,
            'left': left,
            'width': width,
            'height': height,
        }

    def __del__(self):
        self.sct.close()

    def __next__(self) -> cv2.typing.MatLike:
        frame = self.sct.grab(self.monitor)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
