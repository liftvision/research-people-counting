from __future__ import annotations

import dataclasses
import random
import typing

import cv2
import numpy as np

from src.models.base import OnnxModel


__all__ = [
    "ObjectDetectionModel",
    "ObjectDetectionModelOutput",
    "ObjectDetectionModelOutput",
]


random.seed(0)


@dataclasses.dataclass(frozen=True)
class ObjectDetectionModelOutput:
    image: cv2.typing.MatLike
    ratio: float
    dwdh: typing.Tuple[int]
    batches: typing.Tuple[Batch]

    @dataclasses.dataclass(frozen=True)
    class Batch:
        batch_id: int
        x0: int
        y0: int
        x1: int
        y1: int
        cls_id: int
        score: float

    def visualize(self) -> cv2.typing.MatLike:
        image = self.image.copy()
        for batch in self.batches:
            box = (np.array([batch.x0, batch.y0, batch.x1,
                   batch.y1]) - np.array(self.dwdh*2)) / self.ratio
            box = box.round().astype(np.int32).tolist()
            name = ObjectDetectionModel.NAMES[int(batch.cls_id)]
            color = ObjectDetectionModel.COLORS[name]
            cv2.rectangle(image, box[:2], box[2:], color, 2)
            cv2.putText(image, f'{name} {batch.score}', (box[0], box[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=2)
        return image


class ObjectDetectionModel(OnnxModel[cv2.typing.MatLike, ObjectDetectionModelOutput]):
    NAMES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush',
    ]
    COLORS = {name: [random.randint(0, 255) for _ in range(3)] for name in NAMES}

    def __init__(self, model_path='yolov7-tiny.onnx', cuda=False):
        super().__init__(model_path, cuda)

    def predict(self, image: cv2.typing.MatLike) -> ObjectDetectionModelOutput:
        original_image = image
        image, ratio, dwdh = self._preprocess(image)
        inp = {self.inname[0]: self._normalize(image)}
        # ONNX inference
        outputs = self.session.run(self.outname, inp)[0]
        return ObjectDetectionModelOutput(
            image=original_image,
            ratio=ratio,
            dwdh=tuple(dwdh),
            batches=(
                ObjectDetectionModelOutput.Batch(
                    batch_id=batch_id,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    cls_id=cls_id,
                    score=score,
                ) for batch_id, x0, y0, x1, y1, cls_id, score in outputs
            )
        )

    def _preprocess(self, img: cv2.typing.MatLike) -> cv2.typing.MatLike:
        """Preprocess image for inference
        """
        image = img.copy()
        image, ratio, dwdh = letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        return image, ratio, dwdh

    def _normalize(self, image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        image = image.astype(np.float32)
        image /= 255
        return image


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)
