from __future__ import annotations

import dataclasses
import numpy as np
import random
import typing

import cv2

from src.models.base import OnnxModel
from src.utils import letterbox


__all__ = [
    "ObjectDetectionModel",
    "ObjectDetectionModelOutput",
]


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
        names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                'hair drier', 'toothbrush']
        colors = {name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(names)}
        image = self.image.copy()
        for batch in self.batches:
            box = (np.array([batch.x0, batch.y0, batch.x1, batch.y1]) - np.array(self.dwdh*2)) / self.ratio
            box = box.round().astype(np.int32).tolist()
            name = names[int(batch.cls_id)]
            color = colors[name]
            cv2.rectangle(image, box[:2], box[2:], color, 2)
            cv2.putText(image, f'{name} {batch.score}', (box[0], box[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=2)
        return image


class ObjectDetectionModel(OnnxModel[cv2.typing.MatLike, ObjectDetectionModelOutput]):
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
