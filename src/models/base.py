import cv2
import typing
import onnxruntime as ort


__all__ = [
    'OnnxModel',
]


PROVIDERS_GPU = ['CUDAExecutionProvider', 'CPUExecutionProvider']
PROVIDERS_CPU = ['CPUExecutionProvider']


T_IN = typing.TypeVar('T_IN')
T_OUT = typing.TypeVar('T_OUT')


class OnnxModel(typing.Generic[T_IN, T_OUT]):
    def __init__(self, model_path, cuda=False):
        self.model_path = model_path
        self.cuda = cuda
        self.providers = PROVIDERS_GPU if cuda else PROVIDERS_CPU
        self.session = ort.InferenceSession(
            model_path,
            providers=self.providers
        )
        self.outname = [i.name for i in self.session.get_outputs()]
        self.inname = [i.name for i in self.session.get_inputs()]

    def predict(self, input: T_IN) -> T_OUT:
        raise NotImplementedError
