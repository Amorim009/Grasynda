from src.tsdiff_wrapper_gpu import TSDiffWrapper as _TSDiffWrapperGPU


class TSDiffWrapper(_TSDiffWrapperGPU):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("device", "cpu")
        kwargs.setdefault("require_gpu", False)
        super().__init__(*args, **kwargs)
