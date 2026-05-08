from src.timevae_wrapper_gpu import TimeVAEWrapper as _SharedTimeVAEWrapper


class TimeVAEWrapper(_SharedTimeVAEWrapper):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("device", "cpu")
        kwargs.setdefault("require_gpu", False)
        super().__init__(*args, **kwargs)
