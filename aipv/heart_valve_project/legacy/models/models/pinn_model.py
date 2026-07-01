"""PINN model wrapper that delegates to the moved forwardmodel implementation."""
from typing import Any, Optional
try:
    from valve_project.models import forwardmodel as forwardmodel_module
except Exception:
    forwardmodel_module = None


class PINNModel:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._model = None
        if forwardmodel_module is not None:
            self._model = forwardmodel_module

    def predict(self, inputs: Any):
        if self._model is None:
            raise RuntimeError("forwardmodel module not available")
        # prefer a callable ForwardMLP instance if present
        if hasattr(self._model, "ForwardMLP"):
            model_cls = getattr(self._model, "ForwardMLP")
            m = model_cls()
            return m(inputs)
        if hasattr(self._model, "generate_curve"):
            return self._model.generate_curve(*inputs)
        raise RuntimeError("No prediction entrypoint found in forwardmodel")
