"""Models package for valve_project.

Only `forwardmodel` (ForwardMLP) is still used, via
heart_valve_project/models/forwardmodel.py's re-export -- see that module's
docstring. `baseline_mlp.py` and `pinn_model.py` (real duplicates of the
canonical, tested `valvebench.models` implementations, one of them with a
broken import) were removed as dead code.
"""
