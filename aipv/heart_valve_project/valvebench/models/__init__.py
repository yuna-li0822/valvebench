from .baseline_linear import BaselineLinear
from .baseline_mlp import BaselineMLP
from .gbm import GradientBoostingBaseline, MonotoneGradientBoostingBaseline
from .pinn_model import PINNModel
from .ridge import RidgeRegression

# Two models are intentionally not imported/registered here:
# - pinn_scaffold.ScaffoldPINN (the headline PINNVALVE artifact) isn't part
#   of the generic MODEL_REGISTRY benchmark contract (it takes scaffold
#   design params + time, not the patient FEATURE_COLUMNS -> TARGET_COLUMNS
#   shape). Import it directly from valvebench.models.pinn_scaffold.
# - literature_regression.LiteratureRegressionBaseline requires real
#   published coefficients as constructor arguments, which this project
#   doesn't have verified access to (see that module's docstring) -- it
#   can't satisfy MODEL_REGISTRY's no-arg `model_cls()` construction
#   pattern, and shouldn't silently fall back to fabricated numbers.
#   Import it directly from valvebench.models.literature_regression once
#   you have real coefficients to supply.

__all__ = [
    "BaselineLinear",
    "BaselineMLP",
    "GradientBoostingBaseline",
    "MonotoneGradientBoostingBaseline",
    "PINNModel",
    "RidgeRegression",
]
