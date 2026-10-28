"""An ES step read as a displacement from the population's centroid to its centre of mass."""

from .com import (
    as_masses,
    center_of_mass,
    centroid,
    displacement_update,
    es_update,
    identity_residual,
    perturb,
    reference_point,
)
from .sampling import SAMPLERS, antithetic, plain, sample
from .weights import WEIGHTINGS, centred_rank, mean_subtracted, raw, standardised, sums_to_zero

__version__ = "0.1.0"
__all__ = [
    "SAMPLERS",
    "WEIGHTINGS",
    "antithetic",
    "as_masses",
    "center_of_mass",
    "centroid",
    "centred_rank",
    "displacement_update",
    "es_update",
    "identity_residual",
    "mean_subtracted",
    "perturb",
    "plain",
    "raw",
    "reference_point",
    "sample",
    "standardised",
    "sums_to_zero",
]
