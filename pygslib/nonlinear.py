"""Local trial of the Rust CDF and Hermite API.

This replaces the legacy function-based nonlinear API with the native
``gslib_rust_cdf`` classes, enums, and functions.
"""

from gslib_rust_cdf import (
    EcdfContinous,
    HermitePolynomials,
    OrdRelResult,
    TailType,
    VarType,
    distrib_beyond_prob,
    distrib_beyond_zval,
    gslib_distrib_locate,
    gslib_distrib_powint,
    gslib_hyperbolic_cdf,
    gslib_hyperbolic_zval,
    gslib_sort_index_copy,
    mean_variance,
)

__all__ = [
    "EcdfContinous",
    "HermitePolynomials",
    "OrdRelResult",
    "TailType",
    "VarType",
    "distrib_beyond_prob",
    "distrib_beyond_zval",
    "gslib_distrib_locate",
    "gslib_distrib_powint",
    "gslib_hyperbolic_cdf",
    "gslib_hyperbolic_zval",
    "gslib_sort_index_copy",
    "mean_variance",
]
