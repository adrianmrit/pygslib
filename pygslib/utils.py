import matplotlib.pyplot as plt
import numpy as np

from pygslib.nonlinear import EcdfContinous, HermitePolynomials, TailType


def gcos(table, vardhole, varmodel, cutoff, model, cbb, decwt = None,
        ltail=TailType.Linear, utail=TailType.Hyperbolic, ltpar=1, utpar=1.5, K=30,
        ymin=None, ymax=None, ndisc = 1000):
    """
    gcos(table, mask, vardhole, varmodel, cutoff, model, cbb, decwt = None)
    
    Produce global change of support validation (GCOS)
    
    Inputs
    ------
    table: pandas dataframe
        drillhole database
    vardhole: str
        variable on drillhole
    varmodel:
        variable in bmodel
    cutoff:
        cutoff list
    model: pandas dataframe
        block model 
    cbb:
        block variance
    decwt: str default None
        declustering weight variable name
    ltail, utail: TailType
        Lower and upper tail models (default Linear and Hyperbolic).
    ltpar, utpar: floats
        Tail parameters (default 1 and 1.5).
    K: integer
        number of polynomials. Default is 30
    ymin, ymax:
        Must be None. The Rust model infers its authorized interval.
    ndisc: int 
        discretization intervals in cdf. Default is 1000
    """
    
    if ymin is not None or ymax is not None:
        raise ValueError("Rust Hermite models do not support ymin/ymax overrides")

    zin = table[vardhole].to_numpy(dtype=float)
    w = np.ones(zin.size) if decwt is None else table[decwt].to_numpy(dtype=float)
    cdf = EcdfContinous.new_from_data(
        zin, w, float(zin.min()), ltail, ltpar, TailType.Linear, 1.,
        float(zin.max()), utail, utpar, 0., vardhole,
    )
    hermite = HermitePolynomials.new_from_cdf(cdf, ndisc, K, 1., 1., 1.)
    r = hermite.get_r(cbb)
    hermite.set_support(r, r, 1.)
    _, block_grades = hermite.y2z_block(hermite.gaussians)
    block_cdf = EcdfContinous.new_from_ik_cdf(
        block_grades, hermite.percentiles, float(block_grades.min()),
        ltail, ltpar, TailType.Linear, 1., float(block_grades.max()),
        utail, utpar, f"DGM {vardhole}",
    )
    curves = [(f"DGM {vardhole}", block_cdf)]
    for v in varmodel:
        values = model[v].dropna().to_numpy(dtype=float)
        if values.size == 0:
            raise ValueError(f"Block model column {v!r} has no non-null values")
        estimate_cdf = EcdfContinous.new_from_data(
            values, np.ones(values.size), float(values.min()), ltail, ltpar,
            TailType.Linear, 1., float(values.max()), utail, utpar, 0., v,
        )
        curves.append((f"Estimate {v}", estimate_cdf))

    results = [(label, distribution.grade_tonnage(cutoff, ndisc))
               for label, distribution in curves]
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('cutoff')
    ax1.set_ylabel('Tonnage')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Grade')
    for label, (tonnage, above, _) in results:
        ax1.plot(cutoff, tonnage, label=label)
        ax2.plot(cutoff, above)
    ax1.legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)
    fig.tight_layout()
    plt.show()
    return fig
