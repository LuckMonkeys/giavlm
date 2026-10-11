"""Public visual nuisance weighting for experimental first-layer filtering."""
import torch


def visual_weight(values, basis, eigenvalues, relative_ridge):
    """Right multiply by the square root of a regularized residual metric.

    basis rows are orthonormal public visual directions; eigenvalues describe
    their uncentered second moment. Minimizing over a ridge-penalized visual
    coefficient matrix gives this metric on the remaining text residual.
    A zero ridge is an explicitly requested hard projection control.
    """
    if relative_ridge < 0:
        raise ValueError('Visual ridge must be nonnegative')
    if relative_ridge == 0:
        weights = torch.zeros_like(eigenvalues)
    else:
        ridge = relative_ridge * eigenvalues[0]
        weights = (ridge / (eigenvalues + ridge)).sqrt()
    return values - ((values @ basis.T) * (1 - weights)) @ basis
