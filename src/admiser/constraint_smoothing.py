# constraint_smoothing.py
import numpy as np
import cppad_py

def _to_ad(x):
    if isinstance(x, cppad_py.a_double):
        return x
    return cppad_py.a_double(float(x))

def smooth_abs(h, eps_abs: float):
    """Smooth approximation of |h|: sqrt(h^2 + eps^2)."""
    h_ad = _to_ad(h)
    e_ad = _to_ad(eps_abs)
    return np.sqrt(h_ad * h_ad + e_ad * e_ad)

def L_eps(h, eps: float):
    """
    Smoothed hinge L_eps(h) ~ max(h, 0), in three pieces:

      h <= -eps  -> 0
      h >= +eps  -> h
      otherwise  -> (h + eps)^2 / (4*eps)

    Implemented with cond_assign so the comparison is recorded as a real CppAD
    conditional expression. A plain Python if/else would bake the branch taken at
    taping time into the tape and silently give wrong derivatives elsewhere.
    """
    h_ad  = _to_ad(h)
    eps_ad= _to_ad(eps)
    zero  = _to_ad(0.0)
    four  = _to_ad(4.0)

    mid = ((h_ad + eps_ad) * (h_ad + eps_ad)) / (four * eps_ad)  # (h+eps)^2 / (4*eps)

    # first: (h >= eps) ? h : mid
    mid_or_h = cppad_py.a_double()
    mid_or_h.cond_assign(">=", h_ad, eps_ad, h_ad, mid)
    # then:  (h <= -eps) ? 0 : mid_or_h
    out = cppad_py.a_double()
    out.cond_assign("<=", h_ad, -eps_ad, zero, mid_or_h)
    return out
