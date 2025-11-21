# smooth_utils.py
import numpy as np
import cppad_py

def _to_ad(x):
    if isinstance(x, cppad_py.a_double):
        return x
    return cppad_py.a_double(float(x))

def smooth_abs(h, eps_abs: float):
    """|h| 的光滑近似：sqrt(h^2 + eps^2)"""
    h_ad = _to_ad(h)
    e_ad = _to_ad(eps_abs)
    return np.sqrt(h_ad * h_ad + e_ad * e_ad)

def L_eps(h, eps: float):
    """
    平滑 hinge L_eps(h) ≈ max(h, 0)
    三段式：
      h <= -eps  -> 0
      h >= +eps  -> h
      else       -> (h+eps)^2 / (4 eps)
    用 cond_assign 实现，避免 if/else 破坏 AD tape。
    """
    h_ad  = _to_ad(h)
    eps_ad= _to_ad(eps)
    zero  = _to_ad(0.0)
    four  = _to_ad(4.0)

    mid = ((h_ad + eps_ad) * (h_ad + eps_ad)) / (four * eps_ad)  # (h+eps)^2/(4eps)

    # 先做 (h >= eps) ? h : mid
    mid_or_h = cppad_py.a_double()
    mid_or_h.cond_assign(">=", h_ad, eps_ad, h_ad, mid)
    # 再做 (h <= -eps) ? 0 : (mid_or_h)
    out = cppad_py.a_double()
    out.cond_assign("<=", h_ad, -eps_ad, zero, mid_or_h)
    return out
