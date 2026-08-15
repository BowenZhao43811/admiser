# integrators.py
import numpy as np

#: 支持的子步求积（采样）模式。
#:   'rk4'     : 把被积函数当作增广状态、用同一套 RK4 权重积分（4 阶，推荐）
#:   'mid'     : 子步中点，欧拉半步取样（2 阶）
#:   'rk4-mid' : 'mid' 的别名，保留以兼容旧问题文件
#:   'left'    : 子步左端点（1 阶）
#:   'right'   : 子步右端点（1 阶）
QUAD_MODES = ('rk4', 'mid', 'rk4-mid', 'left', 'right')


def rk4_step(x, u, h, dyn):
    """
    经典四阶RK:
        x_{k+1} = x_k + h/6 * (k1 + 2k2 + 2k3 + k4)

    x, u: 可以是 float / numpy.array / 包含 cppad_py.a_double 的数组
    dyn:  用户给的动力学函数 dyn(x,u) -> xdot
    """
    k1 = dyn(x, u)
    k2 = dyn(x + 0.5 * h * k1, u)
    k3 = dyn(x + 0.5 * h * k2, u)
    k4 = dyn(x + h * k3,       u)
    return x + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def rk4_substeps(x, u, dt, f, m_sub=5, accumulate_cb=None, t0=None, quad='rk4'):
    """
    多子步 RK4 积分器，支持每个子步的积分回调 accumulate_cb：
      accumulate_cb(t_samp, x_samp, u, w)   # 用户在回调里做 ∑(w * 被积函数)

    注意回调的第 4 个参数是**求积权重**而不是子步长：一个子步内可能被回调多次，
    权重之和恒等于子步长 h，因此 ∑ w * g(t_samp, x_samp) ≈ ∫ g dt。

    参数：
      x : ndarray (nx,)  元素可为 float 或 cppad_py.a_double
      u : ndarray (nu,)
      dt: 本段总时长
      f : dyn(x,u) -> dx/dt  （与现有一致）
      m_sub: 子步数
      accumulate_cb: 可选回调；若为 None 则不做子步积累
      t0 : 当前段起点时间（float 或 a_double，若 None 则不传时间给回调）
      quad: 采样/求积模式，见 QUAD_MODES
        - 'rk4'     : 在 RK4 的四个级点上按权重 (h/6, h/3, h/3, h/6) 回调。
                      等价于把被积函数作为增广状态 y' = g(t,x,u) 用同一套 RK4 积分，
                      因此 ∫g dt 与状态同为 4 阶精度；当 g 只依赖 t 时退化为 Simpson 法则。
                      不增加任何 f 的求值次数。
        - 'mid'     : 用欧拉半步 (t+0.5h, x + 0.5h*k1)，2 阶
        - 'rk4-mid' : 'mid' 的别名（历史命名，实际就是欧拉半步中点法）
        - 'left'    : 用子步左端 (t, x)，1 阶
        - 'right'   : 用子步右端 (t+h, x_{k+1})，1 阶
    返回：
      x_{k+1}
    """
    if quad not in QUAD_MODES:
        raise ValueError(
            f"unknown quad mode {quad!r}; expected one of {QUAD_MODES}. "
            "（求积模式拼错会导致积分项被静默跳过，因此这里直接报错。）"
        )

    h = dt / m_sub
    t = t0

    for _ in range(m_sub):
        # k1
        k1 = f(x, u)

        # —— 步进前的采样点回调 —— #
        if accumulate_cb is not None:
            if quad == 'left':
                accumulate_cb(t, x, u, h)
            elif quad in ('mid', 'rk4-mid'):
                t_samp = (t + 0.5*h) if (t is not None) else None
                accumulate_cb(t_samp, x + 0.5*h*k1, u, h)
            # 'rk4' 需要 k2/k3，'right' 需要步进结果，都留到下面

        # k2, k3, k4
        k2 = f(x + 0.5*h*k1, u)
        k3 = f(x + 0.5*h*k2, u)
        k4 = f(x + h*k3,     u)

        if accumulate_cb is not None and quad == 'rk4':
            # 与状态同阶的求积：四个 RK4 级点，权重 h/6, h/3, h/3, h/6
            t_mid = (t + 0.5*h) if (t is not None) else None
            t_end = (t + h)     if (t is not None) else None
            accumulate_cb(t,     x,             u, h/6.0)
            accumulate_cb(t_mid, x + 0.5*h*k1,  u, h/3.0)
            accumulate_cb(t_mid, x + 0.5*h*k2,  u, h/3.0)
            accumulate_cb(t_end, x + h*k3,      u, h/6.0)

        # 更新状态
        x = x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        if accumulate_cb is not None and quad == 'right':
            t_samp = (t + h) if (t is not None) else None
            accumulate_cb(t_samp, x, u, h)

        if t is not None:
            t = t + h

    return x
