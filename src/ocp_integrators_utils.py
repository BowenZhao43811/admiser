# integrators.py
import numpy as np

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


def rk4_substeps(x, u, dt, f, m_sub=5, accumulate_cb=None, t0=None, quad='rk4-mid'):
    """
    多子步 RK4 积分器，支持每个子步的积分回调 accumulate_cb：
      accumulate_cb(t_sub, x_sub, u_sub, h_sub)   # 用户在回调里做 ∑(h_sub * 被积函数)
    参数：
      x : ndarray (nx,)  元素可为 float 或 cppad_py.a_double
      u : ndarray (nu,)
      dt: 本段总时长
      f : dyn(x,u) -> dx/dt  （与现有一致）
      m_sub: 子步数
      accumulate_cb: 可选回调；若为 None 则不做子步积累
      t0 : 当前段起点时间（float 或 a_double，若 None 则不传时间给回调）
      quad: 采样点
        - 'left'    : 用子步左端 (t, x)
        - 'right'   : 用子步右端 (t+h, x_{k+1})
        - 'mid'     : 用欧拉半步 (t+0.5h, x + 0.5h*k1)
        - 'rk4-mid' : 用 RK4 的 k1 得到的中点 (等同于 'mid'，命名提示用意)
    返回：
      x_{k+1}
    """
    h = dt / m_sub
    t = t0

    for _ in range(m_sub):
        # k1
        k1 = f(x, u)

        # —— 子步采样并回调 —— #
        if accumulate_cb is not None:
            if quad in ('mid', 'rk4-mid'):
                t_samp = (t + 0.5*h) if (t is not None) else None
                x_samp = x + 0.5*h*k1
                accumulate_cb(t_samp, x_samp, u, h)
            elif quad == 'left':
                t_samp = t
                x_samp = x
                accumulate_cb(t_samp, x_samp, u, h)
            # 'right' 留到步进后回调

        # k2, k3, k4
        k2 = f(x + 0.5*h*k1, u)
        k3 = f(x + 0.5*h*k2, u)
        k4 = f(x + h*k3,     u)

        # 更新状态
        x = x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        if accumulate_cb is not None and quad == 'right':
            t_samp = (t + h) if (t is not None) else None
            x_samp = x
            accumulate_cb(t_samp, x_samp, u, h)

        if t is not None:
            t = t + h

    return x