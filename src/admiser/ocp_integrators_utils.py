# ocp_integrators_utils.py
"""
子步 RK4 积分器，以及沿轨迹求积（quadrature）的方案族。

背景
----
控制参数化把每一段 [t_k, t_k+dt) 上的控制固定为常值，段内用多子步 RK4 把状态
往前推。除了状态之外，我们还需要沿着同一条轨迹算若干个积分：

    目标函数     ∫ L(t,x,u,θ) dt
    积分型约束   ∫ q(t,x,u,θ) dt
    路径约束转译 ∫ L_ε(h(t,x,u,θ)) dt

这些积分**不反馈到状态**，所以可以在推进状态的同时顺带累加。做法是由积分器在
若干采样点上回调 accumulate_cb(t_i, x_i, u, w_i)，调用方在回调里累加
w_i * 被积函数，全部子步累加完即得 ∫。

两个独立的精度
--------------
状态本身始终由经典 RK4 推进，是 4 阶的。但**积分的精度由采样方案单独决定**，
两者并不自动一致：早期版本用欧拉半步中点采样，状态 4 阶而积分只有 2 阶。

一个采样方案的阶数受两件事同时限制：

  1. 求积法则本身的阶（矩形 1 阶、中点/梯形 2 阶、Simpson 4 阶……）
  2. **各采样点上状态估计的精度**

第 2 条常被忽略，却往往才是瓶颈。例如 Simpson 法则本身是 4 阶的，但若中点状态
只取欧拉半步 x+½h·k1（误差 O(h²)），整体就退回 2 阶——白白多做两次被积函数
求值却一分精度都没买到（见下表 simpson 一行的说明）。

设计约束：零额外动力学求值
--------------------------
下面所有方案都只复用 RK4 已经算出的 k1..k4 与由它们构成的中间状态，**不额外
求值 f**。代价只是被积函数 g 的求值次数（见各方案的 n_eval），它直接决定 AD
tape 的规模，因此高阶方案的开销是"tape 变大"而不是"多解一次 ODE"。

方案族（阶数均为实测值，测试问题 x'=x, g=t²x, ∫₀¹t²eᵗdt = e-2）
-----------------------------------------------------------------
    名称          阶   n_eval  采样点
    'left'         1     1     (t,     x)                     左矩形
    'right'        1     1     (t+h,   x_end)                 右矩形
    'midpoint'     2     1     (t+h/2, x+½h·k1)               中点法则
    'trapezoid'    2     2     两端点，权重 h/2, h/2           梯形法则
    'simpson'      3     3     (t,x), (t+h/2, x+¼h(k1+k2)), (t+h, x_end)
    'rk4'          4     4     RK4 的四个级点，权重 h/6,h/3,h/3,h/6

QUAD_SCHEMES 是唯一的方案名来源，不设别名：方案名在全package范围内只有这一套
写法，拼错或用了旧写法都会立即报错，而不是被静默接受成另一种精度。

关于 'simpson' 的中点：取两个半步估计的平均 ½[(x+½h·k1)+(x+½h·k2)] = x+¼h(k1+k2)。
这两个估计对真实中点的误差恰为 ∓⅛h²·f'f，相加时首阶误差相消，因而该平均点是
O(h³) 精度，Simpson 整体达到 3 阶。若中点改用 x+½h·k1 或 x+½h·k2 中的任何单独
一个，整体都只有 2 阶（实测 1.99 / 1.98）——这就是上面第 2 条限制的直接体现。

关于 'rk4'：等价于把被积函数当作增广状态 y' = g(t,x,u) 与 x 一起用同一套 RK4
积分。由于 RK4 对任意光滑 ODE 系统都是 4 阶的，∫g dt 自动与状态同阶。注意第四
个采样点必须是 RK4 第 4 级的自变量 x+h·k3，**而不是**步末状态 x_end；换成 x_end
会破坏 RK4 的阶条件，实测掉到 3 阶。
"""

from typing import NamedTuple

import numpy as np


class QuadScheme(NamedTuple):
    """一个求积方案的元信息（阶数为实测值，n_eval 决定 AD tape 规模）。"""
    order: int      # 全局收敛阶
    n_eval: int     # 每个子步对被积函数的求值次数
    summary: str    # 一句话说明


#: 全部规范方案名 -> 元信息。想知道某个名字是几阶，用 quad_order(name)。
QUAD_SCHEMES = {
    'left':      QuadScheme(1, 1, "左矩形：子步左端点"),
    'right':     QuadScheme(1, 1, "右矩形：子步右端点（步末状态）"),
    'midpoint':  QuadScheme(2, 1, "中点法则：欧拉半步中点 x+½h·k1"),
    'trapezoid': QuadScheme(2, 2, "梯形法则：两端点，权重 h/2, h/2"),
    'simpson':   QuadScheme(3, 3, "Simpson 法则：中点取 x+¼h(k1+k2)（两半步估计的平均）"),
    'rk4':       QuadScheme(4, 4, "RK4 增广状态求积：四个级点，权重 h/6, h/3, h/3, h/6"),
}

def validate_quad_scheme(name: str) -> str:
    """校验求积方案名并原样返回；不认识的名字直接报错。"""
    if name not in QUAD_SCHEMES:
        raise ValueError(
            f"unknown quad scheme {name!r}; expected one of {tuple(QUAD_SCHEMES)}. "
            "（方案名拼错会导致积分项被静默跳过，因此这里直接报错。）"
        )
    return name


def quad_order(name: str) -> int:
    """该求积方案的全局收敛阶。"""
    return QUAD_SCHEMES[validate_quad_scheme(name)].order


def _shift(t, delta):
    """时间偏移；t0 为 None（调用方不需要时间）时保持 None。"""
    return None if t is None else t + delta


def _quad_samples(scheme, t, h, x, k1, k2, k3, x_end):
    """
    给出某个子步 [t, t+h] 上的采样点列表 [(t_i, x_i, w_i), ...]，
    调用方累加 ∑ w_i * g(t_i, x_i) 即得该子步上的 ∫g dt 近似。

    不变量：∑ w_i == h（每个方案都是对同一段长度 h 做求积）。
    入参 k1..k3 与 x_end 均由 RK4 步进过程免费提供，此处不再求值 f。
    """
    t_mid = _shift(t, 0.5 * h)
    t_end = _shift(t, h)

    if scheme == 'left':
        return ((t, x, h),)

    if scheme == 'right':
        return ((t_end, x_end, h),)

    if scheme == 'midpoint':
        # 欧拉半步中点：误差 O(h²)，配中点法则得 2 阶
        return ((t_mid, x + 0.5 * h * k1, h),)

    if scheme == 'trapezoid':
        return ((t,     x,     0.5 * h),
                (t_end, x_end, 0.5 * h))

    if scheme == 'simpson':
        # 两个半步估计的平均，首阶误差相消 -> O(h³) 的中点，整体 3 阶
        x_mid = x + 0.25 * h * (k1 + k2)
        return ((t,     x,     h / 6.0),
                (t_mid, x_mid, 4.0 * h / 6.0),
                (t_end, x_end, h / 6.0))

    if scheme == 'rk4':
        # RK4 的四个级点。末点是第 4 级的自变量 x+h·k3，不是步末状态 x_end
        return ((t,     x,                 h / 6.0),
                (t_mid, x + 0.5 * h * k1,  h / 3.0),
                (t_mid, x + 0.5 * h * k2,  h / 3.0),
                (t_end, x + h * k3,        h / 6.0))

    raise ValueError(f"unhandled quad scheme {scheme!r}")   # pragma: no cover


def rk4_step(x, u, h, dyn):
    """
    单步经典四阶 RK（不带求积回调，供独立使用）：
        x_{k+1} = x_k + h/6 * (k1 + 2k2 + 2k3 + k4)

    x, u: 可以是 float / numpy.array / 包含 cppad_py.a_double 的数组
    dyn:  用户给的动力学函数 dyn(x,u) -> xdot
    """
    k1 = dyn(x, u)
    k2 = dyn(x + 0.5 * h * k1, u)
    k3 = dyn(x + 0.5 * h * k2, u)
    k4 = dyn(x + h * k3,       u)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_substeps(x, u, dt, f, m_sub=5, accumulate_cb=None, t0=None, quad='rk4'):
    """
    多子步 RK4 积分器，可选地在子步内按指定求积方案回调，用于沿轨迹累加积分。

    参数
    ----
    x : ndarray (nx,)
        段初状态，元素可为 float 或 cppad_py.a_double
    u : ndarray (nu,)
        本段的（分段常值）控制
    dt : float | a_double
        本段总时长
    f : callable
        f(x, u) -> dx/dt
    m_sub : int
        子步数；本段被均分为 m_sub 个长 h = dt/m_sub 的子步
    accumulate_cb : callable | None
        accumulate_cb(t_i, x_i, u, w_i)；调用方在回调里累加 w_i * 被积函数。
        为 None 时完全跳过求积（纯状态推进，例如求解后的数值复演）。
        注意第 4 个参数是**求积权重**而不是子步长：一个子步内可能被回调多次，
        权重之和恒等于 h。
    t0 : float | a_double | None
        本段起点时间。为 None 时回调收到的 t_i 也是 None（被积函数不依赖 t 时可用）
    quad : str
        求积方案名，见 QUAD_SCHEMES。默认 'rk4'（4 阶，与状态同阶）

    返回
    ----
    x_{k+1} : 段末状态

    备注
    ----
    无论选哪个求积方案，状态都由完整的 RK4 推进，因此**段末状态与 quad 无关**；
    quad 只影响 ∫ 的精度。各方案都只复用 k1..k4，不额外求值 f。
    """
    scheme = validate_quad_scheme(quad)

    h = dt / m_sub
    t = t0

    for _ in range(m_sub):
        k1 = f(x, u)
        k2 = f(x + 0.5 * h * k1, u)
        k3 = f(x + 0.5 * h * k2, u)
        k4 = f(x + h * k3,       u)

        x_end = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if accumulate_cb is not None:
            for t_i, x_i, w_i in _quad_samples(scheme, t, h, x, k1, k2, k3, x_end):
                accumulate_cb(t_i, x_i, u, w_i)

        x = x_end
        t = _shift(t, h)

    return x
