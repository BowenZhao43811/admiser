# solve_my_ocp_problem_tpl.py
import numpy as np
import matplotlib.pyplot as plt

from admiser import OCPSolver

from my_A_problem_temeplate import problem, N, dt

def main():
    solver = OCPSolver(problem).build()
    res = solver.solve(maxiter=2000, ftol=1e-9, disp=True)

    U_opt  = res["U_opt"]              # shape = (N*nu,) 或 (N,nu) 由你的实现而定
    theta  = res.get("theta_opt", None)
    J_opt  = res["J_opt"]
    X_opt  = res["X_opt"]              # shape = (N+1, nx)
    t_opt  = res["t_opt"]              # shape = (N+1,)
    eq_res = res.get("eq_res", None)   # 若 OCPSolver 返回了等式残差
    in_res = res.get("ineq_res", None) # 若 OCPSolver 返回了不等式残差

    print("\n=== 结果 ===")
    print("J* =", J_opt)
    if theta is not None:
        print("theta* =", theta)
    if eq_res is not None:
        print("eq residual:", eq_res)
    if in_res is not None:
        print("ineq residual:", in_res)

    # 状态
    plt.figure()
    for i in range(X_opt.shape[1]):
        plt.plot(t_opt, X_opt[:, i], label=f"x{i+1}")
    plt.xlabel("t"); plt.ylabel("state"); plt.title("States"); plt.legend()

    # 控制（分段常值 → N 段；若 U_opt 是扁平向量，reshape 一下）
    nu = problem.nu
    U_mat = U_opt.reshape(N, nu) if U_opt.ndim == 1 else U_opt
    tc = np.linspace(0.0, np.sum(np.diff(t_opt)), N)  # 固定dt时等于 np.linspace(0,T,N)
    plt.figure()
    for j in range(nu):
        plt.step(tc, U_mat[:, j], where='pre', label=f"u{j+1}")
    plt.xlabel("t"); plt.ylabel("u"); plt.title("Controls"); plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
