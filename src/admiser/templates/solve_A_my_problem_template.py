# solve_A_my_problem_template.py -- a minimal driver: solve, report, plot
import numpy as np
import matplotlib.pyplot as plt

from admiser import OCPSolver

from my_A_problem_temeplate import problem, N, dt

def main():
    # The only solve entry point. Whether this is a single solve or an eps -> 0
    # continuation is decided by problem.set_transcription(...) in the problem
    # file, so nothing here has to change.
    res = OCPSolver(problem).solve(maxiter=2000, ftol=1e-9)

    U_opt  = res["U_opt"]                # shape = (N*nu,)
    theta  = res.get("theta_opt", None)
    J_opt  = res["J_opt"]
    X_opt  = res["X_opt"]                # shape = (N+1, nx)
    t_opt  = res["t_opt"]                # shape = (N+1,)
    tau    = res.get("tau_opt", None)    # segment durations, when CPET is enabled
    T_opt  = res.get("T_opt", None)      # optimised horizon sum(tau), when CPET is enabled
    eq_res = res.get("eq_resid", None)   # equality residual G(z), should be ~ 0
    in_res = res.get("ineq_resid", None) # inequality residual C(z), should be >= 0
    viol   = res.get("path_viol", None)  # max h(t) on the grid per path constraint, should be <= 0

    print("\n=== results ===")
    print("J* =", J_opt)
    if theta is not None:
        print("theta* =", theta)
    if T_opt is not None:
        print("T*     =", T_opt, " (optimised horizon)")
        print("tau*   =", tau)
    if eq_res is not None:
        print("eq residual:", eq_res)
    if in_res is not None:
        print("ineq residual:", in_res)
    if viol is not None:
        print("max h(t) per path constraint:", viol)
    if len(res.get("rounds", [])) > 1:      # continuation: eps/gamma and violation per round
        for i, r in enumerate(res["rounds"]):
            print(f"  [{i+1}] eps={min(r['eps']):.1e}  gamma={min(r['gamma']):.3e}  "
                  f"J={r['J_opt']:+.10g}  max h(t)={r['max_path_viol']:+.3e}")

    # states
    plt.figure()
    for i in range(X_opt.shape[1]):
        plt.plot(t_opt, X_opt[:, i], label=f"x{i+1}")
    plt.xlabel("t"); plt.ylabel("state"); plt.title("States"); plt.legend()

    # controls (piecewise constant over N segments)
    # t_opt already carries the true knot positions -- uniform without CPET,
    # non-uniform with it -- so use it rather than rebuilding a linspace.
    nu = problem.nu
    U_mat = U_opt.reshape(N, nu) if U_opt.ndim == 1 else U_opt
    tc = t_opt[:-1]
    plt.figure()
    for j in range(nu):
        plt.step(tc, U_mat[:, j], where='post', label=f"u{j+1}")
    plt.xlabel("t"); plt.ylabel("u"); plt.title("Controls"); plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
