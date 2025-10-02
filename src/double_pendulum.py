import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
import sympy as sp
from sympy.vector import CoordSys3D
from matplotlib import animation


## Set plot design style
plt.style.use("seaborn-v0_8-poster")


def main():
    # Parameters
    params = {
        "m1": 0.0,
        "m2": 10.0,
        "l1": 1.0,
        "l2": 1.0,
        "g": 9.81,
        "m_a1": 0.2,
        "m_a2": 0.2,
    }
    ## Initial conditions: [alpha(0), beta(0), dalpha(0), dbeta(0)]
    y0 = [
        np.pi / 3,
        np.pi / 2,
        0.0,
        0.0,
    ]
    t_span = (0, 10)
    t_eval = np.linspace(t_span[0], t_span[1], 500)
    sol = simulate_double_pendulum(params, y0, t_span, t_eval)
    plot_results(sol.t, sol.y)
    ani = animate_double_pendulum(sol.t, sol.y, params["l1"], params["l2"])
    ani.save("../figures/double_pendulum_simulation.gif", writer="pillow", fps=30)


def get_ode_for_double_pendulum():
    ## symbols
    t, g, m1, m2, l1, l2, m_a1, m_a2 = sp.symbols(
        "t g m1 m2 l1 l2 m_a1 m_a2", real=True, positive=True
    )

    ## Gen coordinates (absolute angles to vertical)
    alpha = sp.Function("alpha")(t)
    beta = sp.Function("beta")(t)
    dalpha = sp.diff(alpha, t)
    dbeta = sp.diff(beta, t)
    ddalpha = sp.diff(alpha, (t, 2))
    ddbeta = sp.diff(beta, (t, 2))

    ## Coordinate system
    N = CoordSys3D("N")

    ## Positions
    x_1 = -l1 * sp.sin(alpha) * N.i - l1 * sp.cos(alpha) * N.j
    x_2 = x_1 - l2 * sp.sin(beta) * N.i - l2 * sp.cos(beta) * N.j

    ## Center of mass positions
    x_cm1 = x_1 / 2  # center of mass of rod 1
    x_cm2 = x_1 + (x_2 - x_1) / 2  # center of mass of rod 2

    ## Velocities
    v_1 = sp.diff(x_1, t)
    v_2 = sp.diff(x_2, t)
    v_cm1 = sp.diff(x_cm1, t)
    v_cm2 = sp.diff(x_cm2, t)

    ## speeds squared
    v_1sq = sp.simplify(v_1.magnitude() ** 2)
    v_2sq = sp.simplify(v_2.magnitude() ** 2)
    v_cm1sq = sp.simplify(v_cm1.magnitude() ** 2)
    v_cm2sq = sp.simplify(v_cm2.magnitude() ** 2)

    ## Moments of inertia for rods about center of mass
    I1_cm = m_a1 * l1**2 / 12
    I2_cm = m_a2 * l2**2 / 12

    ## Kinetic energy
    T_bob1 = sp.Rational(1, 2) * m1 * v_1sq
    T_bob2 = sp.Rational(1, 2) * m2 * v_2sq
    T_rod1 = sp.Rational(1, 2) * m_a1 * v_cm1sq + sp.Rational(1, 2) * I1_cm * dalpha**2
    T_rod2 = sp.Rational(1, 2) * m_a2 * v_cm2sq + sp.Rational(1, 2) * I2_cm * dbeta**2
    T = T_bob1 + T_bob2 + T_rod1 + T_rod2
    ## Note: We do not need to add rotational kinetic energy about the pivot points,
    ## because we are using the velocities of the centers of mass.

    ## Potential energy
    ### Parts: m1, m2, m_a1, m_a2
    V = (
        m1 * g * (x_1.dot(N.j))
        + m2 * g * (x_2.dot(N.j))
        + m_a1 * g * (x_cm1.dot(N.j))  ## center of mass at l1/2
        + m_a2 * g * (x_cm2.dot(N.j))  ## center of mass at l2/2
    )

    L = sp.simplify(T - V)

    ## Define generalized coordinates and their derivatives
    q = sp.Matrix([alpha, beta])
    dq = sp.Matrix([dalpha, dbeta])
    ddq = sp.Matrix([ddalpha, ddbeta])

    ## Lagrange's equations of motion for alpha and beta
    LEOM = sp.simplify(sp.diff(sp.diff(L, dq), t) - sp.diff(L, q))
    ## solve for ddalpha
    sol = sp.solve(LEOM, (ddq), simplify=True, rational=False)

    ## Return a function that can be used for numerical evaluation
    ddalpha_fn = sp.lambdify(
        (alpha, beta, dalpha, dbeta, m1, m2, l1, l2, g, m_a1, m_a2),
        sol[ddalpha],
        "numpy",
    )
    ddbeta_fn = sp.lambdify(
        (alpha, beta, dalpha, dbeta, m1, m2, l1, l2, g, m_a1, m_a2),
        sol[ddbeta],
        "numpy",
    )
    return ddalpha_fn, ddbeta_fn


def simulate_double_pendulum(params, y0, t_span, t_eval):
    """
    Simulate the motion of a single pendulum using the derived ODE.

    Parameters:
    - params: Dictionary containing parameters 'm', 'l1', 'g', and 'm_a1'.
    - y0: Initial conditions [alpha(0), dalpha(0)].
    - t_span: Tuple (t0, tf) for the time span of the simulation.
    - t_eval: Array of time points where the solution is evaluated.

    Returns:
    - sol: Object with the simulation results.
    """
    # Get the symbolic expression for ddalpha
    ddalpha_fn, ddbeta_fn = get_ode_for_double_pendulum()

    # Define the system of first-order ODEs
    def odes(t, y):
        alpha, beta, dalpha, dbeta = y
        return [
            dalpha,
            dbeta,
            ddalpha_fn(alpha, beta, dalpha, dbeta, **params),
            ddbeta_fn(alpha, beta, dalpha, dbeta, **params),
        ]

    # Solve the ODEs
    sol = solve_ivp(
        odes, t_span, y0, t_eval=t_eval, method="Radau"
    )  # Implicit since I chose sort of stiff params
    return sol


def plot_results(t, y):
    plt.figure(figsize=(7, 7))
    plt.plot(t, y[0], label="Alpha (rad)")
    plt.plot(t, y[1], label="Beta (rad)")
    plt.plot(t, y[2], label="Angular Velocity Alpha (rad/s)")
    plt.plot(t, y[3], label="Angular Velocity Beta (rad/s)")
    plt.title("Single Pendulum Simulation")
    plt.xlabel("Time (s)")
    plt.ylabel("Values")
    plt.legend()
    plt.grid()
    plt.show()


def animate_double_pendulum(t, y, l1=1.0, l2=1.0):
    """
    Simple 2D animation of the pendulum.
    t : array of time points
    y : 2 x N array from solve_ivp; y[0] = alpha(t)
    l1: length of the pendulum arm
    """
    alpha = y[0]
    beta = y[1]

    # Point positions
    x1 = -l1 * np.sin(alpha)
    y1 = -l1 * np.cos(alpha)
    x2 = x1 - l2 * np.sin(beta)
    y2 = y1 - l2 * np.cos(beta)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-1.2 * (l1 + l2), 1.2 * (l1 + l2))
    ax.set_ylim(-1.2 * (l1 + l2), 1.2 * (l1 + l2))
    ax.set_aspect("equal")
    ax.grid(False)
    ax.axis("off")

    (line,) = ax.plot([], [], "o-", lw=2)
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes, fontsize=28)

    def init():
        line.set_data([], [])
        time_text.set_text("")
        return line, time_text

    def update(i):
        line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
        time_text.set_text(f"t = {t[i]:.2f} s")
        return line, time_text

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(t),
        init_func=init,
        blit=True,
        interval=1000 * (t[1] - t[0]),
    )
    plt.show()
    return ani


if __name__ == "__main__":
    main()
