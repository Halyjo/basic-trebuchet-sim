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
        "m": 1.0,  # mass of the pendulum bob (kg)
        "l1": 1.0,  # length of the pendulum arm (m)
        "g": 9.81,  # acceleration due to gravity (m/s^2)
        "m_a1": 0.1,  # mass of the pendulum arm (kg)
    }
    # Initial conditions: [start angle, start angular velocity]
    y0 = [np.pi / 4, 0.0]  # 45 degrees, 0 initial angular velocity
    t_span = (0, 10)  # from 0 to 10 seconds
    t_eval = np.linspace(t_span[0], t_span[1], 300)
    sol = simulate_single_pendulum(params, y0, t_span, t_eval)
    plot_results(sol.t, sol.y)
    ani = animate_pendulum(sol.t, sol.y, 1.0)
    ani.save("../figures/single_pendulum_simulation.gif", writer="pillow", fps=30)


def get_ode_for_single_pendulum():
    ## symbols
    t, g, m, l1, m_a1 = sp.symbols("t g m l1 m_a1", real=True, positive=True)

    ## Gen coordinates (absolute angles to vertical)
    alpha = sp.Function("alpha")(t)
    dalpha = sp.diff(alpha, t)

    ## Coordinate system
    N = CoordSys3D("N")

    ## Positions
    x_1 = -l1 * sp.sin(alpha) * N.i - l1 * sp.cos(alpha) * N.j

    ## Velocities
    v_1 = sp.diff(x_1, t)

    ## speeds squared
    v_1sq = v_1.magnitude() ** 2

    ## Moments of inertia
    I_1 = m_a1 * l1**2 / 3

    ## Kinetic energy
    T_translation = sp.Rational(1, 2) * m * v_1sq
    T_rotational_inertia = sp.Rational(1, 2) * I_1 * dalpha**2

    T = T_translation + T_rotational_inertia  # + T_rotation

    ## Potential energy
    V = m * g * (x_1.dot(N.j)) + m_a1 * g * (x_1.dot(N.j)) / 2

    L = (T - V).subs({m_a1: 0})  ## assume no mass in the rod for simplicity
    ## Lagrange's equations of motion
    LEOM = sp.simplify(sp.diff(sp.diff(L, dalpha), t) - sp.diff(L, alpha))

    ## solve for ddalpha
    sol = sp.solve(LEOM, (sp.diff(dalpha, t)), simplify=True, rational=False)
    print(f"Does this look familiar for a pendulum?: dd_alpha = {sol[0]}")

    ## Return a function that can be used for numerical evaluation
    ddalpha_fn = sp.lambdify(
        (alpha, dalpha, m, l1, g, m_a1),
        sol[0],
        "numpy",
    )
    return ddalpha_fn


def simulate_single_pendulum(params, y0, t_span, t_eval):
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
    ddalpha_fn = get_ode_for_single_pendulum()

    # Define the system of first-order ODEs
    def odes(t, y):
        alpha, dalpha = y
        return [dalpha, ddalpha_fn(alpha, dalpha, **params)]

    # Solve the ODEs
    sol = solve_ivp(odes, t_span, y0, t_eval=t_eval, method="RK45")
    return sol


def plot_results(t, y):
    plt.figure(figsize=(5, 5))
    plt.plot(t, y[0], label="Angle (rad)")
    plt.plot(t, y[1], label="Angular Velocity (rad/s)")
    plt.title("Single Pendulum Simulation")
    plt.xlabel("Time (s)")
    plt.ylabel("Values")
    plt.legend()
    plt.grid()
    plt.show()


def animate_pendulum(t, y, l1=1.0):
    """
    Simple 2D animation of the pendulum.
    t : array of time points
    y : 2 x N array from solve_ivp; y[0] = alpha(t)
    l1: length of the pendulum arm
    """
    alpha = y[0]

    # Convert angle to Cartesian coordinates
    x = -l1 * np.sin(alpha)
    y_bob = -l1 * np.cos(alpha)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-1.2 * l1, 1.2 * l1)
    ax.set_ylim(-1.2 * l1, 1.2 * l1)
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
        line.set_data([0, x[i]], [0, y_bob[i]])
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
