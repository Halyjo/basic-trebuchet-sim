import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
import sympy as sp
from sympy.vector import CoordSys3D
from matplotlib import animation
import os
import pickle
from matplotlib.collections import LineCollection
import matplotlib as mpl
import warnings
from typing import Literal, Optional


## Set plot design style
plt.style.use("seaborn-v0_8-talk")


def main():
    params = {
        "m1": 0.01,  # mass of the pendulum bob 10 g
        "m2": 0.0,
        "mc": 0.55,  # mass of counter weight 240 g
        "l1": 0.24,  # length of the pendulum arm (m)
        "l2": 0.20,  # length of the sling (m)
        "pp": 0.80,  # pivot proportion (0 < pp < 1)
        "m_a1": 0.01,  # mass of pendulum arm 1 (kg)
        "m_a2": 0.003,  # mass of pendulum arm 2 (kg)
    }

    ## Initial conditions for one simulation
    ## [start alpha, start beta, start angular velocity for alpha, start angular velocity for beta]
    ## See figure in README for angle definitions

    _y0 = [
        np.pi / 15,  ## start alpha
        -12 * np.pi / 13,  ## start beta
        0.0,  ## start angular velocity for alpha
        0.0,  ## start angular velocity for beta
    ]
    t_span = (0, 6)
    n = 10000
    t_eval = np.linspace(t_span[0], t_span[1], n)

    initial_condition_options = [
        [np.pi / 15 - 0.002, -12 * np.pi / 13, 0, 0],
        [np.pi / 15 - 0.001, -12 * np.pi / 13, 0, 0],
        [np.pi / 15, -12 * np.pi / 13, 0, 0],
        [np.pi / 15 + 0.001, -12 * np.pi / 13, 0, 0],
        [np.pi / 15 + 0.002, -12 * np.pi / 13, 0, 0],
    ]

    ## Simulate for each set of initial conditions
    solutions = [
        simulate_trebuchet(params, y_start, t_span, t_eval)
        for y_start in initial_condition_options
    ]

    ## Animate all simulations together with trails for the projectile
    animation = animate_trebuchet(
        solutions[0].t,
        [sol.y for sol in solutions],
        l1=params["l1"],
        l2=params["l2"],
        pivot_proportion=params["pp"],
        show_trails=True,
        trail_window=100,
        frame_step=20,
    )
    animation.save("../figures/trebuchet_simulation.gif", writer="ffmpeg", dpi=300)


def get_ode_for_trebuchet(
    start_fresh=False, show_expressions=False
) -> tuple[callable, callable]:
    """
    Derive the equations of motion for a trebuchet using Lagrange's equations.
    Return functions for numerical evaluation of the second derivatives of the angles.

    Args:
        start_fresh: If True, re-derive the equations even if a cached version exists.
        show_expressions: If True, print the symbolic expressions for the second derivatives.
    Returns:
        ddalpha_fn: Function to compute the second derivative of alpha.
        ddbeta_fn: Function to compute the second derivative of beta.
    """
    ## Symbolic derivation is slow so we store it in a pickle file and reload it
    ## instead of re-deriving it every time.
    if os.path.exists(".symbolic_computation_cache.pickle") and not start_fresh:
        print("Loading cached ODE functions...")
        with open(".symbolic_computation_cache.pickle", "rb") as f:
            payload = pickle.load(f)
            warnings.warn(
                "Using cached ODE functions. Set start_fresh=True to re-derive if you made changes."
            )
            if show_expressions:
                print("ddalpha expression:")
                sp.pprint(payload["ddalpha_expr"])
                print("\nddbeta expression:")
                sp.pprint(payload["ddbeta_expr"])
            ddalpha_fn = sp.lambdify(
                payload["args"],
                payload["ddalpha_expr"],
                "numpy",
            )
            ddbeta_fn = sp.lambdify(
                payload["args"],
                payload["ddbeta_expr"],
                "numpy",
            )
        return ddalpha_fn, ddbeta_fn

    ## symbols
    ## - t: time
    ## - g: gravitational acceleration
    ## - m1: mass of projectile
    ## - m2: mass of sling end (small, just to keep it taut)
    ## - mc: mass of counterweight
    ## - l1: length of main arm (hinged at pivot proportion pp)
    ## - l2: length of sling
    ## - pp: pivot proportion (0 < pp < 1) (fraction of l1 from pivot to projectile)
    ## - m_a1: mass of main arm (assumed uniform)
    ## - m_a2: mass of sling (assumed uniform)

    t, g, m1, m2, mc, l1, l2, pp, m_a1, m_a2 = sp.symbols(
        "t g m1 m2 mc l1 l2 pp m_a1 m_a2", real=True, positive=True
    )
    ## Used to switch to numerical value for g before deriving the equations
    ## to make symbolic computation faster.
    given = {g: 9.81}

    ## generalized coordinates (absolute angles to vertical)
    alpha = sp.Function("alpha")(t)
    beta = sp.Function("beta")(t)
    dalpha = sp.diff(alpha, t)
    dbeta = sp.diff(beta, t)
    ddalpha = sp.diff(alpha, (t, 2))
    ddbeta = sp.diff(beta, (t, 2))

    ## Coordinate system for making 2d projections easy
    N = CoordSys3D("N")

    ## Positions from pivot point (origin)
    ## x_1: position of sling connection point (end of arm 1)
    ## x_2: position of projectile (end of arm 2)
    ## x_counter: position of counterweight (exactly opposite side of arm 1)

    x_1 = -l1 * pp * sp.sin(alpha) * N.i - l1 * pp * sp.cos(alpha) * N.j
    x_2 = x_1 - l2 * sp.sin(beta) * N.i - l2 * sp.cos(beta) * N.j
    x_counter = (
        l1 * (1 - pp) * sp.sin(alpha) * N.i + l1 * (1 - pp) * sp.cos(alpha) * N.j
    )

    ## Center of mass positions
    x_cm1 = (x_counter + x_1) / 2
    x_cm2 = x_1 + (x_2 - x_1) / 2

    ## Velocities
    v_1 = sp.diff(x_1, t)
    v_2 = sp.diff(x_2, t)
    v_counter = sp.diff(x_counter, t)

    v_cm1 = sp.diff(x_cm1, t)
    v_cm2 = sp.diff(x_cm2, t)

    ## speeds squared to use in kinetic energy
    v_1sq = v_1.magnitude() ** 2
    v_2sq = v_2.magnitude() ** 2
    v_countersq = v_counter.magnitude() ** 2
    v_cm1sq = v_cm1.magnitude() ** 2
    v_cm2sq = v_cm2.magnitude() ** 2

    ## Moments of inertia for rods about center of mass
    I1_cm = m_a1 * l1**2 / 12
    I2_cm = m_a2 * l2**2 / 12

    ## When computing kinetic energy, because we have the velocities of the centers of mass,
    ## we only need to add the rotational kinetic energy about the center of mass.

    ## Kinetic energy
    T_bob1 = sp.Rational(1, 2) * m1 * v_1sq
    T_bob2 = sp.Rational(1, 2) * m2 * v_2sq
    T_bob_counter = sp.Rational(1, 2) * mc * v_countersq

    T_rod1 = sp.Rational(1, 2) * m_a1 * v_cm1sq + sp.Rational(1, 2) * I1_cm * dalpha**2
    T_rod2 = sp.Rational(1, 2) * m_a2 * v_cm2sq + sp.Rational(1, 2) * I2_cm * dbeta**2
    T = T_bob1 + T_bob2 + T_bob_counter + T_rod1 + T_rod2
    ## Note: We do not need to add rotational kinetic energy about the pivot points,
    ## because we are using the velocities of the centers of mass.

    ## Potential energy
    V = (
        m1 * g * (x_1.dot(N.j))
        + m2 * g * (x_2.dot(N.j))
        + mc * g * (x_counter.dot(N.j))
        + m_a1 * g * (x_cm1.dot(N.j))
        + m_a2 * g * (x_cm2.dot(N.j))
    )

    L = T - V

    ## Define generalized coordinates and their derivatives together
    q = sp.Matrix([alpha, beta])
    dq = sp.Matrix([dalpha, dbeta])

    ## Lagrange's equations of motion for alpha and beta
    LEOM = (sp.diff(sp.diff(L, dq), t) - sp.diff(L, q)).subs(given)

    ## Solve for ddalpha and ddbeta (second derivatives of the angles)
    M, rhs = sp.linear_eq_to_matrix(LEOM, [ddalpha, ddbeta])
    sol_vec = [sp.simplify(part) for part in M.LUsolve(rhs)]

    if show_expressions:
        print("ddalpha expression:")
        sp.pprint(sol_vec[0])
        print("\nddbeta expression:")
        sp.pprint(sol_vec[1])

    ## Return a function that can be used for numerical evaluation
    args = (alpha, beta, dalpha, dbeta, m1, m2, mc, l1, l2, pp, m_a1, m_a2)

    ## Cache the results to a pickle file for faster loading next time
    payload = {
        "ddalpha_expr": sol_vec[0],
        "ddbeta_expr": sol_vec[1],
        "args": args,
    }

    with open(".symbolic_computation_cache.pickle", "wb") as f:
        pickle.dump(payload, f)

    ## Create the numerical functions to use in simulations
    ddalpha_fn = sp.lambdify(
        args,
        sol_vec[0],
        "numpy",
    )
    ddbeta_fn = sp.lambdify(
        args,
        sol_vec[1],
        "numpy",
    )

    return ddalpha_fn, ddbeta_fn


def simulate_trebuchet(params, y0, t_span, t_eval):
    """
    Simulate the motion of a single pendulum using the derived ODE.

    Arguments:
        params: Dictionary of parameters 'm1', 'm2', 'mc', 'l1', 'l2', 'pp', 'm_a1', and 'm_a2'.
        y0: Initial conditions [alpha(0), beta(0), dalpha(0), dbeta(0)].
        t_span: Tuple (t0, tf) for the time span of the simulation.
        t_eval: Array of time points where the solution is evaluated.

    Returns:
        sol: Object with the simulation results.
    """
    ## Get the symbolic expression for ddalpha
    ddalpha_fn, ddbeta_fn = get_ode_for_trebuchet(show_expressions=True)

    ## Define the system of first-order ODEs
    ## When we have second derivatives, we parameterize them as a system of first-order ODEs
    ## by introducing new variables for the first derivatives. I.e.:
    ## d(alpha)/dt = dalpha
    ## d(dalpha)/dt = ddalpha (given by our derived function)
    ## and similarly for beta.

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
        odes, t_span, y0, t_eval=t_eval, method="DOP853"
    )  # a high order runge-kutta method
    return sol


def animate_trebuchet(
    t,
    y_list,
    l1=1.0,
    l2=1.0,
    pivot_proportion=0.8,
    labels=None,
    # --- trail options ---
    show_trails=True,
    trail_targets=("projectile",),
    trail_mode: Literal["full", "window"] = "window",
    trail_window=100,
    cmap="viridis",
    norm_mode="per_sim",
    trail_lw=0.5,
    trail_alpha=0.9,
    # --- new: frame skipping / playback control ---
    render_fps: Optional[int] = 30,  # target playback FPS (for on-screen + saving)
    frame_step: Optional[int] = None,  # override: use every `frame_step`-th sim sample
):
    """
    Animate multiple trebuchet simulations with optional speed-colored trails,
    while rendering fewer frames for a smooth real-time GIF/video.

    If `frame_step` is given, animation uses frames t[::frame_step].
    Otherwise, it uses `render_fps` to choose a step from the sim dt.

    Trails still use *all* samples up to the current frame index, so they look smooth.
    """
    N = len(t)
    if N < 2:
        raise ValueError("Need at least two time samples to draw trails.")

    # --- choose which frame indices to render ---
    if frame_step is None:
        # infer sim dt from median spacing (robust to tiny jitter)
        sim_dt = float(np.median(np.diff(t)))
        sim_fps = 1.0 / sim_dt if sim_dt > 0 else 60.0
        target_fps = 30 if render_fps is None else render_fps
        step = max(1, int(round(sim_fps / target_fps)))
    else:
        step = max(1, int(frame_step))
        target_fps = render_fps if render_fps is not None else 30

    frame_idx = np.arange(0, N, step, dtype=int)
    if frame_idx[-1] != N - 1:
        frame_idx = np.r_[frame_idx, N - 1]  # ensure last frame is included

    # --- precompute kinematics for all sims ---
    sims = []
    for y in y_list:
        alpha = y[0]
        beta = y[1]
        xcw = l1 * (1.0 - pivot_proportion) * np.sin(alpha)
        ycw = l1 * (1.0 - pivot_proportion) * np.cos(alpha)
        x1 = -l1 * pivot_proportion * np.sin(alpha)
        y1 = -l1 * pivot_proportion * np.cos(alpha)
        xp = x1 - l2 * np.sin(beta)
        yp = y1 - l2 * np.cos(beta)
        sims.append({"xcw": xcw, "ycw": ycw, "x1": x1, "y1": y1, "xp": xp, "yp": yp})

    # helper: segments + speed from full-res samples
    def build_segments_and_speed(x, y, t):
        pts = np.column_stack([x, y]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)  # (N-1, 2, 2)
        dt = np.diff(t)
        vx = np.diff(x) / dt
        vy = np.diff(y) / dt
        speed = np.sqrt(vx**2 + vy**2)  # (N-1,)
        return segs, speed

    # targets to trail
    targets = []
    if "projectile" in trail_targets:
        targets.append(("projectile", "xp", "yp"))
    if "counterweight" in trail_targets:
        targets.append(("counterweight", "xcw", "ycw"))

    trail_data = []
    for s in sims:
        d = {}
        for name, xk, yk in targets:
            segs, spd = build_segments_and_speed(s[xk], s[yk], t)
            d[name] = (segs, spd)
        trail_data.append(d)

    # global normalization for speed colors (optional)
    if show_trails and norm_mode == "global" and targets:
        all_speeds = np.concatenate(
            [d[name][1] for d in trail_data for name, _, _ in targets]
        )
        vmin, vmax = float(np.nanmin(all_speeds)), float(np.nanmax(all_speeds))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
        global_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        global_norm = None

    # --- figure/axes ---
    fig, ax = plt.subplots(figsize=(6, 6))
    span = 1.2 * (l1 + l2)
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_axis_off()

    # skeletons
    lines = []
    for _ in sims:
        (ln,) = ax.plot([], [], "o-", lw=2, alpha=0.85)
        lines.append(ln)

    if labels is not None and len(labels) == len(lines):
        for ln, lab in zip(lines, labels):
            ln.set_label(lab)
        ax.legend(loc="upper right", frameon=True)

    # trail artists
    cmap_obj = mpl.cm.get_cmap(cmap)
    trail_collections = []
    if show_trails and targets:
        for td in trail_data:
            colls = {}
            for name, _, _ in targets:
                segs, spd = td[name]
                if norm_mode == "per_sim":
                    vmin, vmax = float(np.nanmin(spd)), float(np.nanmax(spd))
                    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                        vmin, vmax = 0.0, 1.0
                    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                else:
                    norm = global_norm
                lc = LineCollection(
                    [],
                    cmap=cmap_obj,
                    norm=norm,
                    linewidths=trail_lw,
                    alpha=trail_alpha,
                    capstyle="round",
                    joinstyle="round",
                )
                # cache full-res for slicing
                lc._all_segments = segs
                lc._all_speed = spd
                ax.add_collection(lc)
                colls[name] = lc
            trail_collections.append(colls)
    else:
        trail_collections = None

    time_text = ax.text(0.05, 0.92, "", transform=ax.transAxes)

    def init():
        for ln in lines:
            ln.set_data([], [])
        if trail_collections is not None:
            for colls in trail_collections:
                for lc in colls.values():
                    lc.set_segments([])
                    lc.set_array(np.array([]))
        time_text.set_text("")
        return (*lines, time_text)

    def update(k):
        i = int(k)  # frame index into full-res arrays
        # skeletons
        for ln, s in zip(lines, sims):
            ln.set_data(
                [s["xcw"][i], 0.0, s["x1"][i], s["xp"][i]],
                [s["ycw"][i], 0.0, s["y1"][i], s["yp"][i]],
            )
        # trails (full-res up to i, but sliced per mode)
        if trail_collections is not None and i > 0:
            if trail_mode == "window":
                start = max(0, i - trail_window)
            else:
                start = 0
            stop = i  # include segments up to i-1
            for colls in trail_collections:
                for lc in colls.values():
                    segs = lc._all_segments[start:stop]
                    spd = lc._all_speed[start:stop]
                    lc.set_segments(segs)
                    lc.set_array(spd)
        # time label uses real sim time
        time_text.set_text(f"t = {t[i]:.3f} s")
        artists = [*lines, time_text]
        if trail_collections is not None:
            for colls in trail_collections:
                artists.extend(colls.values())
        return tuple(artists)

    # interval in ms chosen to match target playback FPS
    interval_ms = int(round(1000.0 / target_fps))
    ani = animation.FuncAnimation(
        fig, update, frames=frame_idx, init_func=init, blit=True, interval=interval_ms
    )
    plt.show()
    return ani


if __name__ == "__main__":
    main()
