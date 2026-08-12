"""
Gymnasium environment: 4-robot cluster seeking a scalar-field saddle.

Wraps the repository's QuadCluster directly rather than reimplementing the
plant, so the momentum filter (alpha = 0.7), stiction threshold (0.025 m/s),
velocity cap (0.3 m/s), and the formation shape servo are the same ones every
other experiment in this repo runs against.

The task.  A square of 4 omnidirectional robots samples a scalar field at its
four corners.  Four readings cannot determine a full 2D quadratic (six
coefficients), so the instantaneous Hessian estimate is structurally limited:
the C(4,3) plane-fit estimator used in saddle_point_4_robot.ipynb always
returns a traceless symmetric matrix, and its magnitude scales as
cos(2 theta_err) where theta_err is the formation's misalignment with the field
eigenframe.  At 45 degrees of misalignment it returns exactly zero.  Rotating
the formation is what closes that gap.

Under real actuator dynamics the hand-derived rotation servo saturates: the
tangential speed needed to spin a square of ring radius R at rate omega is
omega * R, capped at 0.3 m/s, so omega_max = 0.3 / R.  Shrinking the formation
buys rotation authority at the cost of a shorter sampling baseline.  Finding
that tradeoff is the policy's job.

Observation is a single frame with no privileged information: no gradient, no
Hessian, no true saddle location, no absolute position.  It is built entirely
from robot-relative geometry and the four readings, so it is invariant to
translation of the whole problem.

Episodes run the full horizon unless the cluster fails, by leaving the domain,
collapsing, or going non-finite.  Reaching the saddle is scored rather than
terminated on; the reason is worked through in the comment inside step().
"""
import contextlib
import io
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

_VF_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "Vector_Fields", "VF_Robot"))
if _VF_ROOT not in sys.path:
    sys.path.insert(0, _VF_ROOT)

from src.robot.quad_cluster import QuadCluster                    # noqa: E402
from src.fields.field_types import AnalyticalScalarField          # noqa: E402
from src.control.quad_kinematics import inverse_kinematics_quad   # noqa: E402

import saddle_fields as sf                                        # noqa: E402
from estimator import plane_fit_estimate                          # noqa: E402


# --------------------------------------------------------------------------
# Defaults.  Anything a caller is likely to sweep lives here as a constant.
# --------------------------------------------------------------------------

FORMATION_YAML = os.path.join(_VF_ROOT, "config", "formations", "quad_square.yaml")

R_MIN, R_MAX = 0.08, 0.40      # commandable ring radius, metres
V_MAX_CMD    = 0.35            # centroid speed command bound, m/s
OMEGA_MAX    = 2.0             # rotation command bound, rad/s

# Deliberately above the achievable ceiling at the default size: 0.3 / 0.15 =
# 2.0 rad/s only at R = 0.15, and less for any larger formation.  The policy
# has to discover either that the cap is real or that shrinking buys authority.

T_MAX        = 600             # steps per episode (60 s at dt = 0.1)
E_TOL        = 0.15            # success radius, metres
R_TARGET     = 0.20            # success formation radius, metres
N_HOLD       = 10              # consecutive steps inside tolerance to succeed

W_TRACK      = 1.0
W_SIZE       = 1.0
SIGMA_R      = 0.35            # width of the tracking kernel, metres
K_PHI        = 2.0             # progress-reward gain
C_STEP       = 0.01            # per-step cost
PENALTY      = 10.0            # NaN / degenerate formation

# reward_mode='metric' constants.  Episodes are fixed-length, so a constant
# offset is irrelevant and a plain -e term is unbiased.  SIGMA_R = 0.35 made the
# old Gaussian numerically zero (3e-4) beyond e = 1.0 while starts are drawn
# from e in [1.0, 2.5], i.e. the task reward was flat over the entire initial
# state distribution.  These terms are informative at every range instead.
E_REF        = 1.0             # distance normalizer, keeps -e/E_REF order 1
W_R_METRIC   = 0.30            # weight on formation size
BONUS        = 1.0             # paid exactly when the success gate is satisfied

R_DEGENERATE = 0.6 * R_MIN     # below this the formation is considered collapsed

# Numerics bound only, well outside the nominal domain (sf.DOMAIN_HALF = 3.0).
# Truncates rather than terminates, so it carries no reward consequence and SB3
# bootstraps the value function through it.  See the note in step().
FAR_FIELD_LIMIT = 8.0


class QuadSaddleEnv(gym.Env):
    """4-robot saddle seeking under Omnibot dynamics.

    Args:
        families:     list of saddle_fields family names, or None for all six.
        obs_mode:     'raw' (default) or 'raw+est'.  'raw+est' appends the hand
                      estimator's gradient and Hessian, which turns the task
                      into "can PPO beat the estimator it was handed".
        action_mode:  'full' (default), 'no_rot' (omega forced to zero), or
                      'fixed_size' (formation size held at its initial value).
        ideal_plant:  True disables the momentum filter, stiction, and the
                      velocity cap.  This is the 0-dynamics ceiling.
        sigma_z:      standard deviation of additive Gaussian reading noise.
        rot_penalty:  coefficient on a |omega| effort term.  Zero by default;
                      the shared actuator budget already supplies an implicit
                      cost, since translation, rotation, and shape change all
                      draw on the same 0.3 m/s per-robot cap.
        r0:           fixed initial ring radius, or None to randomize.
        reward_mode:  'metric' (recommended) is -e/E_REF, minus a formation-size
                      term, plus a BONUS paid exactly when the success gate is
                      satisfied, so reward and metric are identical by
                      construction.  'shaped' is the historical
                      track(e)*size_gain(R) + progress reward - step cost.
                      'pure' is reward = -e and nothing else.  All three build
                      on e = distance to the TRUE saddle, privileged at reward
                      time and never observed.

                      Note on 'pure': it is retained for reproducibility but it
                      is NOT a clean control.  When domain exit was terminal it
                      made running out of bounds strictly optimal above
                      e ~ 1.04, so the earlier finding that 'pure' and 'shaped'
                      failed identically said nothing about reward shape; both
                      were dominated by the termination bug.  Re-run it against
                      the current environment before drawing any conclusion.
    """

    metadata = {"render_modes": []}

    def __init__(self, families=None, obs_mode="raw", action_mode="full",
                 ideal_plant=False, sigma_z=0.0, rot_penalty=0.0, r0=None,
                 t_max=T_MAX, reward_mode="shaped", seed=None):
        super().__init__()

        if obs_mode not in ("raw", "raw+est"):
            raise ValueError(f"obs_mode must be 'raw' or 'raw+est', got {obs_mode!r}")
        if action_mode not in ("full", "no_rot", "fixed_size"):
            raise ValueError(f"bad action_mode {action_mode!r}")
        if reward_mode not in ("shaped", "pure", "metric"):
            raise ValueError(f"bad reward_mode {reward_mode!r}")

        self.families = families
        self.obs_mode = obs_mode
        self.action_mode = action_mode
        self.ideal_plant = ideal_plant
        self.sigma_z = float(sigma_z)
        self.rot_penalty = float(rot_penalty)
        self.r0 = r0
        self.t_max = int(t_max)
        self.reward_mode = reward_mode

        self._rng = np.random.default_rng(seed)

        # Actions are always 4-dimensional and always in [-1, 1]; the disabled
        # channels in the ablation modes are simply ignored.  Keeping the shape
        # fixed means one policy architecture works for every ablation.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        n_obs = 24 + (6 if obs_mode == "raw+est" else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,),
                                            dtype=np.float32)

        self._build_cluster()

        self.field = None
        self.fld = None
        self._t = 0
        self._hold = 0
        self._n_in_tol = 0
        self._ever_held = False
        self._prev_e = 0.0
        self._last_obs = np.zeros(n_obs, dtype=np.float32)
        self.episode_log = None

    # ---------------------------------------------------------------- setup

    def _build_cluster(self):
        """Construct the QuadCluster once and reuse it across episodes.

        Two hazards in QuadCluster.__init__ handled here.  It prints a banner,
        which would be emitted once per worker process, and _initialize_robots
        draws three values from the GLOBAL numpy RNG, which would perturb any
        other seeded stream in the process.  The drawn values are overwritten
        immediately by our own placement, so the state is saved and restored.
        """
        placeholder = AnalyticalScalarField(lambda x, y: 0.0)
        gstate = np.random.get_state()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.cluster = QuadCluster(FORMATION_YAML, placeholder)
        finally:
            np.random.set_state(gstate)

        if self.ideal_plant:
            for rb in self.cluster.robots:
                rb.momentum_alpha = 0.0
                rb.stiction_threshold = 0.0
                rb.max_velocity = 1e9

    def _place(self, centroid, theta, radius):
        """Put the square at an exact pose. QuadCluster.reset cannot do this.

        reset() hardcodes theta_c = pi and offers no heading argument, so
        random initial headings go through inverse kinematics and direct robot
        placement, the same approach used in
        experiments/cluster_orbit_demonstrator.py.
        """
        d = 2.0 * radius
        pos = inverse_kinematics_quad(centroid[0], centroid[1], theta,
                                      d, d, 0.5, 0.5, np.pi / 2.0)
        for i, rb in enumerate(self.cluster.robots):
            rb.position = np.array([pos[2 * i], pos[2 * i + 1]], dtype=float)
            rb.velocity = np.zeros(2, dtype=float)

        self.cluster.desired_d1 = d
        self.cluster.desired_d2 = d
        self.cluster.desired_r1 = 0.5
        self.cluster.desired_r2 = 0.5
        self.cluster.desired_phi = np.pi / 2.0
        self.cluster.prev_theta_c = theta

        self.cluster.center_history = []
        self.cluster.robot_history = []
        self.cluster.velocity_history = []

    # ------------------------------------------------------------ mechanics

    def _robot_xy(self):
        return np.asarray(self.cluster.get_robot_positions(), float).reshape(4, 2)

    def _robot_v(self):
        return np.array([rb.velocity for rb in self.cluster.robots], float)

    def _readings(self, xy):
        z = np.array([self.fld.phi(p[0], p[1]) for p in xy], float)
        if self.sigma_z > 0.0:
            z = z + self._rng.normal(0.0, self.sigma_z, size=4)
        return z

    def _observe(self, xy, v, z):
        """Build the observation.

        Readings and geometry are split into direction and log-magnitude.  The
        raw quantities span orders of magnitude across the six field families
        and across formation sizes; feeding a unit vector plus a log scale
        keeps every channel O(1) without discarding anything.
        """
        c = xy.mean(axis=0)
        vc = v.mean(axis=0)

        zdev = z - z.mean()
        zmag = float(np.linalg.norm(zdev))
        zhat = zdev / zmag if zmag > 1e-12 else np.zeros(4)

        rel = xy - c
        R = float(np.mean(np.linalg.norm(rel, axis=1)))
        rel_hat = (rel / R if R > 1e-9 else rel).ravel()

        vrel = (v - vc).ravel()

        obs = np.concatenate([
            zhat,                               # 4  reading pattern
            [np.log(max(zmag, 1e-12))],         # 1  reading scale
            rel_hat,                            # 8  formation shape and angle
            [np.log(max(R, 1e-9))],             # 1  formation size
            vrel,                               # 8  per-robot momentum state
            vc,                                 # 2  centroid velocity
        ])

        if self.obs_mode == "raw+est":
            # Split into direction and log-magnitude, the same treatment every
            # other channel gets.  Appending g and H raw made them the widest
            # dynamic-range entries in the vector, so they spent much of their
            # time pinned at VecNormalize's clip_obs = 10 and the information
            # was destroyed exactly where it mattered most.  H is always
            # traceless and symmetric (estimator.py), so (H00, H01) carries all
            # of it; m = hypot(H00, H01) is the single curvature scalar and it
            # is what vanishes in the 45-degree blind spot.
            H, g = plane_fit_estimate(xy, z)
            gmag = float(np.hypot(g[0], g[1]))
            ghat = (g / gmag) if gmag > 1e-12 else np.zeros(2)
            m = float(np.hypot(H[0, 0], H[0, 1]))
            hhat = (np.array([H[0, 0], H[0, 1]]) / m) if m > 1e-12 else np.zeros(2)
            obs = np.concatenate([
                obs,
                ghat,                              # 2  gradient direction
                [np.log(max(gmag, 1e-12))],        # 1  gradient magnitude
                hhat,                              # 2  curvature orientation
                [np.log(max(m, 1e-12))],           # 1  curvature scale, -> -inf
                                                   #    at the blind spot
            ])

        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0,
                            neginf=0.0).astype(np.float32)
        self._last_obs = obs
        return obs

    def _ring_radius(self, xy=None):
        xy = self._robot_xy() if xy is None else xy
        return float(np.mean(np.linalg.norm(xy - xy.mean(axis=0), axis=1)))

    def current_obs(self):
        """The observation most recently handed out.

        Cached rather than recomputed so that a controller querying it sees
        exactly what a policy would have been given.  Recomputing would draw a
        fresh reading-noise sample whenever sigma_z > 0.
        """
        return self._last_obs

    # ----------------------------------------------------------------- API

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.fld = sf.sample_field(self._rng, self.families)
        self.field = AnalyticalScalarField(self.fld.phi)
        self.cluster.field = self.field

        start = self.fld.sample_start(self._rng)
        theta = self._rng.uniform(0.0, 2.0 * np.pi)
        radius = self.r0 if self.r0 is not None else self._rng.uniform(0.12, 0.30)
        self._place(start, theta, radius)

        self._t = 0
        self._hold = 0
        self._n_in_tol = 0
        self._ever_held = False
        self._prev_e = float(np.linalg.norm(start - self.fld.saddle))
        self.episode_log = {k: [] for k in
                            ("centroid", "radius", "theta", "omega_cmd",
                             "omega_ach", "e", "reward", "robot_speed",
                             "robots")}

        xy = self._robot_xy()
        return self._observe(xy, self._robot_v(), self._readings(xy)), self._info()

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1.0, 1.0)

        vx = float(a[0]) * V_MAX_CMD
        vy = float(a[1]) * V_MAX_CMD
        om = 0.0 if self.action_mode == "no_rot" else float(a[2]) * OMEGA_MAX

        if self.action_mode != "fixed_size":
            r_cmd = R_MIN + 0.5 * (float(a[3]) + 1.0) * (R_MAX - R_MIN)
            self.cluster.desired_d1 = 2.0 * r_cmd
            self.cluster.desired_d2 = 2.0 * r_cmd

        th_before = self.cluster.prev_theta_c
        self.cluster.move(lambda _c: (vx, vy, om))
        th_after = self.cluster.prev_theta_c
        omega_ach = (th_after - th_before) / self.cluster.timestep

        xy = self._robot_xy()
        v = self._robot_v()
        self._t += 1

        # -- failure modes ------------------------------------------------
        if not np.all(np.isfinite(xy)):
            self._last_obs = np.zeros(self.observation_space.shape, np.float32)
            return (self._last_obs, -PENALTY, True, False,
                    self._info(outcome="nan"))

        c = xy.mean(axis=0)
        e = float(np.linalg.norm(c - self.fld.saddle))
        R = self._ring_radius(xy)

        # Leaving the nominal domain is NOT a terminal event.  It used to be,
        # with terminated=True and -PENALTY, and that was the single largest
        # defect in this environment.  Two reasons it had to go:
        #
        #   1. The boundary is defined in (c - saddle), the exact coordinate the
        #      observation deliberately hides, so the policy could not see the
        #      wall and the critic could not predict the penalty.  The -10
        #      arrived as an unpredictable shock that corrupted the advantage
        #      estimate for every far-field state.
        #   2. -10 against a per-episode return range of ~120 made exiting cost
        #      5-8% of the range for a failure the success metric scores at
        #      100%.  Under reward_mode='pure' it was worse than cosmetic:
        #      driving out was STRICTLY optimal above e ~ 1.04, by up to +200
        #      return, and the whole start annulus sits above that.
        #
        # Measured consequence: the median PPO episode ended at e ~ 3.0 (the
        # boundary) with e_min == e_0, i.e. zero progress, straight out from
        # step one.  Now the only early exits are genuine numerical failures.
        # The far-field guard below is a numerics bound, not a task rule: it
        # truncates rather than terminates, so SB3 bootstraps V(s) and there is
        # no free lunch in running away.
        if e > FAR_FIELD_LIMIT:
            return (self._observe(xy, v, self._readings(xy)),
                    0.0, False, True, self._info(outcome="far_field", e=e))
        if R < R_DEGENERATE or not np.isfinite(R):
            return (self._observe(xy, v, self._readings(xy)),
                    -PENALTY, True, False, self._info(outcome="collapsed", e=e))

        # -- reward -------------------------------------------------------
        in_tol_now = (e < E_TOL) and (R < R_TARGET)

        if self.reward_mode == "metric":
            # Dense everywhere, and identical to the success metric by
            # construction.  Episodes are fixed-length now, so there is no
            # incentive to end early and a constant offset changes nothing.
            #
            # The BONUS term exists because the old reward and the metric
            # disagreed about formation size: success gates on R < R_TARGET
            # (a cliff), while the old size_gain was a gentle 23% ramp with no
            # feature at the gate, and reward_mode='pure' had no size term at
            # all.  Measured consequence for 'pure': 16.7% of steps sat at
            # e < 0.15 but only 0.5% counted as in-tolerance, because the
            # formation was too big.  Two-thirds of its near-misses were thrown
            # away on a criterion its reward never mentioned.
            reward = (-e / E_REF
                      - W_R_METRIC * (R - R_MIN) / (R_MAX - R_MIN)
                      + BONUS * float(in_tol_now)
                      - self.rot_penalty * abs(om))
            self._prev_e = e
        elif self.reward_mode == "pure":
            # Exactly what it says: the true distance to the saddle, and
            # nothing else.  No size term, no shaping, no step cost.  Kept as
            # a direct test of whether the 'shaped' reward's extra structure
            # is what steers the policy onto plateaus and extrema rather than
            # the saddle itself, since those are reward-losers under either
            # reward as long as e is measured honestly, so a bimodal failure
            # here would point at optimization/generalization, not the
            # reward's shape.
            reward = -e
            self._prev_e = e
        else:
            track = W_TRACK * np.exp(-(e * e) / (SIGMA_R * SIGMA_R))
            size_gain = 1.0 + W_SIZE * (R_MAX - np.clip(R, R_MIN, R_MAX)) / (R_MAX - R_MIN)
            # Multiplicative, not additive.  An additive size bonus pays out for
            # shrinking anywhere in the domain, so the optimal policy would
            # collapse the formation and park far from the saddle.
            r_task = track * size_gain

            # Progress reward: K * (e_prev - e).  Exactly zero for standing
            # still, positive for approaching, negative for receding, and it
            # telescopes to K*(e_0 - e_T) over the episode.
            #
            # This was gamma * Phi(s') - Phi(s) with gamma = 0.99 and
            # Phi = -K*e, advertised as policy-invariant potential shaping.
            # It was not.  For a stationary agent that form pays
            # K*(1-gamma)*e = +0.02*e every step, so parking at e=3 and doing
            # nothing earned +0.05/step net of the step cost, and earned MORE
            # the further away it sat.  Measured: the do-nothing controller
            # scored +21.9 on a field it never moved in.  The Ng/Harada/Russell
            # invariance result also requires Phi(terminal) = 0, which never
            # held here.  Using gamma = 1.0 in the shaping term alone removes
            # the idle bonus; the PPO discount stays at 0.99.
            shaping = K_PHI * (self._prev_e - e)
            self._prev_e = e

            reward = r_task + shaping - C_STEP - self.rot_penalty * abs(om)

        # -- station keeping, not a race -----------------------------------
        # Reaching tolerance deliberately does NOT end the episode.  An earlier
        # version terminated on success with a bonus, and PPO correctly learned
        # to avoid succeeding: ending at step 450 of 600 forfeits ~150 steps of
        # tracking reward worth far more than the bonus, so the policy hovered
        # just outside tolerance with an oversized formation.  Letting the
        # episode run its full horizon makes the reward and the metric agree,
        # since both now measure time spent parked on the saddle.
        in_tol = in_tol_now
        self._hold = self._hold + 1 if in_tol else 0
        self._n_in_tol += int(in_tol)
        if self._hold >= N_HOLD:
            self._ever_held = True

        terminated = False
        truncated = self._t >= self.t_max
        outcome = "timeout" if truncated else None

        z = self._readings(xy)
        lg = self.episode_log
        lg["centroid"].append(c.copy())
        lg["radius"].append(R)
        lg["theta"].append(th_after)
        lg["omega_cmd"].append(om)
        lg["omega_ach"].append(omega_ach)
        lg["e"].append(e)
        lg["reward"].append(reward)
        lg["robot_speed"].append(float(np.mean(np.linalg.norm(v, axis=1))))
        lg["robots"].append(xy.copy())

        return (self._observe(xy, v, z), float(reward), terminated, truncated,
                self._info(outcome=outcome, e=e, R=R))

    def _info(self, outcome=None, e=None, R=None):
        info = {
            "family": self.fld.family if self.fld is not None else None,
            "t": self._t,
        }
        if e is not None:
            info["e"] = e
        if R is not None:
            info["R"] = R
        if outcome is not None:
            # Only set on the terminal step, which is where Monitor reads it.
            # Success means the hold criterion was met at some point, and is
            # independent of how the episode ended.
            info["outcome"] = outcome
            info["is_success"] = bool(self._ever_held)
            info["time_in_tol"] = self._n_in_tol / max(self._t, 1)
        return info

    def close(self):
        pass


# --------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------

def _smoke(n_steps=2000, seed=0):
    """Random actions. Checks for NaN, unbounded history, and sane throughput."""
    import time

    env = QuadSaddleEnv(seed=seed)
    obs, _ = env.reset(seed=seed)
    assert obs.shape == env.observation_space.shape, obs.shape

    rng = np.random.default_rng(seed)
    t0 = time.time()
    n_ep, outcomes, max_hist = 0, {}, 0
    for _ in range(n_steps):
        obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, 4))
        assert np.all(np.isfinite(obs)), "non-finite observation"
        assert np.isfinite(r), "non-finite reward"
        max_hist = max(max_hist, len(env.cluster.center_history))
        if term or trunc:
            outcomes[info.get("outcome", "?")] = \
                outcomes.get(info.get("outcome", "?"), 0) + 1
            n_ep += 1
            env.reset()
    dt = time.time() - t0

    print(f"  steps            {n_steps}")
    print(f"  episodes         {n_ep}")
    print(f"  outcomes         {outcomes}")
    print(f"  obs dim          {env.observation_space.shape[0]}")
    print(f"  max history len  {max_hist}  (must stay <= t_max = {env.t_max})")
    print(f"  throughput       {n_steps/dt:.0f} steps/s")
    assert max_hist <= env.t_max, "history is not being cleared per episode"
    return True


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--check", action="store_true",
                   help="run the gymnasium API conformance checker")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.check:
        from gymnasium.utils.env_checker import check_env
        print("gymnasium env_checker ...")
        check_env(QuadSaddleEnv(seed=args.seed), skip_render_check=True)
        print("  PASS")

    if args.smoke or not args.check:
        print("smoke test, random actions ...")
        _smoke(args.steps, args.seed)
        print("  PASS")

        for mode in ("raw+est",):
            print(f"smoke test, obs_mode={mode} ...")
            env = QuadSaddleEnv(obs_mode=mode, seed=args.seed)
            o, _ = env.reset(seed=args.seed)
            print(f"  obs dim {o.shape[0]}  PASS")

        for mode in ("no_rot", "fixed_size"):
            print(f"smoke test, action_mode={mode} ...")
            env = QuadSaddleEnv(action_mode=mode, seed=args.seed)
            env.reset(seed=args.seed)
            for _ in range(50):
                env.step(np.zeros(4))
            print("  PASS")

        print("smoke test, ideal_plant ...")
        env = QuadSaddleEnv(ideal_plant=True, seed=args.seed)
        env.reset(seed=args.seed)
        for _ in range(50):
            env.step(np.array([0.5, 0.0, 0.5, 0.0]))
        print("  PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
