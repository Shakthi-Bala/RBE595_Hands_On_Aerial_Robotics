# main.py
# ---------------------------------------------------------------------
# Aerial Robotics — Quadrotor Navigation Assignment (NED frame)
#
# ✍️ What YOU must implement:
#   1) load_waypoints_csv(csv_path): read (N,E,D) waypoints from CSV.
#      IMPORTANT: Waypoints are in the NED coordinate frame.
#        • N (North): meters, positive forward
#        • E (East):  meters, positive right
#        • D (Down):  meters, positive downward (altitude ↑ means D ↓)
#
#   2) Landing + Takeoff + Resume logic in guidance_update():
#      • Start landing once when D > 0.18 (i.e., you’re close to the ground),
#        then descend to a landing depth, take off (ascend), and RESUME
#        navigating toward the remaining/last waypoint.
#      • Ensure the drone lands AT MOST ONCE per run.
#
# ✅ What’s already provided:
#   • quadrotor rigid-body physics (quad_dynamics.py)
#   • a stabilizing controller (control.py: quad_control)
#   • compulsory rendering via a neural renderer (splat_render.py)
#   • plotting and video export (utils.py)
#
# 🧭 Tips for (2):
#   • Use a simple state machine: NAV → LANDING → TAKEOFF → NAV → … → DONE
#   • Track one-time landing with a boolean flag (e.g., has_landed_once).
#   • During LANDING: keep N,E fixed; increase D toward landing depth (e.g., 0.22).
#   • During TAKEOFF: keep N,E fixed; decrease D toward the next waypoint’s D
#     (or the last waypoint’s D if you’re near the end).
#   • You DO NOT implement low-level control — we give you controller_step().
# ---------------------------------------------------------------------
from pathlib import Path
import sys, importlib.util, importlib.machinery

_here = Path(__file__).parent
cand = sorted(_here.glob("control.cpython-310-*.so"))
if not cand:
    raise ImportError(
        "Local control.cpython-310-*.so not found in Phase2.\n"
        "Run: python3 setup.py build_ext --inplace"
    )
so_path = str(cand[0])

loader = importlib.machinery.ExtensionFileLoader("control", so_path)
spec = importlib.util.spec_from_file_location("control", so_path, loader=loader)
control = importlib.util.module_from_spec(spec)
loader.exec_module(control)
sys.modules["control"] = control   # ensure later 'import control' resolves to this

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless-friendly
import matplotlib
matplotlib.use("Agg")

import csv
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List

import numpy as np
import scipy.io
import cv2
import importlib
from pyquaternion import Quaternion

# --- project modules ---
import quad_dynamics as qd
import control
import drone
import utils
from utils import Config, Paths

from generate_waypoints import (
    generate_3d_infinity_trajectory,
    sample_trajectory_points,
    save_trajectory_plots,
)

# ---------- Mandatory renderer ----------
try:
    from splat_render import SplatRenderer
    HAS_SPLAT = True
except Exception:
    HAS_SPLAT = False


# =================== High-level guidance states ===================
class Mode(str, Enum):
    NAV = "NAV"         # normal waypoint navigation
    LANDING = "LANDING" # controlled descend to landing depth
    TAKEOFF = "TAKEOFF" # controlled ascend after landing
    DONE = "DONE"       # finished (all waypoints reached)


@dataclass
class AppState:
    """Runtime variables that the main loop mutates."""
    t: float
    X: np.ndarray            # full state vector
    mode: Mode
    wp_idx: int
    wps_done: bool
    has_landed_once: bool
    landing_depth: Optional[float]
    landing_start_t: Optional[float]
    landing_end_t: Optional[float]
    takeoff_target_D: Optional[float]
    control_cd: float
    user_cd: float
    render_cd: float
    render_every: float
    render_idx: int
    # (Optional) If you want to visualize takeoff separately later:
    takeoff_start_t: Optional[float] = None
    takeoff_end_t: Optional[float] = None


# =================== Utility functions (provided) ===================
def ensure_dirs(paths: Paths) -> None:
    for d in [paths.LOG_DIR, paths.RENDER_DIR, paths.VIDEO_DIR, paths.PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)

def quat_to_rpy_xyzw(q_xyzw: np.ndarray) -> np.ndarray:
    """q=[w,x,y,z] -> [roll, pitch, yaw] (radians)."""
    qw, qx, qy, qz = q_xyzw
    q = Quaternion(qw, qx, qy, qz)
    yaw, pitch, roll = q.yaw_pitch_roll
    return np.array([roll, pitch, yaw], dtype=np.float32)

def init_renderer(cfg: Config):
    """Rendering is COMPULSORY. If unavailable, fail fast with a clear error."""
    if not HAS_SPLAT:
        raise RuntimeError("SplatRenderer not available — rendering is compulsory for this assignment.")
    return SplatRenderer(cfg.render.config_path, cfg.render.json_path)

def save_render_frame(paths: Paths, rgb: np.ndarray, depth: np.ndarray, idx: int) -> None:
    rgb_path   = os.path.join(paths.RENDER_DIR, f"rgb_{idx:05d}.jpg")
    depth_path = os.path.join(paths.RENDER_DIR, f"depth_{idx:05d}.jpg")
    depth_gray = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY) if (depth.ndim == 3 and depth.shape[2] == 3) else depth
    cv2.imwrite(rgb_path, rgb)
    cv2.imwrite(depth_path, depth_gray)

def init_state(start_pos_ned: np.ndarray, cam_pitch_deg: float) -> np.ndarray:
    """Build initial state vector X from start position & camera pitch (NED)."""
    xyz  = start_pos_ned.copy()           # [N,E,D]
    vxyz = np.zeros(3)                    # [vN,vE,vD]
    quat = np.array([                     # unit quaternion [w,x,y,z]
        np.cos(np.deg2rad(cam_pitch_deg)/2.0),
        0.0,
        np.sin(np.deg2rad(cam_pitch_deg)/2.0),
        0.0
    ])
    pqr  = np.zeros(3)                    # body rates [p,q,r]
    return np.concatenate([xyz, vxyz, quat, pqr])

# TODO - Modify the thresholds to get your drone till the end of the trajectory #######
def get_next_waypoint(waypoints: np.ndarray, current_idx: int,
                      current_pos: np.ndarray, threshold: float = 0.15):
    """
    Simple waypoint sequencer: if we're within 'threshold' meters of the current waypoint,
    advance to the next one.
    Returns: (wp, new_idx, finished_flag)
    """
    if current_idx >= len(waypoints):
        return waypoints[-1], current_idx, True
    wp = waypoints[current_idx]
    if np.linalg.norm(current_pos - wp) < threshold:
        current_idx += 1
        print(f"Reached waypoint {current_idx}/{len(waypoints)}")
        if current_idx >= len(waypoints):
            return waypoints[-1], current_idx, True
        wp = waypoints[current_idx]
    return wp, current_idx, False

def controller_step(ctrl, X, WP, V, A) -> np.ndarray:
    """
    Low-level control hookup (DO NOT MODIFY for this assignment).
    Given:
      X  : full state
      WP : [N_sp, E_sp, D_sp, yaw_sp]
      V  : desired linear velocity [vN, vE, vD] (optional, can be zeros)
      A  : desired linear acceleration [aN, aE, aD] (optional, can be zeros)
    Returns:
      U: the control input for the dynamics integrator.
    """
    return ctrl.step(X, WP, V, A)

def integrate_dynamics(cfg: Config, t: float, X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Semi-implicit Euler step for the rigid-body model."""
    return X + cfg.sim.dynamics_dt * qd.model_derivative(t, X, U, drone)


# =================== PART (A): Waypoint CSV loader ===================
def load_waypoints_csv(csv_path: str) -> np.ndarray:

    rows: List[List[float]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        first_row_checked = False
        for r in reader:
            if not r:
                continue
            # Trim whitespace
            r = [c.strip() for c in r]
            # If the first row looks like a header, skip it once
            if not first_row_checked:
                first_row_checked = True
                header_like = any(s.lower() in ("n", "north") for s in r) or \
                              any(s.lower() in ("e", "east")  for s in r) or \
                              any(s.lower() in ("d", "down", "depth") for s in r)
                if header_like:
                    continue  # skip header
                # else fall through to try parse as data

            # Expect exactly 3 columns (N, E, D)
            if len(r) < 3:
                continue
            try:
                n = float(r[0])
                e = float(r[1])
                d = float(r[2])
                rows.append([n, e, d])
            except Exception:
                # Skip lines that don't parse cleanly
                continue

    if not rows:
        raise ValueError(f"No valid waypoints parsed from CSV: {csv_path}")

    wps = np.asarray(rows, dtype=np.float32)
    if wps.ndim != 2 or wps.shape[1] != 3:
        raise ValueError(f"Waypoint CSV must yield Kx3 array, got shape {wps.shape}")
    return wps
    """
    TODO (STUDENTS): Implement reading waypoints from CSV.

    Requirements:
      • Input CSV should contain waypoints in the NED frame.
      • Expected columns (with or without a header row): N, E, D   (floats, meters)
      • Return a numpy array of shape [K, 3] with dtype float32:
            [[N0, E0, D0],
             [N1, E1, D1],
             ...]
      • Be robust to the presence/absence of a header. If the first row isn’t a header,
        treat it as data.

    Hints:
      • Use Python's csv module (imported above).
      • Strip whitespace and guard with try/except around float() casts.
      • Skip rows that don’t parse properly.

    NOTE:
      • Coordinates are NED. D is "Down" (positive downward). Altitude up = -D.

    Replace the NotImplementedError below with your solution.
    """
    
    raise NotImplementedError("Implement CSV waypoint loading here (N,E,D in NED frame).")


def build_waypoints(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the list of navigation waypoints.

    Behavior:
      • If cfg.traj.use_csv and the file exists, use the CSV (students' loader).
      • Otherwise, generate a default 3D "infinity" trajectory and sample points.

    Returns:
      sampled_wps: [K,3]  — the navigation waypoints (N,E,D).
      full_traj:   [N,3]  — dense param. trajectory (for plotting context).
      times:       [N]    — param. times for the dense trajectory.
    """
    if cfg.traj.use_csv and os.path.exists(cfg.traj.csv_path):
        sampled = load_waypoints_csv(cfg.traj.csv_path)  # ← your function
        full_traj = sampled.copy()
        times = np.linspace(0.0, cfg.traj.total_time, len(full_traj), dtype=np.float32)
        print(f"[waypoints] Loaded {len(sampled)} from {cfg.traj.csv_path}")
        return sampled, full_traj, times

    # Fallback: generate an infinity trajectory and sample K waypoints.
    full_traj, times = generate_3d_infinity_trajectory(
        total_time=cfg.traj.total_time, dt=0.05,
        z_high=cfg.traj.z_high, z_low=cfg.traj.z_low,
        radius_x=cfg.traj.radius_x, radius_y=cfg.traj.radius_y, center=(0.0, 0.0)
    )
    sampled, _, _ = sample_trajectory_points(
        full_traj, times, num_samples=cfg.traj.num_waypoints,
        ordered=not cfg.traj.randomize_waypoints, random_seed=None
    )
    save_trajectory_plots(full_traj, times, sampled_points=sampled,
                          z_high=cfg.traj.z_high, z_low=cfg.traj.z_low, save_dir=cfg.paths.PLOTS_DIR)
    print(f"[waypoints] Generated {len(sampled)} sampled points from infinity trajectory")
    return sampled, full_traj, times


# =================== PART (B): Landing/Takeoff ===================
def guidance_update(state: AppState, cfg: Config, sampled_wps: np.ndarray,
                    WP: np.ndarray, V: np.ndarray, A: np.ndarray,
                    gt_wp_log: List[np.ndarray], exec_wp_log: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Update guidance setpoints based on the current mode (state machine).

    TODO (STUDENTS): Implement LANDING and TAKEOFF logic, and the transition triggers.
    The landing trigger is a CONSTANT threshold in Down (D):
        Start landing if D > 0.18  (i.e., altitude is low), but only ONCE during the run.

    Modes:
      NAV:
        - Follow waypoints using get_next_waypoint().
        - If landing condition is met and we have NOT landed before, switch to LANDING.
      LANDING:
        - Progressively increase D (go down) until reaching a final landing depth (e.g., 0.22).
        - When reached, switch to TAKEOFF. Record landing_start/end times if you want to plot.
      TAKEOFF:
        - Progressively decrease D (go up) toward the next waypoint’s D (or last WP).
        - When reached (within tolerance), switch back to NAV and continue.

    Provided setpoints for the controller:
      WP = [N_sp, E_sp, D_sp, yaw_sp]
      V  = desired linear velocity (optional; zeros are fine)
      A  = desired linear acceleration (optional; zeros are fine)

    NOTE: Coordinates are NED. D is positive down. Altitude up is -D.
    """
    pos_NED = state.X[0:3].copy()
    D_now   = pos_NED[2]

    TRIGGER_D   = getattr(getattr(cfg, "land", cfg), "trigger_D", 0.18)
    # Final landing depth (go slightly deeper than trigger)
    LAND_D      = getattr(getattr(cfg, "land", cfg), "landing_D", 0.22)
    # Vertical rates (per guidance tick)
    STEP_DOWN   = getattr(getattr(cfg, "land", cfg), "step_down", 0.0035)  # increase D
    STEP_UP     = getattr(getattr(cfg, "land", cfg), "step_up",   0.0035)  # decrease D
    # Tolerance for reaching targets in D
    D_TOL       = getattr(getattr(cfg, "land", cfg), "d_tol",     0.01)
    # Render cadence overrides
    RENDER_LAND = getattr(getattr(cfg, "render", cfg), "every_sec_land",
                          getattr(cfg.render, "every_sec_nav", 0.2))

    # ---------------- NAV mode (already implemented) ----------------
    if state.mode == Mode.NAV:
        # TODO : Add a one-time landing trigger here when D_now > 0.18
        # Suggested guard:
        if (not state.wps_done) and (not state.has_landed_once) and (state.t > 2.0) and (D_now > 0.18):
            state.mode = Mode.LANDING
            state.landing_depth = D_now
            state.landing_start_t = state.t
            state.render_every = cfg.render.every_sec_land
            print(f"[LANDING] start t={state.t:.2f}s, D={D_now:.3f}")

        if state.mode == Mode.NAV:
            target_wp, state.wp_idx, state.wps_done = get_next_waypoint(sampled_wps, state.wp_idx, pos_NED)
            if state.wps_done:
                print(f"[NAV] all waypoints completed t={state.t:.2f}s")
                state.mode = Mode.DONE
                target_wp = pos_NED 
            n_sp, e_sp, d_sp = target_wp
            yaw_sp = 0.0
            WP[:] = np.array([n_sp, e_sp, d_sp, yaw_sp], dtype=np.float32)
            V[:]  = 0.0
            A[:]  = 0.0
            state.render_every = cfg.render.every_sec_nav
            gt_wp_log.append(target_wp.copy())
            exec_wp_log.append(WP[:3].copy())

    # ---------------- LANDING mode (YOU implement) ----------------
    elif state.mode == Mode.LANDING:
        # TODO (STUDENTS): descend toward a final landing depth (example: 0.22)
        # Strategy:
        #   1) Keep N,E setpoints fixed at the current position.
        #   2) Increase D setpoint gradually by a small step each update (e.g., 0.002–0.004),
        #      not exceeding the final landing depth.
        #   3) Optionally set a small downward velocity in V (e.g., +0.005 in D).
        #
        # Stop condition:
        #   When you reach the final landing depth (within a tiny tolerance),
        #   set:
        #     state.landing_end_t   = state.t
        #     state.has_landed_once = True
        #     state.takeoff_target_D = depth of next pending waypoint
        #     state.mode = Mode.TAKEOFF
        #     state.render_every = cfg.render.every_sec_nav
        #
        # Also log the current target for plotting:
        #   last_idx = max(0, min(state.wp_idx, len(sampled_wps)-1))
        #   gt_wp_log.append(sampled_wps[last_idx].copy())
        #   exec_wp_log.append(WP[:3].copy())
        # Hold current N,E; decrease D toward takeoff_target_D
        n_sp, e_sp = pos_NED[0], pos_NED[1]
        d_target   = state.landing_depth if state.landing_depth is not None else LAND_D
        d_sp       = min(D_now + STEP_DOWN, d_target)
        WP[:] = np.array([n_sp, e_sp, d_sp, 0.0], dtype=np.float32)
        V[:]  = np.array([0.0, 0.0, +STEP_DOWN], dtype=np.float32)  
        A[:]  = 0.0
        last_idx = max(0, min(state.wp_idx, len(sampled_wps) - 1))
        gt_wp_log.append(sampled_wps[last_idx].copy())
        exec_wp_log.append(WP[:3].copy())

        if abs(d_sp - d_target) <= D_TOL:
            state.landing_end_t = state.t
            state.has_landed_once = True
            resume_idx = min(state.wp_idx, len(sampled_wps) - 1)
            state.takeoff_target_D = float(sampled_wps[resume_idx][2])
            state.mode = Mode.TAKEOFF
            state.takeoff_start_t = state.t
            state.render_every = cfg.render.every_sec_nav
            print(f"[LANDING] complete t={state.t:.2f}s at D≈{d_sp:.3f}; TAKEOFF to D={state.takeoff_target_D:.3f}")

    # ---------------- TAKEOFF mode (YOU implement) ----------------
    elif state.mode == Mode.TAKEOFF:
        # TODO (STUDENTS): ascend back up (decrease D) toward state.takeoff_target_D,
        # then switch back to NAV when close (within a small epsilon, e.g., 0.01).
        # Strategy:
        #   1) Keep N,E setpoints fixed at the current position.
        #   2) Decrease D setpoint gradually each update (e.g., -0.002 to -0.004 per tick)
        #      until you reach the target D.
        #   3) Optionally set a small upward velocity in V (e.g., -0.005 in D).
        #
        # When done, set:
        #   state.mode = Mode.NAV
        #   (optionally) state.takeoff_end_t = state.t
        #
        # Also log the current target for plotting:
        #   resume_idx = max(0, min(state.wp_idx, len(sampled_wps)-1))
        #   gt_wp_log.append(sampled_wps[resume_idx].copy())
        #   exec_wp_log.append(WP[:3].copy())
        n_sp, e_sp = pos_NED[0], pos_NED[1]
        tgt_D = state.takeoff_target_D if state.takeoff_target_D is not None else D_now
        # Move D upward (decrease)
        d_sp = D_now - STEP_UP
        if d_sp < tgt_D:
            d_sp = tgt_D  # clamp so we don't overshoot above the target

        WP[:] = np.array([n_sp, e_sp, d_sp, 0.0], dtype=np.float32)
        V[:]  = np.array([0.0, 0.0, -STEP_UP], dtype=np.float32)  # optional feed-forward
        A[:]  = 0.0

        # Log against the waypoint we are resuming toward
        resume_idx = max(0, min(state.wp_idx, len(sampled_wps) - 1))
        gt_wp_log.append(sampled_wps[resume_idx].copy())
        exec_wp_log.append(WP[:3].copy())

        # Done with takeoff once we're near target D
        if abs(d_sp - tgt_D) <= D_TOL:
            state.mode = Mode.NAV
            state.takeoff_end_t = state.t
            print(f"[TAKEOFF] complete t={state.t:.2f}s; resume NAV at wp {state.wp_idx+1}/{len(sampled_wps)}")

    # ---------------- DONE mode ----------------
    elif state.mode == Mode.DONE:
        # Hold position (no-op).
        WP[:] = np.array([state.X[0], state.X[1], state.X[2], 0.0], dtype=np.float32)
        V[:]  = 0.0
        A[:]  = 0.0
        gt_wp_log.append(state.X[0:3].copy())
        exec_wp_log.append(WP[:3].copy())

    return WP, V, A


# =================== Rendering + Logging (provided) ===================
def render_if_needed(state: AppState, cfg: Config, renderer, render_times: List[float]) -> None:
    """Render the RGB/Depth frame at the current pose on a fixed cadence."""
    if state.render_cd > 0.0:
        return
    ex, ey, ez = state.X[0], state.X[1], state.X[2]
    qw, qx, qy, qz = state.X[6:10]
    rpy = quat_to_rpy_xyzw(np.array([qw, qx, qy, qz]))
    pose = np.array([ex, ey, ez])  # NED position
    rgb, depth = renderer.render(pose, rpy)  # may raise if renderer misconfigured
    save_render_frame(cfg.paths, rgb, depth, state.render_idx)
    render_times.append(state.t)
    state.render_idx += 1
    state.render_cd = state.render_every

def save_logs(cfg: Config, t_log, X_log, U_log, gt_wp_log, exec_wp_log, sampled_wps, full_traj, render_times, landing_mask):
    """Persist results to disk for later inspection (MAT + CSV)."""
    scipy.io.savemat(os.path.join(cfg.paths.LOG_DIR, 'simulation_data.mat'), {
        'time': np.asarray(t_log),
        'state': np.asarray(X_log),
        'control': np.asarray(U_log),
        'gt_waypoints': np.asarray(gt_wp_log),
        'executed_waypoints': np.asarray(exec_wp_log),
        'sampled_waypoints': np.asarray(sampled_wps),
        'full_trajectory': np.asarray(full_traj),
        'render_times': np.asarray(render_times),
        'landing_mask': np.asarray(landing_mask),
    })
    with open(os.path.join(cfg.paths.LOG_DIR, 'gt_waypoints.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['N','E','D']); w.writerows(np.asarray(gt_wp_log).tolist())
    with open(os.path.join(cfg.paths.LOG_DIR, 'executed_waypoints.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['N','E','D']); w.writerows(np.asarray(exec_wp_log).tolist())
    scipy.io.savemat(os.path.join(cfg.paths.LOG_DIR, 'gt_vs_exec.mat'), {
        'time': np.asarray(t_log),
        'gt_pos_ned': np.asarray(gt_wp_log),
        'exec_pos_ned': np.asarray(exec_wp_log),
    })

def post_plots_and_video(cfg: Config, USER_DT: float, t_log, X_log, gt_wp_log, landing_mask, render_times):
    """Produce static plots and a 3‑panel RGB|Depth|3D video."""
    utils.save_separate_trajectory_plots(cfg.paths.LOG_DIR, USER_DT, PLOT_3D=True)
    utils.create_render_plot_video(
        render_dir=cfg.paths.RENDER_DIR,
        out_path=cfg.paths.COMBINED_VIDEO,
        fps=cfg.render.fps,
        t=np.asarray(t_log),
        pos_ned=np.asarray(X_log)[:, 0:3],
        gt_points=np.asarray(gt_wp_log) if len(gt_wp_log) else None,
        landing_mask=np.asarray(landing_mask),
        render_times=render_times,
        stop_at_landing=False,  # include takeoff + resume, if implemented
    )


# =================== Main entry (provided) ===================
def main():
    # Live-reload local modules (useful during iterative development).
    importlib.reload(control)
    importlib.reload(qd)
    importlib.reload(drone)

    # Load config and ensure output folders exist.
    cfg = Config()
    ensure_dirs(cfg.paths)

    # Create the controller and get its internal update rate.
    controller = control.quad_control()
    control_dt = controller.dt

    # Waypoints: either your CSV or the fallback infinity trajectory.
    sampled_wps, full_traj, traj_times = build_waypoints(cfg)

    # Initialize the drone state: start at the first waypoint’s position with a small camera pitch.
    X0 = init_state(sampled_wps[0], cfg.sim.cam_pitch_deg)

    # Initialize runtime state (starts in NAV mode).
    state = AppState(
        t=0.0, X=X0, mode=Mode.NAV, wp_idx=0, wps_done=False,
        has_landed_once=False, landing_depth=None, landing_start_t=None,
        landing_end_t=None, takeoff_target_D=None, control_cd=0.0,
        user_cd=0.0, render_cd=0.0, render_every=cfg.render.every_sec_nav,
        render_idx=0
    )

    # Renderer (compulsory). Fails immediately if not available.
    renderer = init_renderer(cfg)

    # Logs
    t_log: List[float] = []
    X_log: List[np.ndarray] = []
    U_log: List[np.ndarray] = []
    gt_wp_log: List[np.ndarray] = []
    exec_wp_log: List[np.ndarray] = []
    render_times: List[float] = []

    # Guidance buffers (setpoints)
    WP = np.zeros(4, dtype=np.float32)   # [N,E,D,yaw]
    V  = np.zeros(3, dtype=np.float32)   # desired linear velocity
    A  = np.zeros(3, dtype=np.float32)   # desired linear acceleration

    sim_wall_start = time.time()

    # ===== Main simulation loop =====
    while state.t < cfg.sim.stop_time and state.mode != Mode.DONE:
        # High-level guidance update (where you implement landing/takeoff logic)
        if state.user_cd <= 0.0:
            WP, V, A = guidance_update(state, cfg, sampled_wps, WP, V, A, gt_wp_log, exec_wp_log)
            state.user_cd = cfg.sim.user_dt

        # Low-level controller update (already implemented and available to you)
        if state.control_cd <= 0.0:
            U = controller_step(controller, state.X, WP, V, A)
            state.control_cd = control_dt

        # Integrate the rigid-body dynamics
        state.X = integrate_dynamics(cfg, state.t, state.X, U)

        # Render at a fixed cadence (saves RGB & Depth frames)
        render_if_needed(state, cfg, renderer, render_times)

        # Update loop timers
        state.control_cd -= cfg.sim.dynamics_dt
        state.user_cd    -= cfg.sim.dynamics_dt
        state.render_cd  -= cfg.sim.dynamics_dt
        state.t          += cfg.sim.dynamics_dt

        # Log state & control
        t_log.append(state.t)
        X_log.append(state.X.copy())
        U_log.append(U.copy())

    sim_wall_dur = time.time() - sim_wall_start
    print(f"Simulation wall time: {sim_wall_dur:.2f}s")

    # Mark the landing interval for visual overlays (if you recorded it)
    t_arr = np.asarray(t_log)
    landing_mask = np.zeros_like(t_arr, dtype=bool)
    if (state.landing_start_t is not None) and (state.landing_end_t is not None):
        landing_mask = (t_arr >= state.landing_start_t) & (t_arr <= state.landing_end_t)

    # Persist results and make plots & video
    save_logs(cfg, t_log, X_log, U_log, gt_wp_log, exec_wp_log, sampled_wps, full_traj, render_times, landing_mask)
    post_plots_and_video(cfg, cfg.sim.user_dt, t_log, X_log, gt_wp_log, landing_mask, render_times)

    print("Done.")
    print(f"  Logs:   {cfg.paths.LOG_DIR}")
    print(f"  Plots:  {cfg.paths.PLOTS_DIR}")
    print(f"  Video:  {cfg.paths.COMBINED_VIDEO}")


if __name__ == "__main__":
    main()
