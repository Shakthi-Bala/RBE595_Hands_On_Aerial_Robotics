import os
import glob
import io
import random
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ensure headless
import matplotlib.pyplot as plt
import scipy.io
import math
from typing import Optional, List


from dataclasses import dataclass, field

@dataclass
class Paths:
    LOG_DIR: str = "./log"
    RENDER_DIR: str = "./render"
    VIDEO_DIR: str = "./videos"
    PLOTS_DIR: str = "./plots"
    COMBINED_VIDEO: str = "/home/alien/MyDirectoryID_p0/Phase2/videos"

@dataclass
class RenderCfg:
    every_sec_nav: float = 0.08
    every_sec_land: float = 0.05
    fps: int = 30
    # Mandatory rendering paths
    config_path: str = "wb_frames_splat/wb_frames_colmap/splatfacto/2025-08-16_152121/config.yml"
    json_path: str   = "/home/alien/VizFlyt/vizflyt_ws/src/vizflyt_viewer/render_settings/render_config.json"

@dataclass
class SimCfg:
    stop_time: float = 120.0
    dynamics_dt: float = 0.01
    user_dt: float = 0.10
    cam_pitch_deg: float = -3.0

@dataclass
class TrajCfg:
    use_csv: bool = True
    csv_path: str = "./waypoints.csv"
    total_time: float = 30.0
    num_waypoints: int = random.randint(150, 300)
    z_high: float = -0.03
    z_low: float  =  0.18
    radius_x: float = random.uniform(0.5, 0.7)
    radius_y: float = random.uniform(0.2, 0.4)
    randomize_waypoints: bool = True

@dataclass
class LandTakeoffCfg:
    land_start_depth: float = 0.15   # CONSTANT threshold
    land_final_depth: float = 0.22
    land_step_per_user_dt: float = 0.004
    land_vel_d: float = 0.01
    land_min_time: float = 2.0
    takeoff_step_per_user_dt: float = 0.004
    takeoff_vel_d: float = -0.01
    takeoff_eps: float = 0.01

@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    render: RenderCfg = field(default_factory=RenderCfg)
    sim: SimCfg = field(default_factory=SimCfg)
    traj: TrajCfg = field(default_factory=TrajCfg)
    lt: LandTakeoffCfg = field(default_factory=LandTakeoffCfg)


def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

############### Plotting functions #############################################################
def save_separate_trajectory_plots(LOG_DIR: str, USER_DT: float, PLOT_3D: bool = True):
    """Save separate plots for GT vs Executed comparison (GT shown as green dots).
    Loads from LOG_DIR/gt_vs_exec.mat which must contain exec_pos_ned and gt_pos_ned.
    """
    mat_path = os.path.join(LOG_DIR, 'gt_vs_exec.mat')
    mat_data = scipy.io.loadmat(mat_path)
    exec_pos_ned = mat_data['exec_pos_ned']   # [K,3] (N,E,D)
    gt_pos_ned   = mat_data['gt_pos_ned']     # [K,3] (N,E,D)

    exec_x, exec_y, exec_d = exec_pos_ned[:, 0], exec_pos_ned[:, 1], exec_pos_ned[:, 2]
    gt_x,   gt_y,   gt_d   = gt_pos_ned[:, 0],   gt_pos_ned[:, 1],   gt_pos_ned[:, 2]

    GT_DOT_COLOR     = "tab:green"
    EXEC_LINE_COLOR  = "tab:blue"
    GT_DOT_KW = dict(c=GT_DOT_COLOR, s=24, marker='o', edgecolors='none', label="GT waypoints")

    # 2D XY
    plt.figure()
    plt.plot(exec_x, exec_y, color=EXEC_LINE_COLOR, linewidth=2.0, label="Executed (NED)")
    if gt_x.size > 0:
        plt.scatter(gt_x, gt_y, **GT_DOT_KW)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("North N (m)"); plt.ylabel("East E (m)")
    plt.title("XY Trajectory: Executed vs GT (GT as green dots)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "comparison_traj_xy.png"), dpi=160)
    plt.close()

    # Z(up) vs time
    t_exec = np.arange(len(exec_x)) * USER_DT
    t_gt   = np.arange(len(gt_x))   * USER_DT
    plt.figure()
    plt.plot(t_exec, -exec_d, color=EXEC_LINE_COLOR, linewidth=2.0, label="Executed z (up)")
    if gt_d.size > 0:
        plt.scatter(t_gt, -gt_d, **GT_DOT_KW)
    plt.xlabel("Time (s)"); plt.ylabel("Z up (m)")
    plt.title("Altitude: Executed vs GT (GT as green dots)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "comparison_z_vs_time.png"), dpi=160)
    plt.close()

    # 3D
    if PLOT_3D:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(exec_x, exec_y, -exec_d, color=EXEC_LINE_COLOR, linewidth=2.0, label="Executed (NED→up)")
        if gt_x.size > 0:
            ax.scatter(gt_x, gt_y, -gt_d, **GT_DOT_KW)
        ax.set_xlabel("North N (m)"); ax.set_ylabel("East E (m)"); ax.set_zlabel("Z up (m)")
        ax.set_title("3D Trajectory: Executed vs GT (GT as green dots)")
        ax.legend(); plt.tight_layout()
        plt.savefig(os.path.join(LOG_DIR, "comparison_traj_3d.png"), dpi=160)
        plt.close()

    print("Separate GT vs Executed comparison plots saved (GT = green dots).")


# ---------- NEW: RGB | Depth | 3D trajectory ----------
def _load_depth_frame(path: str, vmin: float = None, vmax: float = None) -> Optional[np.ndarray]:
    """Load a depth frame (JPG/PNG 8/16-bit or NPY float) and colorize to BGR."""
    if path.lower().endswith(".npy"):
        d = np.load(path).astype(np.float32)
    else:
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is None:
            return None
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if im.dtype == np.uint16:
            d = im.astype(np.float32)
        else:
            d = im.astype(np.float32)

    # Auto vmin/vmax if not provided (robust percentiles per frame)
    if vmin is None or vmax is None:
        finite = np.isfinite(d)
        if np.any(finite):
            vals = d[finite]
            if vmin is None: vmin = float(np.percentile(vals, 2.0))
            if vmax is None: vmax = float(np.percentile(vals, 98.0))
            if vmax <= vmin: vmax = vmin + 1.0
        else:
            vmin, vmax = 0.0, 1.0

    d_norm = np.clip((d - vmin) / (vmax - vmin), 0.0, 1.0)
    depth_u8 = (d_norm * 255.0).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
    return depth_color

# TODO - Write code to create a video with RGB | DEPTH | 3D Trajectory #
def create_render_plot_video(
    render_dir: str,
    out_path: str,
    fps: int,
    t: np.ndarray,                 # [T] log times from main loop
    pos_ned: np.ndarray,           # [T,3] state positions (N,E,D)
    gt_points: Optional[np.ndarray],  # [K,3] GT waypoints
    landing_mask: np.ndarray,      # [T] bool: True during landing
    render_times: List[float],     # len == number of saved rgb frames
    stop_at_landing: bool = True,
    view_elev: int = 25,
    view_azim: int = 135,
    depth_vmin: float = None,
    depth_vmax: float = None,
):
    """Compose an MP4 with three panels: [RGB | Depth | 3D Trajectory]."""
    
    pass