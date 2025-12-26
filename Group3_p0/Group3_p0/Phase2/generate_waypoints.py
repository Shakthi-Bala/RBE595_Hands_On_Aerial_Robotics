# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from typing import Optional, Tuple


# def generate_3d_infinity_trajectory(
#     total_time: float = 20.0,
#     dt: float = 0.05,
#     z_high: float = -0.10,   # upper lobe (higher altitude) -> negative (up)
#     z_low: float = 0.18,     # lower lobe (closer to ground) -> positive (down)
#     radius_x: float = 0.6,
#     radius_y: float = 0.3,
#     center: Tuple[float, float] = (0.0, 0.0),
# ):
#     """Return (trajectory[N,3], times[N]) where columns are [N, E, D]."""
#     times = np.arange(0.0, total_time + 1e-9, dt)
#     traj = np.zeros((len(times), 3), dtype=np.float32)
#     cx, cy = center

#     for i, t in enumerate(times):
#         theta = 2.0 * np.pi * t / total_time
#         # Lemniscate of Gerono in NE plane
#         n = cx + radius_x * np.sin(theta)
#         e = cy + radius_y * np.sin(theta) * np.cos(theta)
#         # Smooth D oscillation between z_high (neg) and z_low (pos)
#         d = z_low + 0.5 * (z_high - z_low) * (1.0 + np.cos(theta))
#         traj[i] = [n, e, d]

#     return traj, times


# def sample_trajectory_points(
#     trajectory: np.ndarray,
#     times: np.ndarray,
#     num_samples: int = 60,
#     ordered: bool = True,
#     random_seed: Optional[int] = None,
# ):
#     """Sample waypoints. If ordered=False and random_seed=None, you'll get a new random
#     set every run (desired for this project)."""
#     N = len(trajectory)
#     if ordered:
#         idx = np.linspace(0, N - 1, num_samples, dtype=int)
#     else:
#         if random_seed is not None:
#             np.random.seed(random_seed)
#         idx = np.sort(np.random.choice(N, size=num_samples, replace=False))
#     return trajectory[idx], times[idx], idx


# def save_trajectory_plots(
#     trajectory: np.ndarray,
#     times: np.ndarray,
#     sampled_points: Optional[np.ndarray] = None,
#     z_high: Optional[float] = None,
#     z_low: Optional[float] = None,
#     save_dir: str = "./plots",
# ):
#     os.makedirs(save_dir, exist_ok=True)
#     n, e, d = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

#     # 3D plot
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot(n, e, d, 'b-', lw=2, alpha=0.85, label='Full Trajectory')
#     ax.scatter(n[0], e[0], d[0], c='g', s=60, marker='o', label='Start')
#     ax.scatter(n[-1], e[-1], d[-1], c='orange', s=60, marker='s', label='End')
#     if sampled_points is not None and len(sampled_points) > 0:
#         ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2],
#                    c='r', s=30, marker='o', alpha=0.9, label='Sampled WPs')
#     ax.set_xlabel('North N (m)')
#     ax.set_ylabel('East E (m)')
#     ax.set_zlabel('Down D (m)  (+ = down)')
#     ax.set_title('3D Infinity (NED)')
#     ax.legend(); ax.grid(True)
#     fig.tight_layout(); fig.savefig(os.path.join(save_dir, '3d_infinity.png'), dpi=150)
#     plt.close(fig)

#     # XY projection
#     plt.figure(figsize=(8, 6))
#     plt.plot(n, e, 'b-', lw=2, alpha=0.85, label='Full Trajectory')
#     if sampled_points is not None and len(sampled_points) > 0:
#         plt.scatter(sampled_points[:, 0], sampled_points[:, 1], c='r', s=25, label='Sampled WPs')
#     plt.scatter(n[0], e[0], c='g', s=50, marker='o', label='Start')
#     plt.scatter(n[-1], e[-1], c='orange', s=50, marker='s', label='End')
#     plt.axis('equal'); plt.xlabel('North N (m)'); plt.ylabel('East E (m)')
#     plt.title('Infinity Trajectory — NE Plane'); plt.grid(True); plt.legend()
#     plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'infinity_xy.png'), dpi=150)
#     plt.close()

#     # D vs time
#     plt.figure(figsize=(10, 5))
#     plt.plot(times, d, 'b-', lw=2, label='Depth D (NED)')
#     if z_high is not None: plt.axhline(z_high, ls='--', c='g', alpha=0.6, label=f'z_high={z_high:.2f}')
#     if z_low  is not None: plt.axhline(z_low,  ls='--', c='r', alpha=0.6, label=f'z_low={z_low:.2f}')
#     plt.xlabel('Time (s)'); plt.ylabel('Down D (m)  (+ = down)')
#     plt.title('Depth Profile Over Time'); plt.grid(True); plt.legend()
#     plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'depth_vs_time.png'), dpi=150)
#     plt.close()

# generate_waypoints.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def generate_3d_infinity_trajectory(
    total_time: float = 20.0,
    dt: float = 0.05,
    z_high: float = -0.10,   # upper lobe (higher altitude) -> negative (up)
    z_low: float = 0.18,     # lower lobe (closer to ground) -> positive (down)
    radius_x: float = 0.6,
    radius_y: float = 0.3,
    center: Tuple[float, float] = (0.0, 0.0),
):
    """Return (trajectory[N,3], times[N]) where columns are [N, E, D] in NED frame."""
    times = np.arange(0.0, total_time + 1e-9, dt)
    traj = np.zeros((len(times), 3), dtype=np.float32)
    cx, cy = center

    for i, t in enumerate(times):
        theta = 2.0 * np.pi * t / total_time
        # Lemniscate of Gerono in NE plane
        n = cx + radius_x * np.sin(theta)
        e = cy + radius_y * np.sin(theta) * np.cos(theta)
        # Smooth D oscillation between z_high (neg) and z_low (pos)
        d = z_low + 0.5 * (z_high - z_low) * (1.0 + np.cos(theta))
        traj[i] = [n, e, d]

    return traj, times


def sample_trajectory_points(
    trajectory: np.ndarray,
    times: np.ndarray,
    num_samples: int = 60,
    ordered: bool = True,
    random_seed: Optional[int] = None,
):
    """Sample waypoints. If ordered=False and random_seed=None, you'll get a new random
    set every run (useful for variety). Returns (sampled_traj[K,3], sampled_times[K], idx[K])."""
    N = len(trajectory)
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    num_samples = min(num_samples, N)
    if ordered:
        idx = np.linspace(0, N - 1, num_samples, dtype=int)
    else:
        if random_seed is not None:
            np.random.seed(random_seed)
        idx = np.sort(np.random.choice(N, size=num_samples, replace=False))
    return trajectory[idx], times[idx], idx


def save_trajectory_plots(
    trajectory: np.ndarray,
    times: np.ndarray,
    sampled_points: Optional[np.ndarray] = None,
    z_high: Optional[float] = None,
    z_low: Optional[float] = None,
    save_dir: str = "./plots",
):
    os.makedirs(save_dir, exist_ok=True)
    n, e, d = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

    # 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(n, e, d, 'b-', lw=2, alpha=0.85, label='Full Trajectory')
    ax.scatter(n[0], e[0], d[0], c='g', s=60, marker='o', label='Start')
    ax.scatter(n[-1], e[-1], d[-1], c='orange', s=60, marker='s', label='End')
    if sampled_points is not None and len(sampled_points) > 0:
        ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2],
                   c='r', s=30, marker='o', alpha=0.9, label='Sampled WPs')
    ax.set_xlabel('North N (m)')
    ax.set_ylabel('East E (m)')
    ax.set_zlabel('Down D (m)  (+ = down)')
    ax.set_title('3D Infinity (NED)')
    ax.legend(); ax.grid(True)
    fig.tight_layout(); fig.savefig(os.path.join(save_dir, '3d_infinity.png'), dpi=150)
    plt.close(fig)

    # XY projection
    plt.figure(figsize=(8, 6))
    plt.plot(n, e, 'b-', lw=2, alpha=0.85, label='Full Trajectory')
    if sampled_points is not None and len(sampled_points) > 0:
        plt.scatter(sampled_points[:, 0], sampled_points[:, 1], c='r', s=25, label='Sampled WPs')
    plt.scatter(n[0], e[0], c='g', s=50, marker='o', label='Start')
    plt.scatter(n[-1], e[-1], c='orange', s=50, marker='s', label='End')
    plt.axis('equal'); plt.xlabel('North N (m)'); plt.ylabel('East E (m)')
    plt.title('Infinity Trajectory — NE Plane'); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'infinity_xy.png'), dpi=150)
    plt.close()

    # D vs time
    plt.figure(figsize=(10, 5))
    plt.plot(times, d, 'b-', lw=2, label='Depth D (NED)')
    if z_high is not None: plt.axhline(z_high, ls='--', c='g', alpha=0.6, label=f'z_high={z_high:.2f}')
    if z_low  is not None: plt.axhline(z_low,  ls='--', c='r', alpha=0.6, label=f'z_low={z_low:.2f}')
    plt.xlabel('Time (s)'); plt.ylabel('Down D (m)  (+ = down)')
    plt.title('Depth Profile Over Time'); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'depth_vs_time.png'), dpi=150)
    plt.close()


def write_waypoints_csv(path: str, waypoints: np.ndarray) -> None:
    """
    Save waypoints to CSV with header 'N,E,D' in NED coordinates.
    Each row is a waypoint: N (m), E (m), D (m).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write("N,E,D\n")
        np.savetxt(f, waypoints, delimiter=",", fmt="%.6f")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a 3D infinity trajectory, sample waypoints, and save to waypoints.csv (NED)."
    )
    parser.add_argument("--csv", type=str, default="./waypoints.csv",
                        help="Output CSV path for sampled waypoints (default: ./waypoints.csv)")
    parser.add_argument("--plots_dir", type=str, default="./plots",
                        help="Directory to save preview plots (default: ./plots)")
    parser.add_argument("--total_time", type=float, default=30.0,
                        help="Total time of the infinity trajectory (s)")
    parser.add_argument("--dt", type=float, default=0.05,
                        help="Time step for generating the dense trajectory (s)")
    parser.add_argument("--z_high", type=float, default=-0.03,
                        help="Upper-lobe D value (negative = up)")
    parser.add_argument("--z_low", type=float, default=0.22,
                        help="Lower-lobe D value (positive = down)")
    parser.add_argument("--radius_x", type=float, default=0.6,
                        help="Infinity radius on N axis (m)")
    parser.add_argument("--radius_y", type=float, default=0.3,
                        help="Infinity radius on E axis (m)")
    parser.add_argument("--num_waypoints", type=int, default=300,
                        help="How many sampled waypoints to output")
    parser.add_argument("--ordered", action="store_true",
                        help="If set, sample waypoints in-order along the curve (default: random)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for sampling (only used if not --ordered)")
    args = parser.parse_args()

    # 1) Generate dense trajectory (NED: columns [N,E,D])
    traj, times = generate_3d_infinity_trajectory(
        total_time=args.total_time,
        dt=args.dt,
        z_high=args.z_high,
        z_low=args.z_low,
        radius_x=args.radius_x,
        radius_y=args.radius_y,
        center=(0.0, 0.0),
    )

    # 2) Sample K waypoints from the dense curve
    sampled, times_s, idx = sample_trajectory_points(
        traj, times, num_samples=args.num_waypoints,
        ordered=args.ordered, random_seed=args.seed
    )

    # 3) Save plots (optional but helpful for sanity‑check)
    save_trajectory_plots(
        traj, times,
        sampled_points=sampled,
        z_high=args.z_high, z_low=args.z_low,
        save_dir=args.plots_dir
    )

    # 4) Write sampled waypoints to CSV with NED header
    write_waypoints_csv(args.csv, sampled)

    print(f"[ok] Wrote {len(sampled)} waypoints to {args.csv}")
    print(f"[ok] Plots saved under {args.plots_dir}")


if __name__ == "__main__":
    main()
