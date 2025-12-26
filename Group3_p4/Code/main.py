from splat_render import SplatRenderer
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pyquaternion import Quaternion
from control import QuadrotorController
from quad_dynamics import model_derivative
import tello
import os
import threading, queue, time, torch
import sys
import argparse

# Import RAFT
sys.path.append(os.path.abspath(os.path.join(
    "/home/alien/YourDirectoryID_p4/Code",
    "/home/alien/YourDirectoryID_p4/external/core"
)))

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCALE = 0.5   # scale for RAFT input


#Algorithm
# 1. Get optical data from RAFT only when the drone gets closer to the window
# 2. Once optical flow data is obtained, Use canny edge detector to detect edge and draw a contour around it
# 3. Get the Largest area contour and obatin the centre of the detected contour.
# 4. Perform visual servoing (horizontal and vertical alignment).
# 5. Once aligned, Move forward 0.5 meters
# 6. End the video stream

def make_panel_rgb(color_rgb):
    h, w, _ = color_rgb.shape
    panel = np.hstack([color_rgb])
    return np.ascontiguousarray(panel, dtype=np.uint8)


def frame_to_tensor_rgb(frame_rgb, device=DEVICE, scale=SCALE):
    rgb = frame_rgb  
    if scale != 1.0:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float()  # HWC -> CHW
    t = t[None]  # add batch dim
    return t.to(device)


def compute_flow(model, f1, f2):
    padder = InputPadder(f1.shape)
    f1, f2 = padder.pad(f1, f2)
    flow_low, flow_up = model(f1, f2, iters=20, test_mode=True)
    return flow_up


def flow_to_vis(flow):
    flo = flow[0].permute(1, 2, 0).cpu().numpy()
    vis = flow_viz.flow_to_image(flo)
    return vis  # RGB uint8


def detect_hole_from_flow(flow_vis):
    gray = cv2.cvtColor(flow_vis, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = flow_vis.copy()
    hole_center = None

    if len(contours) == 0:
        return overlay, hole_center

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < 500:  # Thresholding the larger area
        return overlay, hole_center

    cv2.drawContours(overlay, [largest], -1, (0, 255, 0), 2)
    M = cv2.moments(largest)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        hole_center = (cX, cY)
        cv2.circle(overlay, hole_center, 5, (255, 0, 0), -1)
        cv2.putText(overlay, "HOLE", (cX - 10, cY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return overlay, hole_center


def build_side_by_side_panel(fpv_bgr, overlay_rgb=None, flow_ratio=0.4):
    h, w, _ = fpv_bgr.shape
    panel_h = h
    panel_w = w

    # 60% FPV, 40% FLOW by default
    flow_ratio = np.clip(flow_ratio, 0.1, 0.9)
    fpv_w = int((1.0 - flow_ratio) * panel_w)
    flow_w = panel_w - fpv_w

    # Resize FPV to left side
    fpv_resized = cv2.resize(fpv_bgr, (fpv_w, panel_h), interpolation=cv2.INTER_AREA)

    # Right side: overlay (if available) or blank
    if overlay_rgb is not None:
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        overlay_resized = cv2.resize(overlay_bgr, (flow_w, panel_h), interpolation=cv2.INTER_NEAREST)
    else:
        overlay_resized = np.zeros((panel_h, flow_w, 3), dtype=np.uint8)

    panel = np.hstack([fpv_resized, overlay_resized])

    # Text labels
    cv2.putText(panel, "FPV", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(panel, "FLOW+HOLE", (fpv_w + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return panel


class FrameSaverThread(threading.Thread):
    def __init__(self, frame_dir, model_viz=None, device=None, hz=20.0, frame_count=100):
        super().__init__(daemon=True)
        self.frame_dir = frame_dir
        os.makedirs(self.frame_dir, exist_ok=True)
        self.model_viz = model_viz
        self.device = device if device is not None else DEVICE
        self.interval = 1.0 / float(hz)
        self.queue = queue.Queue(maxsize=32)
        self.last_save = time.monotonic() - self.interval
        self.count = 0
        self.frame_count = frame_count
        self._stop_evt = threading.Event()

        self.prev_tensor = None
        self.hole_center = None
        self.last_flow_vis = None

        self.last_hole_overlay_big = None
        self.last_color_with_hole = None

    def enqueue(self, color_rgb):
        try:
            if self.queue.full():
                _ = self.queue.get_nowait()
            self.queue.put_nowait(color_rgb)
        except queue.Full:
            pass

    def stop(self):
        self._stop_evt.set()

    def run(self):
        with torch.no_grad():
            latest_panel = None
            while not self._stop_evt.is_set():
                try:
                    item = self.queue.get(timeout=0.05)
                    while True:
                        try:
                            item = self.queue.get_nowait()
                        except queue.Empty:
                            break
                except queue.Empty:
                    item = None

                if item is not None:
                    color_rgb = item
                    latest_panel = make_panel_rgb(color_rgb)

                    if self.model_viz is not None:
                        cur_tensor = frame_to_tensor_rgb(color_rgb, device=self.device, scale=SCALE)

                        if self.prev_tensor is not None:
                            flow = compute_flow(self.model_viz, self.prev_tensor, cur_tensor)
                            flow_vis = flow_to_vis(flow)
                            self.last_flow_vis = flow_vis.copy()

                            hole_overlay, center_small = detect_hole_from_flow(flow_vis)

                            if center_small is not None:
                                cxs, cys = center_small
                                cx = int(cxs / SCALE)
                                cy = int(cys / SCALE)
                                self.hole_center = (cx, cy)

                                hole_overlay_big = cv2.resize(
                                    hole_overlay,
                                    (color_rgb.shape[1], color_rgb.shape[0]),
                                    interpolation=cv2.INTER_NEAREST
                                )

                                self.last_hole_overlay_big = hole_overlay_big.copy()
                                self.last_color_with_hole = color_rgb.copy()

                                debug_path = os.path.join(self.frame_dir, "hole_debug.png")
                                cv2.imwrite(
                                    debug_path,
                                    cv2.cvtColor(hole_overlay_big, cv2.COLOR_RGB2BGR)
                                )

                        self.prev_tensor = cur_tensor

                now = time.monotonic()
                if latest_panel is not None and (now - self.last_save) >= self.interval:
                    if self.count < self.frame_count:
                        fname = os.path.join(self.frame_dir, f"frame_{self.count:06d}.png")
                        cv2.imwrite(fname, cv2.cvtColor(latest_panel, cv2.COLOR_RGB2BGR))
                        self.count += 1
                        self.last_save = now


################################################
#### Navigation Function ########################
################################################
def goToWaypoint(currentPose,
                 targetPose,
                 velocity=0.1,
                 renderer=None,
                 video_writer=None,
                 log_dir="./log",
                 frame_saver_thread=None):
    """
    Navigate quadrotor to a target waypoint.
    """
    dt = 0.01
    tolerance = 0.1
    max_time = 30.0
    
    controller = QuadrotorController(tello)
    param = tello
    
    pos = np.array(currentPose['position'])
    rpy = np.array(currentPose['rpy'])
    
    vel = np.zeros(3)
    pqr = np.zeros(3)
    
    roll, pitch, yaw = rpy
    quat = (
        Quaternion(axis=[0, 0, 1], radians=yaw)
        * Quaternion(axis=[0, 1, 0], radians=pitch)
        * Quaternion(axis=[1, 0, 0], radians=roll)
    )
    
    current_state = np.concatenate([pos, vel, [quat.x, quat.y, quat.z, quat.w], pqr])
    
    target_position = np.array(targetPose)
    
    distance = np.linalg.norm(target_position - pos)
    estimated_time = min(distance / velocity * 2.0, max_time)
    
    print(f"  Navigating: {pos} → {target_position}")
    print(f"  Distance: {distance:.2f}m, Est. time: {estimated_time:.1f}s")
    
    if distance < tolerance:
        print("  Already at target!")
        return {'position': pos, 'rpy': rpy}
    
    num_points = int(estimated_time / dt)
    time_points = np.linspace(0, estimated_time, num_points)
    
    direction = target_position - pos
    unit_direction = direction / distance
    
    trajectory_points = []
    velocities = []
    accelerations = []
    
    accel_time = min(1.0, estimated_time * 0.25)
    decel_time = accel_time
    cruise_time = estimated_time - accel_time - decel_time
    
    cruise_vel = min(
        velocity,
        distance / (0.5 * accel_time + cruise_time + 0.5 * decel_time)
    )
    
    for t in time_points:
        if t <= accel_time:
            vel_mag = (cruise_vel / accel_time) * t
            acc_mag = cruise_vel / accel_time
            progress = 0.5 * (cruise_vel / accel_time) * t * t / distance
        elif t <= accel_time + cruise_time:
            vel_mag = cruise_vel
            acc_mag = 0.0
            progress = (0.5 * cruise_vel * accel_time + cruise_vel * (t - accel_time)) / distance
        else:
            t_decel = t - accel_time - cruise_time
            vel_mag = cruise_vel - (cruise_vel / decel_time) * t_decel
            acc_mag = -cruise_vel / decel_time
            progress = (
                0.5 * cruise_vel * accel_time
                + cruise_vel * cruise_time
                + cruise_vel * t_decel
                - 0.5 * (cruise_vel / decel_time) * t_decel * t_decel
            ) / distance
        
        progress = np.clip(progress, 0.0, 1.0)
        position = pos + progress * direction
        vel_vec = vel_mag * unit_direction
        acc_vec = acc_mag * unit_direction
        
        trajectory_points.append(position)
        velocities.append(vel_vec)
        accelerations.append(acc_vec)
    
    trajectory_points = np.array(trajectory_points)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)
    
    controller.set_trajectory(trajectory_points, time_points, velocities, accelerations)
    
    state = current_state.copy()
    state_final = state.copy()
    
    for i, t in enumerate(time_points):
        control_input = controller.compute_control(state, t)
        
        current_pos = state[0:3]
        error = np.linalg.norm(current_pos - target_position)
        
        print(f"t={t:6.3f}s | current={current_pos} | target={target_position} | err={error:.4f} m")
        
        if renderer is not None and video_writer is not None:
            qx, qy, qz, qw = state[6], state[7], state[8], state[9]
            quat_tmp = Quaternion(w=qw, x=qx, y=qy, z=qz)
            yaw_tmp, pitch_tmp, roll_tmp = quat_tmp.yaw_pitch_roll
            rpy_tmp = np.array([0.0, 0.0, yaw_tmp])  # keep camera upright
            
            color_frame, _, _ = renderer.render(current_pos, rpy_tmp)

            # Enqueue for flow + detection
            if frame_saver_thread is not None:
                frame_saver_thread.enqueue(color_frame)

                # Use latest overlay from thread if available
                overlay_rgb = frame_saver_thread.last_hole_overlay_big
            else:
                overlay_rgb = None

            frame_bgr = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
            panel_bgr = build_side_by_side_panel(frame_bgr, overlay_rgb, flow_ratio=0.4)

            video_writer.write(panel_bgr)
        
        if error < tolerance and t > 1.0:
            print(f"  ✓ Reached at t={t:.2f}s, error={error:.3f}m")
            state_final = state
            break
        
        if i < len(time_points) - 1:
            sol = solve_ivp(
                lambda t_local, X: model_derivative(t_local, X, control_input, param),
                [t, t + dt],
                state,
                method='RK45',
                max_step=dt
            )
            state = sol.y[:, -1]
            state_final = state
    else:
        error = np.linalg.norm(state_final[0:3] - target_position)
        print(f"  Final error: {error:.3f}m")
        print(f"CurrentPose at end: {state_final[0:3]}")
        print(f"TargetPosition: {targetPose}")
    
    final_pos = state_final[0:3]
    final_quat = Quaternion(state_final[9], state_final[6], state_final[7], state_final[8])
    final_ypr = final_quat.yaw_pitch_roll
    final_rpy = np.array([final_ypr[2], final_ypr[1], final_ypr[0]])
    
    newPose = {
        'position': final_pos,
        'rpy': final_rpy
    }
    
    return newPose


def main(renderer, model_path):
    os.makedirs('./log', exist_ok=True)
    
    currentPose = {
        'position': np.array([0.0, 0.0, 0.0]),
        'rpy': np.radians([0.0, 0.0, 0.0])
    }

    test_color, _, _ = renderer.render(currentPose['position'], currentPose['rpy'])
    h, w, _ = test_color.shape
    video_path = "./log/drone_3windows.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, 30, (w, h))

    model_path = "/home/alien/YourDirectoryID_p4/external/models/raft-sintel.pth"
    print("Loading RAFT model...")

    raft_args = argparse.Namespace(
        small=False,
        mixed_precision=False,
        alternate_corr=False
    )

    model = RAFT(raft_args)

    saver_thread = None
    hole_center_final = None

    try:
        color_image, depth_image, metric_depth = renderer.render(
            currentPose['position'],
            currentPose['rpy']
        )

        img_h, img_w, _ = color_image.shape
        img_cx = img_w / 2.0
        img_cy = img_h / 2.0

        x_coor = 0.1
        y_coor = -0.1
        z_coor = -0.1
        cur_pos = currentPose['position'].copy()
        
        targetPose1 = np.array([
            cur_pos[0] + x_coor - 0.04,
            cur_pos[1] + y_coor + 0.1,
            cur_pos[2] + z_coor
        ])

        currentPose = goToWaypoint(
            currentPose,
            targetPose1,
            velocity=0.31,
            renderer=renderer,
            video_writer=video_writer,
            log_dir="./log",
        )
        time.sleep(0.5)

        pos_2 = currentPose['position'].copy()
        targetPose2 = np.array([
            pos_2[0] + 0.297,
            pos_2[1] - 0.12,
            pos_2[2] + 0.12
        ])

        currentPose = goToWaypoint(
            currentPose,
            targetPose2,
            velocity=0.1,
            renderer=renderer,
            video_writer=video_writer,
            log_dir="./log",
        )
        
        checkpoint = torch.load(model_path)
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_key = k[7:]
            else:
                new_key = k
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)

        model = model.to(DEVICE)
        model.eval()

        saver_thread = FrameSaverThread(
            frame_dir="./log/frames",
            model_viz=model,
            device=DEVICE,
            hz=5.0,
            frame_count=200
        )
        saver_thread.start()

        # ---- Start moving while RAFT runs ----
        pos_3 = currentPose['position'].copy()
        targetPose3 = np.array([
            pos_3[0],
            pos_3[1] + 0.25,
            pos_3[2] - 0.05
        ])

        currentPose = goToWaypoint(
            currentPose,
            targetPose3,
            velocity=1.0,
            renderer=renderer,
            video_writer=video_writer,
            frame_saver_thread=saver_thread,
            log_dir="./log",
        )

        time.sleep(0.5)

        # Final hole center from thread
        hole_center_final = saver_thread.hole_center
        print(f"[HOLE DETECTION] Final hole center (px): {hole_center_final}")
        
        if hole_center_final is not None:
            cx, cy = hole_center_final
            dx = (cx - img_cx)
            dy = (cy - img_cy)

            lateral_scale_z = 0.002  
            lateral_scale_y = 0.002 * 0.16  
            dy_world = (-dx * lateral_scale_y)
            dz_world = -dy * lateral_scale_z

            center_pos = currentPose['position'].copy()
            Center_targetPose = np.array([
                center_pos[0],
                center_pos[1] - dy_world,
                center_pos[2]
            ])

            currentPose = goToWaypoint(
                currentPose,
                Center_targetPose,
                velocity=0.31,
                renderer=renderer,
                video_writer=video_writer,
                frame_saver_thread=saver_thread,
                log_dir="./log",
            )

            scale_offet_y = 0.38
            scale_offet_z = -0.05

            time.sleep(0.5)
            intermediate_pose = currentPose['position'].copy()
            Forward_targetPose = np.array([
                intermediate_pose[0] + 0.5,
                intermediate_pose[1] + scale_offet_y,
                intermediate_pose[2] + scale_offet_z
            ])

            currentPose = goToWaypoint(
                currentPose,
                Forward_targetPose,
                velocity=0.31,
                renderer=renderer,
                video_writer=video_writer,
                frame_saver_thread=saver_thread,
                log_dir="./log",
            )
        else:
            print("[HOLE DETECTION] No hole center detected; skipping alignment.")

        cv2.imwrite('rendered_frame_window.png', cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))

        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)
        cv2.imwrite('depth_frame_window.png', depth_normalized)

        print(f'Saved frame for window at position {currentPose["position"]}')

    finally:
        if video_writer is not None:
            video_writer.release()

        if saver_thread is not None:
            saver_thread.stop()
            saver_thread.join(timeout=2.0)
            print(f"no of frames saved are {saver_thread.count}")
            print(f"Final detected hole center (px): {saver_thread.hole_center}")

            if saver_thread.last_hole_overlay_big is not None:
                final_overlay_path = os.path.join("./log/frames", "hole_final_overlay.png")
                cv2.imwrite(
                    final_overlay_path,
                    cv2.cvtColor(saver_thread.last_hole_overlay_big, cv2.COLOR_RGB2BGR)
                )
                print(f"[HOLE DETECTION] Final overlay saved to {final_overlay_path}")

            if saver_thread.last_color_with_hole is not None:
                final_frame_path = os.path.join("./log/frames", "hole_final_frame.png")
                cv2.imwrite(
                    final_frame_path,
                    cv2.cvtColor(saver_thread.last_color_with_hole, cv2.COLOR_RGB2BGR)
                )
                print(f"[HOLE DETECTION] Final RGB frame saved to {final_frame_path}")

        print(f"[INFO] Video saved to {video_path}")


if __name__ == "__main__":
    config_path = "/home/alien/YourDirectoryID_p4/data/p4_colmap_nov6_1000_splat/p4_colmap_nov6_1000/splatfacto/2025-11-06_161816/config.yml"
    json_path = "/home/alien/YourDirectoryID_p4/render_settings/render_settings.json"

    renderer = SplatRenderer(config_path, json_path)
    main(renderer, model_path=None)
