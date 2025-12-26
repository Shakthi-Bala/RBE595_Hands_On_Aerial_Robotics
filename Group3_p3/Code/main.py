import os
import numpy as np
import cv2
from scipy.integrate import solve_ivp
from pyquaternion import Quaternion
import torch
import torch.nn as nn
import torch.nn.functional as F

from splat_render import SplatRenderer
from control import QuadrotorController
from quad_dynamics import model_derivative
import tello
import time


import window_detector as wd


# ============================================================
# 1. ATTENTION U-NET 
# ============================================================
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # encoder
        self.dconv_down1 = double_conv(in_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.maxpool = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = double_conv(512, 1024)

        # upsample
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # attention blocks
        self.att4 = AttentionBlock(F_g=1024, F_l=512, F_int=256)
        self.att3 = AttentionBlock(F_g=512, F_l=256, F_int=128)
        self.att2 = AttentionBlock(F_g=256, F_l=128, F_int=64)
        self.att1 = AttentionBlock(F_g=128, F_l=64, F_int=32)

        # decoder convs
        self.dconv_up4 = double_conv(1024 + 512, 512)
        self.dconv_up3 = double_conv(512 + 256, 256)
        self.dconv_up2 = double_conv(256 + 128, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        x = self.bottleneck(x)

        g = self.upsample(x)
        att4 = self.att4(g=g, x=conv4)
        x = torch.cat([g, att4], dim=1)
        x = self.dconv_up4(x)

        g = self.upsample(x)
        att3 = self.att3(g=g, x=conv3)
        x = torch.cat([g, att3], dim=1)
        x = self.dconv_up3(x)

        g = self.upsample(x)
        att2 = self.att2(g=g, x=conv2)
        x = torch.cat([g, att2], dim=1)
        x = self.dconv_up2(x)

        g = self.upsample(x)
        att1 = self.att1(g=g, x=conv1)
        x = torch.cat([g, att1], dim=1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)
        return out


# ============================================================
# 2. Navigation helper
# ============================================================
def goToWaypoint(currentPose,
                 targetPose,
                 velocity=0.1,
                 renderer=None,
                 video_writer=None,
                 log_dir="./log"):
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

    state = np.concatenate([pos, vel, [quat.x, quat.y, quat.z, quat.w], pqr])

    target_position = np.array(targetPose)
    distance = np.linalg.norm(target_position - pos)
    if distance < 1e-6:
        distance = 1e-6

    estimated_time = min(distance / max(velocity, 1e-3) * 2.0, max_time)

    num_points = max(int(estimated_time / dt), 2)
    time_points = np.linspace(0, estimated_time, num_points)

    direction = target_position - pos
    unit_direction = direction / np.linalg.norm(direction)

    trajectory_points = []
    velocities = []
    accelerations = []

    accel_time = min(1.0, estimated_time * 0.25)
    decel_time = accel_time
    cruise_time = estimated_time - accel_time - decel_time

    cruise_vel = min(
        velocity,
        distance / (0.5 * accel_time + cruise_time + 0.5 * decel_time),
    )

    for t in time_points:
        if t <= accel_time:
            vel_mag = (cruise_vel / accel_time) * t
            acc_mag = cruise_vel / accel_time
            progress = 0.5 * (cruise_vel / accel_time) * t * t / distance
        elif t <= accel_time + cruise_time:
            vel_mag = cruise_vel
            acc_mag = 0.0
            progress = (
                0.5 * cruise_vel * accel_time
                + cruise_vel * (t - accel_time)
            ) / distance
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

    controller.set_trajectory(
        np.array(trajectory_points),
        time_points,
        np.array(velocities),
        np.array(accelerations)
    )

    os.makedirs(log_dir, exist_ok=True)
    state_final = state.copy()

    for i, t in enumerate(time_points):
        control_input = controller.compute_control(state, t)

        current_pos = state[0:3]
        error = np.linalg.norm(current_pos - target_position)
        if error < tolerance and t > 1.0:
            state_final = state
            if renderer is not None and video_writer is not None:
                frame = render_from_state(renderer, state)
                video_writer.write(frame)
            break

        if i < len(time_points) - 1:
            sol = solve_ivp(
                lambda t, X: model_derivative(t, X, control_input, param),
                [t, t + dt],
                state,
                method='RK45',
                max_step=dt,
            )
            state = sol.y[:, -1]
            state_final = state

        if renderer is not None and video_writer is not None:
            frame = render_from_state(renderer, state)
            video_writer.write(frame)

    final_pos = state_final[0:3]
    final_quat = Quaternion(state_final[9], state_final[6], state_final[7], state_final[8])
    yaw, pitch, roll = final_quat.yaw_pitch_roll
    final_rpy = np.array([0.0, 0.0, yaw], dtype=float)

    return {
        'position': final_pos,
        'rpy': final_rpy
    }


def render_from_state(renderer, state):
    pos = state[0:3]
    qx, qy, qz, qw = state[6], state[7], state[8], state[9]
    quat = Quaternion(qw, qx, qy, qz)
    yaw, pitch, roll = quat.yaw_pitch_roll
    rpy = np.array([0.0, 0.0, yaw], dtype=float)

    color_image, depth_image, metric_depth = renderer.render(pos, rpy)
    return color_image  



def main(renderer):
    os.makedirs('./log', exist_ok=True)
    os.makedirs('./log/postprocess', exist_ok=True)

    currentPose = {
        'position': np.array([0.0, 0.0, 0.0]),
        'rpy': np.radians([0.0, 0.0, 0.0])
    }

    # video writer
    test_color, _, _ = renderer.render(currentPose['position'], currentPose['rpy'])
    h, w, _ = test_color.shape
    video_path = "./log/drone_3windows.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, 30, (w, h))

    # load model
    MODEL_PATH = "/home/alien/YourDirectoryID_p3/UNet_background_attention_1.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wd.DEVICE = DEVICE

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("[main] model loaded (strict=False)")

    
    k_y = 0.002
    k_z = 0.002

    num_windows = 3
    align_iters_per_window = 7
    forward_dist = 0.55 
     
    #Scale offset
    scale_x_offset = {
        0: 0.0,
        1: 0.0,
        2: 0.005
    }
    scale_y_offset = {
        0: 0.0,
        1: 0.005,
        2: 0.03
    }
    scale_z_offset = {
        0: 0.0,
        1: 0.0,
        2: 0.0
    }

    
    #Forward offset
    forward_x_offset = {
        0: 0.55,
        1: 0.45,
        2: 0.35  
    }
    forward_y_offset = {
        0: -0.06,
        1: -0.18,
        2: 0.1
    }  
    forward_z_offset = {
        0: -0.006,
        1: -0.04,
        2: 0.1
    }

    for win_idx in range(num_windows):
        print(f"\n===== WINDOW {win_idx + 1} / {num_windows} =====")

        
        this_window_align_iters = 2 if win_idx == 2 else align_iters_per_window
        for it in range(this_window_align_iters):
            color_image, depth_image, metric_depth = renderer.render(
                currentPose['position'],
                currentPose['rpy']
            )
            # write current frame to video
            video_writer.write(color_image)

            # run detector directly on this RGB frame
            mask, center, area, corners = wd.extract_window_featues(model, color_image)

            # save postprocess for this window+iter
            if mask is not None:
                cv2.imwrite(f'./log/postprocess/w{win_idx}_mask_iter_{it:03d}.png', mask)
                annot = color_image.copy()
                if center is not None:
                    cv2.circle(annot, center, 5, (0, 0, 255), -1)
                if corners is not None:
                    cv2.polylines(annot, [np.int32(corners)], True, (0, 255, 0), 2)
                cv2.imwrite(f'./log/postprocess/w{win_idx}_annot_iter_{it:03d}.png', annot)

            if center is None:
                print(f"[W{win_idx} ALIGN {it}] no window detected, skipping correction")
                continue

            cx, cy = center
            img_h, img_w, _ = color_image.shape
            img_cx = img_w / 2.0
            img_cy = img_h / 2.0

            err_x_px = cx - img_cx
            err_y_px = cy - img_cy
            print(f"[W{win_idx} ALIGN {it}] err_px=({err_x_px:.1f}, {err_y_px:.1f})")

            
            cur_pos = currentPose['position'].copy()
            y_corr = -k_y * err_x_px
            z_corr = -k_z * err_y_px

            targetPose = np.array([
                cur_pos[0] + scale_x_offset[win_idx],                         # stay in x while aligning
                cur_pos[1] - 0.19 * y_corr + scale_y_offset[win_idx],         
                cur_pos[2] - 0.005 + scale_z_offset[win_idx]        
            ])

            currentPose = goToWaypoint(
                currentPose,
                targetPose,
                velocity=0.2,
                renderer=renderer,
                video_writer=video_writer,
                log_dir="./log"
            )

        # FORWARD PHASE for this window
        print(f"[W{win_idx}] moving forward {forward_dist} m in x")
        targetPose = currentPose['position'].copy()
        

        cur_pos = currentPose['position'].copy()
        forward_pose = np.array([
            cur_pos[0] + forward_x_offset[win_idx],
            cur_pos[1] + forward_y_offset[win_idx],
            cur_pos[2] + forward_z_offset[win_idx]
        ])
        currentPose = goToWaypoint(
            currentPose,
            forward_pose,
            velocity=0.2,
            renderer=renderer,
            video_writer=video_writer,
            log_dir="./log"
        )

    video_writer.release()
    print(f"[INFO] Video saved to {video_path}")
    print(f"[INFO] Postprocess outputs saved to ./log/postprocess/")


if __name__ == "__main__":
    config_path = "/home/alien/YourDirectoryID_p3/data/washburn-env6-itr0-1fps/washburn-env6-itr0-1fps_nf_format/splatfacto/2025-03-06_201843/config.yml"
    json_path = "/home/alien/YourDirectoryID_p3/render_settings/render_settings_2.json"

    renderer = SplatRenderer(config_path, json_path)
    main(renderer)

#window1: -0.05
#window2: 

# forward_y_offset = {
#         0: -0.06,
#         1: -0.15,
#         2: 0.4
#     }   

#     scale_y_offset = {
#         0: 0.0,
#         1: 0.005,
#         2: 0.2
#     }

#     scale_z_offset = {
#         0: 0.0,
#         1: 0.0,
#         2: -0.2
#     }

#     forward_z_offset = {
#         0: -0.006,
#         1: -0.05,
#         2: 0.02
#     }

#     forward_x_offset = {
#         0: 0.55,
#         1: 0.49,
#         2: 0.3
#     }