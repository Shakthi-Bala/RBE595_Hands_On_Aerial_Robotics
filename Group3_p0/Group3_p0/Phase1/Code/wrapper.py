import sys, pathlib
import numpy as np
from scipy import io
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

#To import a module within same folder
THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(THIS_DIR))
from rotplot import rotplot

#Paths
imu_path = "/home/alien/MyDirectoryID_p0/Phase1/Data/Train/IMU/imuRaw1.mat"
vic_path = "/home/alien/MyDirectoryID_p0/Phase1/Data/Train/Vicon/viconRot1.mat"
param_path = "/home/alien/MyDirectoryID_p0/Phase1/IMUParams.mat"

# {.get} method to reterive data from dict
def get_data(d,main):
    return d.get(main)

#load imu data
def load_imu_data(path):
    m = io.loadmat(path)
    vals = get_data(m,"vals") #(6xN)
    ts = np.asarray(get_data(m,"ts")).ravel() #(N,)
    return np.asarray(vals),ts

#load vicon data
def load_vicon_data(path):
    v = io.loadmat(path)
    rots = np.asarray(v["rots"]) #(3x3xN)
    ts = np.asarray(get_data(v,"ts")).ravel()
    return rots,ts

def load_params(path):
    p = io.loadmat(path)
    IMUParams = np.asarray(p["IMUParams"])  #[sx sy sz], [bax bay baz]
    sx, sy, sz = IMUParams[0]
    bax, bay, baz = IMUParams[1]
    return (sx, sy, sz), (bax, bay, baz)

#Gyro logic starts here
#Gyro calib
def gyro_counts_to_rads(counts, n_bias=200):
    """ω~ = (3300/1023) * (π/180) * 0.3 * (ω - bg)"""
    n0 = min(n_bias, counts.size)
    bg = counts[:n0].mean()
    scale = (3300.0/1023.0) * (np.pi/180.0) * 0.3
    return scale * (counts - bg)

#load the data
vals, t_imu = load_imu_data(imu_path)
rots_vic_3x3xN, t_vic = load_vicon_data(vic_path)
(scales, biases) = load_params(param_path)

#[ax ay az ωz ωx ωy]
ax, ay, az  = vals[0], vals[1], vals[2]
wz_c, wx_c, wy_c = vals[3], vals[4], vals[5]

#Convert gyro to rads
wx = gyro_counts_to_rads(wx_c)
wy = gyro_counts_to_rads(wy_c)
wz = gyro_counts_to_rads(wz_c)

#Software time synchronization(took idea from GPT)
def nearest_indices(t_ref, t_query):
    """For each t_query[i], get index of nearest t_ref."""
    idx = np.searchsorted(t_ref, t_query, side="left")
    idx = np.clip(idx, 0, len(t_ref)-1)
    left = np.maximum(idx - 1, 0)
    choose_left = (idx == len(t_ref)) | (
        (idx > 0) & (np.abs(t_query - t_ref[left]) <= np.abs(t_query - t_ref[idx]))
    )
    idx[choose_left] = left[choose_left]
    return idx

#Gyro Integration
def integrate_gyro(ts, wx, wy, wz, R0_3x3):
    """
    Integrate body rates to orientation.
    R_{k+1} = R_k * Exp(ω_k * dt_k).
    Returns scipy Rotation sequence (length N).
    """
    N = ts.size
    R_seq = [R.from_matrix(R0_3x3)]
    for k in range(N - 1):
        dt = ts[k+1] - ts[k]
        if dt <= 0:
            dt = 1e-6
        rotvec = np.array([wx[k], wy[k], wz[k]]) * dt  # (rad)
        dR = R.from_rotvec(rotvec)
        R_seq.append(R_seq[-1] * dR)  
    return R.concatenate(R_seq)

#3D orientation using rotplot
def plot_3d_frames_from_rotation_sequence(R_seq, t_seq, title, n_frames=3):
    mats = R_seq.as_matrix()
    N = mats.shape[0]
    if N < 1:
        raise ValueError("Empty rotation sequence.")
    idx = np.linspace(0, N - 1, n_frames, dtype=int)

    fig = plt.figure(figsize=(4 * n_frames, 4))
    fig.suptitle(title, fontsize=12)
    for i, k in enumerate(idx, 1):
        ax = fig.add_subplot(1, n_frames, i, projection='3d')
        rotplot(mats[k], currentAxes=ax)
        ax.set_title(f"t = {t_seq[k]:.3f}s\nidx {k}")
    return fig

def rmat_stack_to_R(rots_3x3xN):
    """Convert 3x3xN to scipy Rotation (length N)."""
    mats = np.transpose(rots_3x3xN, (2, 0, 1))  # (N,3,3)
    return R.from_matrix(mats)

# Initial orientation from nearest Vicon at t0
idx0 = nearest_indices(t_vic, np.array([t_imu[0]]))[0]
R0 = rots_vic_3x3xN[:, :, idx0]
#Gyro logic ends here


#Acceleerometer logic
#Accel-only orientation
def accel_to_tilt_orient(ts, ax, ay, az, scales, biases, yaw0_rad=0.0):
    sx, sy, sz = scales
    bax, bay, baz = biases

    #ã = (a + b) / s
    ax_p = (ax + bax) / sx
    ay_p = (ay + bay) / sy
    az_p = (az + baz) / sz

    acc = np.vstack([ax_p, ay_p, az_p]).T  # (N,3)
    acc_norm = np.linalg.norm(acc, axis=1, keepdims=True)
    acc_unit = acc / np.clip(acc_norm, 1e-9, None)  # direction only

    f_w = np.array([0.0, 0.0, -1.0])  # world gravity direction

    R_list = []
    for u in acc_unit:
        v = np.cross(f_w, u)
        s = np.linalg.norm(v)
        c = float(np.dot(f_w, u))

        if s < 1e-9:  # nearly parallel or antiparallel
            if c > 0:
                dR = R.identity()
            else:
                dR = R.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0]))  # 180° about x
        else:
            axis = v / s
            angle = np.arctan2(s, c)
            dR = R.from_rotvec(axis * angle)

        R_yaw = R.from_euler('Z', yaw0_rad)  # constant yaw
        R_list.append(R_yaw * dR)            # world-yaw ∘ tilt

    return R.concatenate(R_list)

# initial yaw from Vicon
R_vic_full = rmat_stack_to_R(rots_vic_3x3xN)
idx0 = nearest_indices(t_vic, np.array([t_imu[0]]))[0]
yaw0_deg = R_vic_full[idx0].as_euler('ZYX', degrees=True)[0]
yaw0_rad = np.deg2rad(yaw0_deg)

# Accel-only orientation (tilt + constant yaw) 
R_acc = accel_to_tilt_orient(t_imu, ax, ay, az, scales, biases, yaw0_rad=yaw0_rad)
#Accelerometer logic ends here

#Complementary filter logic
#Alpha computation
def compute_alpha(ts, alpha=None, tau=None, fc=None):
    if alpha is not None:
        return float(np.clip(alpha, 0.0, 0.999999)), None, None

    dts = np.diff(ts)
    dts = dts[dts > 0]
    Ts = float(np.median(dts)) if dts.size else 1e-2

    if tau is None and fc is not None and fc > 0:
        tau = 1.0 / (2.0 * np.pi * fc)
    if tau is None or tau <= 0:
        tau = 0.5  # sensible default ~0.5 s
    a = tau / (tau + Ts)
    return float(np.clip(a, 0.0, 0.999999)), Ts, tau

def accel_tilt_from_calibrated(ax, ay, az):
    # Normalize to unit vector
    g = np.vstack([ax, ay, az]).T
    g /= np.clip(np.linalg.norm(g, axis=1, keepdims=True), 1e-9, None)

    # Standard tilt from gravity (ZYX convention)
    roll  = np.arctan2(g[:,1], g[:,2])                         # atan2(ay, az)
    pitch = np.arctan2(-g[:,0], np.sqrt(g[:,1]**2 + g[:,2]**2))# atan2(-ax, sqrt(ay^2+az^2))
    return roll, pitch

def accel_lowpass_filter(ts, ax, ay, az, scales, biases, alpha):
    sx, sy, sz = scales
    bax, bay, baz = biases

    ax_p_1 = (ax + bax) / sx
    ay_p_1 = (ay + bay) / sy
    az_p_1 = (az + baz) / sz

    roll_a, pitch_a = accel_tilt_from_calibrated(ax_p_1, ay_p_1, az_p_1)

    N = len(ts)
    roll_lp  = np.zeros(N)
    pitch_lp = np.zeros(N)
    yaw_lp   = np.zeros(N)  # accel can't observe yaw

    roll_lp[0]  = roll_a[0]
    pitch_lp[0] = pitch_a[0]

    for k in range(N-1):
        roll_lp[k+1]  = (1 - alpha) * roll_a[k+1]  + alpha * roll_lp[k]
        pitch_lp[k+1] = (1 - alpha) * pitch_a[k+1] + alpha * pitch_lp[k]
        # yaw_lp stays 0

    return roll_lp, pitch_lp, yaw_lp

def gyro_highpass_filter(ts, wx, wy, wz, alpha):
    N = len(ts)
    wx_hp = np.zeros(N); wy_hp = np.zeros(N); wz_hp = np.zeros(N)
    for k in range(N-1):
        wx_hp[k+1] = (1 - alpha) * wx_hp[k] + (1 - alpha) * (wx[k+1] - wx[k])
        wy_hp[k+1] = (1 - alpha) * wy_hp[k] + (1 - alpha) * (wy[k+1] - wy[k])
        wz_hp[k+1] = (1 - alpha) * wz_hp[k] + (1 - alpha) * (wz[k+1] - wz[k])
    return wx_hp, wy_hp, wz_hp

def complementary_filter_angles(ts, wx, wy, wz, roll_lp, pitch_lp, yaw_lp, alpha):
    N = len(ts)
    roll  = np.zeros(N); pitch = np.zeros(N); yaw = np.zeros(N)
    roll[0], pitch[0], yaw[0] = roll_lp[0], pitch_lp[0], 0.0

    for k in range(N-1):
        dt = ts[k+1] - ts[k]
        if dt <= 0: dt = 1e-6
        # Integrate gyro per axis
        roll_g_next  = roll[k]  + wx[k+1] * dt
        pitch_g_next = pitch[k] + wy[k+1] * dt
        yaw_g_next   = yaw[k]   + wz[k+1] * dt

        roll[k+1]  = (1 - alpha) * roll_g_next  + alpha * roll_lp[k+1]
        pitch[k+1] = (1 - alpha) * pitch_g_next + alpha * pitch_lp[k+1]
        yaw[k+1]   = yaw_g_next  # same as (1-α)*yaw_g_next + α*0

    return roll, pitch, yaw

def complementary_filter(ts, vals, vicon_rots_3x3xN, vicon_ts,
                         alpha=None, tau=None, fc=None, imu_params_path=None):
    # IMU channels: [ax ay az wz wx wy]
    ax, ay, az  = vals[0], vals[1], vals[2]
    wz_c, wx_c, wy_c = vals[3], vals[4], vals[5]

    # Convert gyro to rad/s
    wx = gyro_counts_to_rads(wx_c)
    wy = gyro_counts_to_rads(wy_c)
    wz = gyro_counts_to_rads(wz_c)

    # Compute alpha 
    alpha_val, Ts_est, tau_used = compute_alpha(ts, alpha=alpha, tau=tau, fc=fc)

    # Accel low-pass tilt
    if imu_params_path is None:
        raise ValueError("Provide IMUParams.mat path for accel calibration.")
    (scales, biases) = load_params(imu_params_path)
    roll_lp, pitch_lp, yaw_lp = accel_lowpass_filter(ts, ax, ay, az, scales, biases, alpha_val)

    _ = gyro_highpass_filter(ts, wx, wy, wz, alpha_val)

    # Fuse angles per axis
    roll, pitch, yaw = complementary_filter_angles(ts, wx, wy, wz,
                                                   roll_lp, pitch_lp, yaw_lp,
                                                   alpha_val)

    # Convert fused/parts to Rotation sequences (ZYX: yaw, pitch, roll)
    eul_fused = np.column_stack([yaw, pitch, roll])
    eul_acc   = np.column_stack([np.zeros_like(yaw), pitch_lp, roll_lp])
    # Gyro-only angles for plotting 
    yaw_g  = np.cumsum(np.r_[0, np.diff(ts)] * np.r_[wz[:1], wz[1:]])
    pitch_g= np.cumsum(np.r_[0, np.diff(ts)] * np.r_[wy[:1], wy[1:]])
    roll_g = np.cumsum(np.r_[0, np.diff(ts)] * np.r_[wx[:1], wx[1:]])
    eul_gyro = np.column_stack([yaw_g, pitch_g, roll_g])

    R_fused = R.from_euler('ZYX', eul_fused)
    R_acc   = R.from_euler('ZYX', eul_acc)
    R_gyro  = R.from_euler('ZYX', eul_gyro)

    # Align Vicon to IMU timeline for comparison
    idx_match = nearest_indices(vicon_ts, ts)
    R_vic = rmat_stack_to_R(vicon_rots_3x3xN[:, :, idx_match])

    return {
        "R_fused": R_fused,
        "R_gyro": R_gyro,
        "R_acc": R_acc,
        "R_vic": R_vic,
        "alpha": alpha_val,
        "Ts": Ts_est,
        "tau": tau_used
    }

out = complementary_filter(t_imu, vals, rots_vic_3x3xN, t_vic, fc=0.3, imu_params_path=param_path)
R_fused, R_gyro, R_acc, R_vic = out["R_fused"], out["R_gyro"], out["R_acc"], out["R_vic"]
alpha_val, Ts_est, tau_used = out["alpha"], out["Ts"], out["tau"]

# Integrate gyro-only orientation
R_gyro = integrate_gyro(t_imu, wx, wy, wz, R0)

# Vicon orientation aligned to IMU timestamps
idx_match = nearest_indices(t_vic, t_imu)
R_vic = rmat_stack_to_R(rots_vic_3x3xN[:, :, idx_match])
# Accel-only orientation (tilt + constant yaw) 
R_acc = accel_to_tilt_orient(t_imu, ax, ay, az, scales, biases, yaw0_rad=yaw0_rad)

# 3D orientation using rotplot  
plot_3d_frames_from_rotation_sequence(R_vic,  t_imu, "Vicon 3D Orientation", n_frames=3)
plot_3d_frames_from_rotation_sequence(R_gyro, t_imu, "Gyro-only 3D Orientation", n_frames=3)
plot_3d_frames_from_rotation_sequence(R_acc,  t_imu, "Accel-only 3D Orientation", n_frames=3)
plot_3d_frames_from_rotation_sequence(R_fused, t_imu, "Fused 3D Orientation (snapshots)", n_frames=3)

plt.show()

