# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import glob
import json
import os

import numpy as np
import torch


def _quat_apply(q, v):
    """Apply quaternion q (w,x,y,z) to vector v. q: (..., 4), v: (..., 3)."""
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack(
        [
            (1 - 2 * (qy * qy + qz * qz)) * v[..., 0] + 2 * (qx * qy - qz * qw) * v[..., 1] + 2 * (qx * qz + qy * qw) * v[..., 2],
            2 * (qx * qy + qz * qw) * v[..., 0] + (1 - 2 * (qx * qx + qz * qz)) * v[..., 1] + 2 * (qy * qz - qx * qw) * v[..., 2],
            2 * (qx * qz - qy * qw) * v[..., 0] + 2 * (qy * qz + qx * qw) * v[..., 1] + (1 - 2 * (qx * qx + qy * qy)) * v[..., 2],
        ],
        dim=-1,
    )


def _quat_conjugate(q):
    """q (w,x,y,z) -> conjugate."""
    out = q.clone()
    out[..., 1:4] = -out[..., 1:4]
    return out


def _quat_rotate_inverse(q, v):
    """Rotate v by inverse of q. q: (..., 4), v: (..., 3)."""
    return _quat_apply(_quat_conjugate(q), v)


def _quat_to_gravity_3d(q):
    """World gravity [0,0,-1] in body frame (amp_share 同款). q: (..., 4) wxyz."""
    gravity_w = torch.zeros(*q.shape[:-1], 3, device=q.device, dtype=q.dtype)
    gravity_w[..., 2] = -1.0
    return _quat_rotate_inverse(q, gravity_w)


def _load_npz_to_amp_64(device, path):
    """Load amp_share-style .npz and convert to 64-dim AMP obs (same layout as PB_AMP roban).
    Root rotation uses 3D gravity only. Returns (frames_tensor (T, 64), dt, num_frames).
    """
    data = np.load(path, allow_pickle=True)
    dof_positions = torch.tensor(data["dof_positions"], dtype=torch.float32, device=device)
    dof_velocities = torch.tensor(data["dof_velocities"], dtype=torch.float32, device=device)
    body_positions = torch.tensor(data["body_positions"], dtype=torch.float32, device=device)
    body_rotations = torch.tensor(data["body_rotations"], dtype=torch.float32, device=device)
    body_linear_velocities = torch.tensor(data["body_linear_velocities"], dtype=torch.float32, device=device)
    body_angular_velocities = torch.tensor(data["body_angular_velocities"], dtype=torch.float32, device=device)
    body_names = data["body_names"].tolist()
    dof_names = data["dof_names"].tolist()
    fps = float(data["fps"])
    dt = 1.0 / fps
    T = dof_positions.shape[0]

    ref_name = "base_link"
    key_names = ["zarm_r4_link", "zarm_l4_link", "leg_r6_link", "leg_l6_link"]
    ref_idx = body_names.index(ref_name)
    key_idxs = [body_names.index(n) for n in key_names]

    # Joint order for 21 DoF: waist, leg_l1..6, leg_r1..6, zarm_l1..4, zarm_r1..4 (match Roban S14)
    order = (
        ["waist_yaw_joint"]
        + [f"leg_l{i}_joint" for i in range(1, 7)]
        + [f"leg_r{i}_joint" for i in range(1, 7)]
        + [f"zarm_l{i}_joint" for i in range(1, 5)]
        + [f"zarm_r{i}_joint" for i in range(1, 5)]
    )
    dof_indexes = [dof_names.index(n) for n in order]

    joint_pos_21 = dof_positions[:, dof_indexes]
    joint_vel_21 = dof_velocities[:, dof_indexes]

    root_pos = body_positions[:, ref_idx, :]
    root_quat = body_rotations[:, ref_idx, :]
    root_height = root_pos[:, 2:3]
    root_gravity_3d = _quat_to_gravity_3d(root_quat)
    root_lin_vel_w = body_linear_velocities[:, ref_idx, :]
    root_ang_vel_w = body_angular_velocities[:, ref_idx, :]
    root_lin_vel_b = _quat_rotate_inverse(root_quat, root_lin_vel_w)
    root_ang_vel_b = _quat_rotate_inverse(root_quat, root_ang_vel_w)

    key_pos_w = body_positions[:, key_idxs, :]
    key_rel = key_pos_w - root_pos.unsqueeze(1)
    root_quat_rep = root_quat.unsqueeze(1).expand(-1, 4, -1).reshape(T * 4, 4)
    key_rel_flat = key_rel.reshape(T * 4, 3)
    key_in_root = _quat_rotate_inverse(root_quat_rep, key_rel_flat).reshape(T, 12)

    frames_64 = torch.cat(
        (
            joint_pos_21,
            joint_vel_21,
            root_height,
            root_gravity_3d,
            root_lin_vel_b,
            root_ang_vel_b,
            key_in_root,
        ),
        dim=-1,
    )
    return frames_64, dt, T


class AMPLoader:
    # Roban S14: 21 DoF (1 waist + 12 legs + 8 arms). AMP obs 64-dim (amp_share 同款)
    JOINT_POS_SIZE = 21
    JOINT_VEL_SIZE = 21
    ROOT_STATE_SIZE = 10  # height(1) + gravity(3) + lin_vel(3) + ang_vel(3)
    END_EFFECTOR_POS_SIZE = 12

    JOINT_POSE_START_IDX = 0
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    JOINT_VEL_START_IDX = JOINT_POSE_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    ROOT_STATE_START_IDX = JOINT_VEL_END_IDX
    ROOT_STATE_END_IDX = ROOT_STATE_START_IDX + ROOT_STATE_SIZE

    END_POS_START_IDX = ROOT_STATE_END_IDX
    END_POS_END_IDX = END_POS_START_IDX + END_EFFECTOR_POS_SIZE

    def __init__(
        self,
        device,
        time_between_frames,
        data_dir="",
        preload_transitions=False,
        num_preload_transitions=1000000,
        motion_files=None,
    ):
        """Expert dataset provides AMP observations.

        motion_files: list of paths, or dict {path: weight} (same as amp_share).
        Supports .txt/.json (PB_AMP Frames format) and .npz (amp_share format).
        time_between_frames: Amount of time in seconds between transition.
        """
        if motion_files is None:
            motion_files = glob.glob("datasets/motion_amp_expert/*")
        if isinstance(motion_files, dict):
            motion_file_list = list(motion_files.keys())
            weights_from_config = np.array(list(motion_files.values()), dtype=np.float64)
            weights_from_config = weights_from_config / np.sum(weights_from_config)
        else:
            motion_file_list = list(motion_files)
            weights_from_config = None

        self.device = device
        self.time_between_frames = time_between_frames

        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        for i, motion_file in enumerate(motion_file_list):
            self.trajectory_names.append(os.path.splitext(os.path.basename(motion_file))[0])
            if motion_file.lower().endswith(".npz"):
                frames_64, dt, num_frames = _load_npz_to_amp_64(device, motion_file)
                self.trajectories.append(frames_64)
                self.trajectories_full.append(frames_64)
                self.trajectory_idxs.append(i)
                w = float(weights_from_config[i]) if weights_from_config is not None else 1.0
                self.trajectory_weights.append(w)
                self.trajectory_frame_durations.append(dt)
                traj_len = (num_frames - 1) * dt
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(num_frames))
                print(f"Loaded {traj_len:.2f}s motion from {motion_file} (npz, weight={w:.3f}).")
            else:
                with open(motion_file) as f:
                    motion_json = json.load(f)
                    motion_data = np.array(motion_json["Frames"])
                    num_cols = motion_data.shape[1]
                    if num_cols == 67:
                        # 旧 67 维格式：root_rotation 6D → 转为 64 维（用 normal 取反得 gravity）
                        # [0:43], -[46:49], [49:67] -> 43+3+18=64
                        part0 = motion_data[:, :43]
                        normal = motion_data[:, 46:49]
                        gravity_3d = -normal
                        part1 = motion_data[:, 49:67]
                        frames_64 = np.concatenate([part0, gravity_3d, part1], axis=1)
                        motion_tensor = torch.tensor(frames_64, dtype=torch.float32, device=device)
                    else:
                        motion_tensor = torch.tensor(
                            motion_data[:, : AMPLoader.END_POS_END_IDX], dtype=torch.float32, device=device
                        )
                    self.trajectories.append(motion_tensor)
                    self.trajectories_full.append(motion_tensor)
                    self.trajectory_idxs.append(i)
                    if weights_from_config is not None:
                        w = float(weights_from_config[i])
                    else:
                        w = float(motion_json.get("MotionWeight", 1.0))
                    self.trajectory_weights.append(w)
                    frame_duration = float(motion_json["FrameDuration"])
                    self.trajectory_frame_durations.append(frame_duration)
                    traj_len = (motion_data.shape[0] - 1) * frame_duration
                    self.trajectory_lens.append(traj_len)
                    self.trajectory_num_frames.append(float(motion_data.shape[0]))
                    print(f"Loaded {traj_len:.2f}s motion from {motion_file} (weight={w:.3f}).")

        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights, dtype=np.float64)
        self.trajectory_weights = self.trajectory_weights / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f"Preloading {num_preload_transitions} transitions")
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)

            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print("Finished preloading")

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]

        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst

        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, frame1, frame2, blend):
        return (1.0 - blend) * frame1 + blend * frame2

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low

        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int64), np.ceil(p * n).astype(np.int64)
        all_frame_amp_starts = torch.zeros(
            len(traj_idxs), AMPLoader.END_POS_END_IDX - AMPLoader.JOINT_POSE_START_IDX, device=self.device
        )
        all_frame_amp_ends = torch.zeros(
            len(traj_idxs), AMPLoader.END_POS_END_IDX - AMPLoader.JOINT_POSE_START_IDX, device=self.device
        )
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][
                :, AMPLoader.JOINT_POSE_START_IDX : AMPLoader.END_POS_END_IDX
            ]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][
                :, AMPLoader.JOINT_POSE_START_IDX : AMPLoader.END_POS_END_IDX
            ]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        return torch.cat([amp_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        joints0, joints1 = AMPLoader.get_joint_pose(frame0), AMPLoader.get_joint_pose(frame1)
        joint_vel_0, joint_vel_1 = AMPLoader.get_joint_vel(frame0), AMPLoader.get_joint_vel(frame1)

        blend_joint_q = self.slerp(joints0, joints1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat([blend_joint_q, blend_joints_vel])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.preloaded_s[idxs, AMPLoader.JOINT_POSE_START_IDX : AMPLoader.END_POS_END_IDX]
                s_next = self.preloaded_s_next[idxs, AMPLoader.JOINT_POSE_START_IDX : AMPLoader.END_POS_END_IDX]
            else:
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                for traj_idx, frame_time in zip(traj_idxs, times):
                    s.append(self.get_frame_at_time(traj_idx, frame_time))
                    s_next.append(self.get_frame_at_time(traj_idx, frame_time + self.time_between_frames))

                s = torch.vstack(s)
                s_next = torch.vstack(s_next)
            yield s, s_next

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_joint_pose(pose):
        return pose[AMPLoader.JOINT_POSE_START_IDX : AMPLoader.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(poses):
        return poses[:, AMPLoader.JOINT_POSE_START_IDX : AMPLoader.JOINT_POSE_END_IDX]

    def get_joint_vel(pose):
        return pose[AMPLoader.JOINT_VEL_START_IDX : AMPLoader.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(poses):
        return poses[:, AMPLoader.JOINT_VEL_START_IDX : AMPLoader.JOINT_VEL_END_IDX]

    def get_end_pos(pose):
        return pose[AMPLoader.END_POS_START_IDX : AMPLoader.END_POS_END_IDX]

    def get_end_pos_batch(poses):
        return poses[:, AMPLoader.END_POS_START_IDX : AMPLoader.END_POS_END_IDX]
