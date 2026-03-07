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

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv
    from legged_lab.envs.tienkung.tienkung_env import TienKungEnv


def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv | TienKungEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: BaseEnv | TienKungEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def ang_acc_xy_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """与 amp_roban_share 等价的 xy 平面角加速度 L2 惩罚。

    - 使用世界系下质心角速度 `root_com_ang_vel_w[:, :2]`（roll, pitch 分量）；
    - 通过 (vel_t - vel_{t-1}) / dt 近似角加速度；
    - 返回每个 env 的 L2^2 惩罚（需配合负权重使用，如 -5e-5）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 当前 roll/pitch 角速度（世界系）
    ang_vel = asset.data.root_com_ang_vel_w[:, :2]

    # 初始化上一时刻角速度缓存
    if not hasattr(env, "_prev_ang_vel_xy"):
        env._prev_ang_vel_xy = ang_vel.clone()

    # 角加速度近似
    ang_acc = (ang_vel - env._prev_ang_vel_xy) / env.step_dt
    env._prev_ang_vel_xy = ang_vel.clone()

    # L2 penalty
    return torch.sum(torch.square(ang_acc), dim=1)


def joint_power_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """与 amp_roban_share 等价的关节功率 L2 惩罚。

    使用 (applied_torque * joint_vel) 计算每个关节的瞬时功率，并对指定关节求绝对值和。
    返回正值（功率越大越大），需配合负权重使用。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_power = asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(joint_power), dim=1)


def energy(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_torques_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """关节力矩 L2 惩罚：对指定关节的 applied_torque 平方和，需配合负权重使用。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_mean_acc_l2(
    env: BaseEnv | TienKungEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 8000.0,
) -> torch.Tensor:
    """与 amp_roban_share 等价的“平均关节加速度 L2”惩罚。

    - 使用 (joint_vel_t - joint_vel_{t-1}) / dt 近似每个关节的加速度；
    - 对平方加速度减去阈值后 clamp 到 [0, 7500]，再对关节求和；
    - 返回值需配合负权重（如 -6e-6）使用。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 当前速度
    joint_vel = asset.data.joint_vel
    # 上一时刻速度缓存
    if not hasattr(env, "_prev_joint_vel_for_acc"):
        env._prev_joint_vel_for_acc = joint_vel.clone()
    square_acc = torch.square((joint_vel - env._prev_joint_vel_for_acc) / env.step_dt)
    # 与 amp_roban_share 一致：减去阈值后 clamp 到 [0, 7500]
    square_acc = torch.clamp(square_acc - threshold, min=0.0, max=7500.0)
    acc_l2 = torch.sum(square_acc[:, asset_cfg.joint_ids], dim=1)
    env._prev_joint_vel_for_acc = joint_vel.clone()
    return acc_l2


def joint_pos_limits(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """与 IsaacLab/amp_roban_share 等价的软关节位置限幅惩罚。

    使用 ArticulationData.soft_joint_pos_limits 作为软限幅，超出部分累加为正惩罚。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def action_rate_l2(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )


def action_smoothness_l2(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    """与 amp_roban_share 等价的三点差分动作平滑惩罚。

    使用当前动作、上一步动作、上上步动作：
        a_t - 2 * a_{t-1} + a_{t-2}
    的 L2 范数平方作为平滑性成本（需配合负权重使用）。
    """
    buf = env.action_buffer._circular_buffer.buffer
    # 需要至少 3 步历史；若不足则返回 0
    if buf.shape[1] < 3:
        return torch.zeros(env.num_envs, device=buf.device)
    a_t = buf[:, -1, :]
    a_t_1 = buf[:, -2, :]
    a_t_2 = buf[:, -3, :]
    smooth = a_t - 2.0 * a_t_1 + a_t_2
    return torch.sum(torch.square(smooth), dim=1)


def undesired_contacts(env: BaseEnv | TienKungEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv | TienKungEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def orientation_l2(
    env: BaseEnv | TienKungEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    angular_threshold: float = 0.1,
) -> torch.Tensor:
    """与 amp_roban_share 等价的“直行时关节偏移惩罚”（orientation_l2）。

    - 当 yaw 命令 |ω_cmd| < angular_threshold 时，惩罚关节相对默认位置的 L1 偏移；
    - 转弯时（|ω_cmd| 较大）不施加惩罚，给关节更多自由度。
    """
    cmd = env.command_generator.command
    ang_vel_cmd = cmd[:, 2]
    is_straight = torch.abs(ang_vel_cmd) < angular_threshold

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_pos_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.abs(joint_pos - joint_pos_default), dim=1)
    return torch.where(is_straight, deviation, torch.zeros_like(deviation))


def is_terminated(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def stand_still_penalty(
    env: BaseEnv,
    joint_vel_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint movement when velocity commands are inside the deadband.

    当指令落在统一死区内（is_command_active == False）时，机器人应像雕塑一样
    完全静止。如果关节仍在运动（norm(joint_vel) > threshold），施加惩罚。

    与步态奖励的死区门控配合使用：
    - 有指令时：步态奖励生效，静止惩罚不生效
    - 无指令时：步态奖励归零，静止惩罚生效

    Args:
        env: 环境实例（需要有 is_command_active 属性）
        joint_vel_threshold: 关节速度范数容忍上限 (rad/s)
        asset_cfg: 机器人配置

    Returns:
        正值惩罚（使用负权重），仅在无指令且关节仍在运动时非零
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel_norm = torch.norm(asset.data.joint_vel, dim=1)

    # 使用统一死区 mask
    if hasattr(env, "is_command_active"):
        is_active = env.is_command_active
    else:
        # 回退逻辑（兼容旧环境）
        cmd = env.command_generator.command
        is_active = (torch.norm(cmd[:, :2], dim=1) > 0.2) | (torch.abs(cmd[:, 2]) > 0.2)

    should_stand = ~is_active
    penalty = torch.clamp(joint_vel_norm - joint_vel_threshold, min=0.0)
    return penalty * should_stand.float()


def stand_static_vel_without_cmd(
    env: BaseEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_threshold: float = 0.05,
    ext_force_threshold: float = 10.0,
) -> torch.Tensor:
    """与 amp_roban_share 等价的“无命令静止速度惩罚”。

    - 当速度命令接近 0 且外力较小时，惩罚所有关节的速度（L1 范数）；
    - 鼓励在无命令时机器人关节完全静止。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    vel_penalty = torch.sum(torch.abs(joint_vel), dim=-1)

    command = env.command_generator.command
    is_static_cmd = (torch.norm(command[:, :2], dim=1) < cmd_threshold) & (
        torch.abs(command[:, 2]) < cmd_threshold
    )

    if hasattr(asset, "_external_force_b"):
        is_low_ext_force = torch.norm(asset._external_force_b[:, 0, :], dim=1) < ext_force_threshold
    else:
        is_low_ext_force = torch.ones_like(is_static_cmd, dtype=torch.bool)

    valid_mask = is_static_cmd & is_low_ext_force
    return vel_penalty * valid_mask.float()


def stand_still_without_cmd(
    env: BaseEnv | TienKungEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """与 amp_roban_share 等价的“无命令时关节偏移惩罚”。

    - 当线速度命令和角速度命令都很小且外力不大时：
      惩罚关节相对默认位置的 L1 偏移；
    - 用于鼓励在无速度命令下机器人尽量回到默认站姿。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(diff_angle), dim=-1)

    cmd = env.command_generator.command
    lin_ok = torch.norm(cmd[:, :2], dim=1) < 0.1
    ang_ok = torch.abs(cmd[:, 2]) < 0.1
    if hasattr(asset, "_external_force_b"):
        ext_ok = torch.norm(asset._external_force_b[:, 0, :], dim=1) < 20.0
    else:
        ext_ok = torch.ones_like(lin_ok, dtype=torch.bool)

    mask = lin_ok & ang_ok & ext_ok
    return reward * mask.float()


def feet_parallel_when_standing(
    env: BaseEnv | TienKungEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="leg_[l,r]6_link"),
    std: float = 0.02,
) -> torch.Tensor:
    """与 amp_roban_share 等价：静止站立时奖励双脚在根坐标系 x 方向对齐（平行）。

    - 仅在线速度/角速度命令都很小（站立指令）时激活；
    - 在机器人根坐标系下，比较左右脚 x 坐标的差异，差越小奖励越高（高斯核）。
    """
    cmd = env.command_generator.command
    is_standing_still = (
        (torch.abs(cmd[:, 0]) < 0.1)
        & (torch.abs(cmd[:, 1]) < 0.1)
        & (torch.abs(cmd[:, 2]) < 0.1)
    )

    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # [N,2,3]
    root_pos = asset.data.root_pos_w
    left_rel = feet_pos_w[:, 0, :] - root_pos
    right_rel = feet_pos_w[:, 1, :] - root_pos

    root_quat = asset.data.root_quat_w
    left_root = math_utils.quat_rotate_inverse(root_quat, left_rel)
    right_root = math_utils.quat_rotate_inverse(root_quat, right_rel)

    left_x = left_root[:, 0]
    right_x = right_root[:, 0]
    x_err = torch.abs(left_x - right_x)
    parallel_reward = torch.exp(-torch.square(x_err) / (std**2))

    return torch.where(is_standing_still, parallel_reward, torch.zeros_like(parallel_reward))


def foot_lift_height_during_rotation(
    env: BaseEnv | TienKungEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.06,
    height_tolerance: float = 0.02,
    rotation_threshold: float = 0.1,
    linear_threshold: float = 0.1,
    max_air_time: float = 0.3,
) -> torch.Tensor:
    """与 amp_roban_share 等价的 foot_lift_height_during_rotation 奖励。

    - 原地旋转 / 侧走时：鼓励单脚支撑 + 单脚抬到目标高度；
    - 仅在 `is_rotating_or_steponplace & one_foot_support` 且 air_time 合法时生效。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取命令（PB_AMP 使用 command_generator）
    cmd = env.command_generator.command
    lin_vel_x = cmd[:, 0]
    lin_vel_y = cmd[:, 1]
    ang_vel_z = cmd[:, 2]

    # 原地旋转 或 侧向行走
    is_rotating_or_steponplace = (
        (torch.abs(ang_vel_z) > rotation_threshold)
        & (torch.abs(lin_vel_x) < linear_threshold)
        & (torch.abs(lin_vel_y) < linear_threshold)
    ) | (
        (torch.abs(ang_vel_z) < rotation_threshold)
        & (torch.abs(lin_vel_x) < linear_threshold)
        & (torch.abs(lin_vel_y) > linear_threshold)
    )

    # 抬腿时间不能过长
    is_air_time_valid = (
        contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] < max_air_time
    )

    # 脚高度
    feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    # 接触状态
    in_contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 5.0
    num_feet_in_contact = torch.sum(in_contact.float(), dim=1)
    one_foot_support = num_feet_in_contact == 1

    # 高度奖励（高斯核）
    height_error = torch.abs(feet_height - target_height)
    height_reward = torch.exp(-torch.square(height_error / height_tolerance))

    # 只对离地脚且 air_time 合法的脚给奖励
    lift_reward = torch.where(~in_contact, height_reward, torch.zeros_like(height_reward))
    lift_reward = torch.where(is_air_time_valid, lift_reward, torch.zeros_like(lift_reward))
    lift_reward_sum = torch.sum(lift_reward, dim=1)

    reward = torch.where(
        is_rotating_or_steponplace & one_foot_support,
        lift_reward_sum,
        torch.zeros_like(lift_reward_sum),
    )
    return reward


def fft_dof_symmetry(
    env: BaseEnv | TienKungEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names_pairs: list[str] | None = None,
    angular_threshold: float = 0.8,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """与 amp_roban_share 等价的关节频域对称性惩罚。

    - 对每对左右关节，维护一段动作历史（当前 + 前 99 步），在时间维上做 FFT；
    - 比较左右关节在频域上的幅值差异，差异越大 → 惩罚越大（返回负值）；
    - 外力较大或大角速度转弯时弱化惩罚。
    """
    if joint_names_pairs is None:
        joint_names_pairs = ["leg_[l,r]6_joint"]

    # 初始化历史缓存：[num_envs, num_pairs*2, history_len]
    history_len = 100
    if not hasattr(env, "joint_action_history_sym"):
        env.joint_action_history_sym = torch.zeros(
            env.num_envs, len(joint_names_pairs) * 2, history_len, device=env.device
        )
    env.joint_action_history_sym = torch.roll(env.joint_action_history_sym, 1, dims=2)

    asset: Articulation = env.scene[asset_cfg.name]
    symmetry_metric = torch.zeros(env.num_envs, device=env.device)

    # 当前动作（假定与关节顺序一致）
    actions = env.action  # [num_envs, num_actions]

    for i, joint_pair in enumerate(joint_names_pairs):
        joint_idxs = asset.find_joints(joint_pair)[0]
        # asset.find_joints 返回的是 Python 列表，这里按 amp_share 语义要求必须是成对关节
        if len(joint_idxs) != 2:
            continue
        joint_action = actions[:, joint_idxs]  # [N, 2]
        env.joint_action_history_sym[:, i * 2 : i * 2 + 2, 0] = joint_action

        joint_history = env.joint_action_history_sym[:, i * 2 : i * 2 + 2]  # [N,2,T]
        joint_history_centered = joint_history - joint_history.mean(dim=2, keepdim=True)

        fft_vals = torch.fft.rfft(joint_history_centered, dim=2)
        fft_magnitudes = torch.abs(fft_vals)

        freq_diff = torch.abs(fft_magnitudes[:, 0, :] - fft_magnitudes[:, 1, :])
        topk = min(20, freq_diff.shape[1])
        topk_vals = torch.topk(freq_diff, k=topk, dim=1).values
        symmetry_metric += torch.sum(topk_vals, dim=1)

    # 外力与大角速度下削弱惩罚
    if hasattr(asset, "_external_force_b"):
        without_external_force_apply = torch.norm(asset._external_force_b[:, 0, :], dim=1) < 5.0
    else:
        without_external_force_apply = torch.ones_like(symmetry_metric, dtype=torch.bool)

    cmd = env.command_generator.command
    big_angular = torch.abs(cmd[:, 2]) > angular_threshold

    symmetry_metric = symmetry_metric * without_external_force_apply
    symmetry_metric[big_angular] = symmetry_metric[big_angular] * 0.5

    # 返回负惩罚，clip 上限与 amp_roban_share 一致
    return -torch.clamp(symmetry_metric, max=300.0)


def feet_air_time_positive_biped(
    env: BaseEnv | TienKungEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward


def feet_slide(
    env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def feet_slide_yaw(
    env: BaseEnv | TienKungEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """与 amp_roban_share 等价的足部转动滑移惩罚（yaw 方向）。

    - 只在有接触的脚上，根据身体各脚链接的 yaw 角速度大小惩罚；
    - 仅在有 yaw 命令（转向指令幅值 > 0.1）时激活。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    asset: Articulation = env.scene[asset_cfg.name]
    body_ang_vel_z = asset.data.body_ang_vel_w[:, sensor_cfg.body_ids, 2:3]
    reward_ang_vel = torch.sum(body_ang_vel_z.norm(dim=-1) * contacts, dim=1)
    # 仅在有非零 yaw 命令时启用
    cmd = env.command_generator.command
    has_yaw_cmd = torch.abs(cmd[:, 2]) > 0.1
    reward_ang_vel = reward_ang_vel * has_yaw_cmd.float()
    return reward_ang_vel


def body_force(
    env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def contact_forces(
    env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg, threshold: float = 350.0
) -> torch.Tensor:
    """与 amp_share 一致：足底接触力范数超过 threshold 时对超出部分施加惩罚（配合负权重）。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_norm = torch.norm(net_forces, dim=-1)
    excess = torch.clamp(force_norm - threshold, min=0.0)
    return excess.sum(dim=1)


def gravity_aligned_when_stopping(
    env: BaseEnv | TienKungEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["leg_[l,r]6_joint"]),
    std: float = 0.25,
) -> torch.Tensor:
    """与 amp_roban_share 等价的“停止时重力对齐”奖励。

    - 当速度命令几乎为 0 时，奖励躯干姿态与重力方向对齐（即站直）；
    - 使用 pitch 角的高斯奖励，pitch 越接近 0 奖励越高。
    """
    cmd = env.command_generator.command
    is_zero_cmd = torch.norm(cmd[:, :2], dim=1) < 0.1

    asset: Articulation = env.scene[asset_cfg.name]
    root_quat = asset.data.root_quat_w
    w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
    # 近似提取 pitch 角（与 amp_share 中实现保持一致，带微小偏移）
    pitch = torch.asin(2.0 * (w * y - x * z)) - 0.02
    reward = torch.exp(-torch.square(pitch) / std)

    masked_reward = torch.zeros_like(reward)
    masked_reward[is_zero_cmd] = reward[is_zero_cmd]
    return masked_reward


def joint_deviation_l1(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    zero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) < 0.1
    return torch.sum(torch.abs(angle), dim=1) * zero_flag


def joint_deviation_l1_straight_only(
    env: BaseEnv | TienKungEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    angular_threshold: float = 0.1,
) -> torch.Tensor:
    """与 amp_roban_share 等价：仅在“直行”时惩罚关节偏离默认。

    - yaw 命令 |ω_cmd| < angular_threshold 时：
        deviation = ∑ |joint_pos - default_joint_pos|
    - 转弯时（|ω_cmd| 较大）不施加惩罚（返回 0），允许关节有更多自由度。
    """
    cmd = env.command_generator.command
    ang_vel_cmd = cmd[:, 2]
    is_straight = torch.abs(ang_vel_cmd) < angular_threshold

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_pos_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.abs(joint_pos - joint_pos_default), dim=1)
    return torch.where(is_straight, deviation, torch.zeros_like(deviation))


def joint_vel_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """与 amp_roban_share 等价：关节速度 L2 惩罚。

    使用 joint_vel 相对于默认速度（通常为 0）的平方和，需配合负权重使用。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_vel), dim=1)


def feet_pos_acc_l2(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """与 amp_roban_share 等价：脚部位置加速度 L2 惩罚。

    - 使用 body_pos_w 计算脚的速度和加速度；
    - 对加速度 clamp 后平方求和。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]

    if not hasattr(env, "_prev_feet_pos"):
        env._prev_feet_pos = feet_pos.clone()

    feet_vel = (feet_pos - env._prev_feet_pos) / env.step_dt

    if not hasattr(env, "_prev_feet_vel"):
        env._prev_feet_vel = feet_vel.clone()

    feet_acc = (feet_vel - env._prev_feet_vel) / env.step_dt
    feet_acc = torch.clamp(feet_acc, min=-30.0, max=30.0)

    env._prev_feet_pos = feet_pos.clone()
    env._prev_feet_vel = feet_vel.clone()

    return torch.sum(torch.square(feet_acc), dim=(1, 2))


def knee_pos_acc_l2(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """与 amp_roban_share 等价：膝盖位置加速度 L2 惩罚。"""
    asset: Articulation = env.scene[asset_cfg.name]
    knee_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]

    if not hasattr(env, "_prev_knee_pos"):
        env._prev_knee_pos = knee_pos.clone()

    knee_vel = (knee_pos - env._prev_knee_pos) / env.step_dt

    if not hasattr(env, "_prev_knee_vel"):
        env._prev_knee_vel = knee_vel.clone()

    knee_acc = (knee_vel - env._prev_knee_vel) / env.step_dt
    knee_acc = torch.clamp(knee_acc, min=-30.0, max=30.0)

    env._prev_knee_pos = knee_pos.clone()
    env._prev_knee_vel = knee_vel.clone()

    return torch.sum(torch.square(knee_acc), dim=(1, 2))


def knee_pos_z_limit(
    env: BaseEnv | TienKungEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    knee_z_limit: float = 0.58,
) -> torch.Tensor:
    """与 amp_roban_share 等价：膝盖高度超限惩罚。

    - 若膝盖的 z 高度超过 knee_z_limit，则对超出部分做二次惩罚；
    - 用于约束膝盖过高抬起，保持自然行走姿态。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    knee_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # [N, n_knee, 3]
    knee_z = knee_pos[:, :, 2]
    excess = (knee_z - knee_z_limit).clamp(min=0.0)
    return torch.sum(torch.square(excess), dim=1)


def knee_ankle_y_position_difference(
    env: BaseEnv | TienKungEnv,
    knee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="leg_[l,r]4_link"),
    ankle_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="leg_[l,r]6_link"),
) -> torch.Tensor:
    """与 amp_roban_share 等价的膝盖-脚踝 y 方向位置差惩罚。

    计算在机器人根坐标系下，左右两条腿膝盖和脚踝 y 方向位置差的绝对值之和。
    返回正值（差越大越大），需配合负权重使用。
    """
    asset: Articulation = env.scene["robot"]
    # 世界系下 body 位置
    knee_pos_w = asset.data.body_link_pos_w[:, knee_cfg.body_ids, :]
    ankle_pos_w = asset.data.body_link_pos_w[:, ankle_cfg.body_ids, :]
    # 根姿态
    root_quat = asset.data.root_link_quat_w
    # 转到根坐标系
    knee_body = math_utils.quat_rotate_inverse(root_quat.unsqueeze(1), knee_pos_w)
    ankle_body = math_utils.quat_rotate_inverse(root_quat.unsqueeze(1), ankle_pos_w)
    # 取 y 分量
    knee_y = knee_body[:, :, 1]
    ankle_y = ankle_body[:, :, 1]
    y_diff = torch.abs(knee_y - ankle_y)
    total_y_diff = torch.sum(y_diff, dim=1)
    return total_y_diff


def joint_pos_tracking_l2(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Track joint positions relative to default positions using L2 norm.
    This reward encourages joints to stay close to their default/rest positions.
    
    Args:
        env: The environment instance
        asset_cfg: Scene entity configuration specifying which joints to track
        
    Returns:
        L2 norm of joint position deviations (lower is better, so use negative weight)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_pos_diff), dim=1)


def joint_pos_tracking_exp(
    env: BaseEnv | TienKungEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Track joint positions relative to default positions using exponential reward.
    This reward gives higher values when joints are close to their default positions.
    
    Args:
        env: The environment instance
        std: Standard deviation for the exponential reward (controls sensitivity)
        asset_cfg: Scene entity configuration specifying which joints to track
        
    Returns:
        Exponential reward based on joint position deviations (higher is better)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    error = torch.sum(torch.square(joint_pos_diff), dim=1)
    return torch.exp(-error / std**2)


def joint_vel_tracking_l2(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Track joint velocities relative to default velocities (usually zero).
    This reward encourages smooth, controlled joint movements.
    
    Args:
        env: The environment instance
        asset_cfg: Scene entity configuration specifying which joints to track
        
    Returns:
        L2 norm of joint velocity deviations (lower is better, so use negative weight)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel_diff = asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_vel_diff), dim=1)


def deadzone_penalty(
    env: BaseEnv,
    deadzone: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize hip roll (L2/R2) joints when they deviate beyond a dead zone from default.

    Since L1/R1 axes are at 45° with axis=(0, ±0.707, -0.707), they mix pitch
    and yaw but have NO X-component. This means L2 (roll, axis=(1,0,0)) does
    NOT need to compensate for L1 motion during forward walking.

    A dead zone allows minor balance adjustments while preventing leg splaying.
    Recommended dead zone: 0.1–0.15 rad for forward walking.

    Args:
        env: The environment instance
        deadzone: Maximum allowed deviation from default before penalty applies (rad)
        asset_cfg: Scene entity configuration for hip roll joints

    Returns:
        Quadratic penalty for deviations beyond the dead zone
    """
    asset: Articulation = env.scene[asset_cfg.name]
    deviation = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    excess = torch.abs(deviation) - deadzone
    excess = torch.clamp(excess, min=0.0)
    return torch.sum(torch.square(excess), dim=1)


def hip_roll_conditional_penalty(
    env: BaseEnv,
    deadzone_base: float = 0.1,
    deadzone_max: float = 0.3,
    vel_threshold: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Velocity-conditioned hip roll penalty. Dead zone expands when lateral
    velocity command is large, allowing proper lateral stepping while
    preventing splay during forward walking.

    When lateral velocity is near zero → tight dead zone (deadzone_base)
    When lateral velocity is high → expanded dead zone (deadzone_max)

    Args:
        env: The environment instance
        deadzone_base: Dead zone when not moving laterally (rad), default 0.1
        deadzone_max: Dead zone when moving laterally at full speed (rad), default 0.3
        vel_threshold: Lateral velocity at which dead zone reaches maximum (m/s)
        asset_cfg: Scene entity configuration for hip roll joints

    Returns:
        Quadratic penalty for deviations beyond the velocity-conditioned dead zone
    """
    asset: Articulation = env.scene[asset_cfg.name]
    deviation = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    # Compute velocity-dependent dead zone
    lateral_vel_cmd = torch.abs(env.command_generator.command[:, 1])
    vel_ratio = torch.clamp(lateral_vel_cmd / vel_threshold, max=1.0)
    deadzone = deadzone_base + (deadzone_max - deadzone_base) * vel_ratio

    # Apply dead zone (broadcast deadzone to per-joint dimension)
    excess = torch.abs(deviation) - deadzone.unsqueeze(1)
    excess = torch.clamp(excess, min=0.0)
    return torch.sum(torch.square(excess), dim=1)


def hip_yaw_conditional_penalty(
    env: BaseEnv,
    deadzone_base: float = 0.15,
    deadzone_max: float = 0.4,
    vel_threshold: float = 0.5,
    toein_deadzone_ratio: float = 0.5,
    toein_signs: list[float] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Angular-velocity-conditioned hip yaw (L3/R3) penalty with asymmetric
    dead zone: toe-in direction is penalized more heavily.

    Dead zone expands when angular velocity command |ω_z| is large (turning),
    and is tighter in the toe-in direction to prevent pigeon-toed posture.

    Toe-in direction:
        L3 (default=-0.287): deviation < 0 → toein_sign = -1
        R3 (default=+0.287): deviation > 0 → toein_sign = +1

    Args:
        env: The environment instance
        deadzone_base: Dead zone when not turning (rad)
        deadzone_max: Dead zone when turning at full speed (rad)
        vel_threshold: Angular velocity at which dead zone reaches maximum (rad/s)
        toein_deadzone_ratio: Toe-in dead zone = deadzone * this ratio (< 1.0 = tighter)
        toein_signs: Per-joint sign indicating toe-in direction, e.g. [-1.0, 1.0]
                     for [L3, R3]. If None, penalty is symmetric.
        asset_cfg: Scene entity configuration for hip yaw joints

    Returns:
        Quadratic penalty for deviations beyond the asymmetric dead zone
    """
    asset: Articulation = env.scene[asset_cfg.name]
    deviation = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    # Velocity-dependent base dead zone (angular velocity)
    ang_vel_cmd = torch.abs(env.command_generator.command[:, 2])
    vel_ratio = torch.clamp(ang_vel_cmd / vel_threshold, max=1.0)
    deadzone = deadzone_base + (deadzone_max - deadzone_base) * vel_ratio  # [num_envs]

    # Asymmetric dead zone: tighter in toe-in direction
    if toein_signs is not None:
        signs = torch.tensor(toein_signs, device=asset.device, dtype=torch.float32)  # [num_joints]
        # is_toein: True when deviation is in toe-in direction
        is_toein = (deviation * signs.unsqueeze(0)) > 0  # [num_envs, num_joints]
        # Toe-in gets tighter dead zone, toe-out gets normal dead zone
        dz_expanded = deadzone.unsqueeze(1).expand_as(deviation)  # [num_envs, num_joints]
        effective_deadzone = torch.where(is_toein, dz_expanded * toein_deadzone_ratio, dz_expanded)
    else:
        effective_deadzone = deadzone.unsqueeze(1)

    excess = torch.abs(deviation) - effective_deadzone
    excess = torch.clamp(excess, min=0.0)
    return torch.sum(torch.square(excess), dim=1)


def body_orientation_l2(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_rotate_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_orientation_l2(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalty for foot orientation deviation from horizontal (anti-edge-walking/tiptoeing).

    Sum of L2 orientation error over all bodies in asset_cfg (e.g. left and right foot).
    Use with negative weight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    total = torch.zeros(env.num_envs, device=asset.data.body_quat_w.device)
    for bid in asset_cfg.body_ids:
        body_orientation = math_utils.quat_rotate_inverse(
            asset.data.body_quat_w[:, bid, :], asset.data.GRAVITY_VEC_W
        )
        total = total + torch.sum(torch.square(body_orientation[:, :2]), dim=1)
    return total


def feet_stumble(env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_too_near_humanoid(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_distance_penalty(
    env: BaseEnv | TienKungEnv,
    min_dist: float = 0.20,
    max_dist: float = 0.32,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize when the distance between the two feet is outside [min_dist, max_dist].

    - distance < min_dist: penalty = (min_dist - distance)^2  (too close)
    - distance > max_dist: penalty = (distance - max_dist)^2  (too far)
    - min_dist <= distance <= max_dist: penalty = 0            (OK)

    Uses 3D Euclidean distance between the two foot body positions.

    Args:
        env: The environment instance
        min_dist: Minimum allowed distance between feet (m)
        max_dist: Maximum allowed distance between feet (m)
        asset_cfg: Scene entity configuration with exactly 2 body names (left/right foot)

    Returns:
        Quadratic penalty for feet distance outside the allowed range
    """
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)

    too_close = (min_dist - distance).clamp(min=0.0)
    too_far = (distance - max_dist).clamp(min=0.0)
    return torch.square(too_close) + torch.square(too_far)


# Regularization Reward
def ankle_torque(env: TienKungEnv) -> torch.Tensor:
    """Penalize large torques on the ankle joints."""
    return torch.sum(torch.square(env.robot.data.applied_torque[:, env.ankle_joint_ids]), dim=1)


def ankle_action(env: TienKungEnv) -> torch.Tensor:
    """Penalize ankle joint actions."""
    return torch.sum(torch.abs(env.action[:, env.ankle_joint_ids]), dim=1)


def hip_roll_action(env: TienKungEnv) -> torch.Tensor:
    """Penalize hip roll joint actions."""
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[0], env.right_leg_ids[0]]]), dim=1)


def hip_yaw_action(env: TienKungEnv) -> torch.Tensor:
    """Penalize hip yaw joint actions."""
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[2], env.right_leg_ids[2]]]), dim=1)


def feet_y_distance(env: TienKungEnv) -> torch.Tensor:
    """Penalize foot y-distance when the commanded y-velocity is low, to maintain a reasonable spacing."""
    leftfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[0], :] - env.robot.data.root_link_pos_w[:, :]
    rightfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[1], :] - env.robot.data.root_link_pos_w[:, :]
    leftfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), leftfoot)
    rightfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), rightfoot)
    y_distance_b = torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1] - 0.299)
    y_vel_flag = torch.abs(env.command_generator.command[:, 1]) < 0.1
    return y_distance_b * y_vel_flag


# Periodic gait-based reward function
def gait_clock(phase, air_ratio, delta_t):
    """
    Generate periodic gait clock signals for foot swing and stance phases.

    This function constructs two phase-dependent signals:
    - `I_frc`: active during swing phase (used for penalizing ground force)
    - `I_spd`: active during stance phase (used for penalizing foot speed)

    Transitions between swing and stance are smoothed within a margin of `delta_t`
    to create differentiable transitions.

    Parameters
    ----------
    phase : torch.Tensor
        Normalized gait phase in [0, 1], shape: [num_envs].
    air_ratio : torch.Tensor
        Proportion of the gait cycle spent in swing phase, shape: [num_envs].
    delta_t : float
        Transition width around phase boundaries for smooth interpolation.

    Returns
    -------
    I_frc : torch.Tensor
        Gait-based swing-phase clock signal, range [0, 1], shape: [num_envs].
    I_spd : torch.Tensor
        Gait-based stance-phase clock signal, range [0, 1], shape: [num_envs].

    Notes
    -----
    - The transitions at the boundaries (e.g., swing→stance) are linear interpolations.
    - Used in reward shaping to associate expected behavior with gait phases.
    """
    swing_flag = (phase >= delta_t) & (phase <= (air_ratio - delta_t))
    stand_flag = (phase >= (air_ratio + delta_t)) & (phase <= (1 - delta_t))

    trans_flag1 = phase < delta_t
    trans_flag2 = (phase > (air_ratio - delta_t)) & (phase < (air_ratio + delta_t))
    trans_flag3 = phase > (1 - delta_t)

    I_frc = (
        1.0 * swing_flag
        + (0.5 + phase / (2 * delta_t)) * trans_flag1
        - (phase - air_ratio - delta_t) / (2.0 * delta_t) * trans_flag2
        + 0.0 * stand_flag
        + (phase - 1 + delta_t) / (2 * delta_t) * trans_flag3
    )
    I_spd = 1.0 - I_frc
    return I_frc, I_spd


def gait_feet_frc_perio(env: TienKungEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Penalize foot force during the swing phase of the gait."""
    left_frc_swing_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_frc_swing_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    left_frc_score = left_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return left_frc_score + right_frc_score


def gait_feet_spd_perio(env: TienKungEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Penalize foot speed during the support phase of the gait."""
    left_spd_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_spd_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_spd_score = left_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 0])))
    right_spd_score = right_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 1])))
    return left_spd_score + right_spd_score


def gait_feet_frc_support_perio(env: TienKungEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Reward that promotes proper support force during stance (support) phase."""
    left_frc_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_frc_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_frc_score = left_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return left_frc_score + right_frc_score


# ══════════════════════════════════════════════════════════════════════════════
# Natural Landing Rewards
# ══════════════════════════════════════════════════════════════════════════════


def contact_ground_straight_knee(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.1,
) -> torch.Tensor:
    """Reward straight (extended) knees at the moment of touchdown.

    人类自然行走时，落地腿膝盖是接近伸直的。如果机器人弯着膝盖落地，
    往往伴随重心下沉、步态不稳、步幅过大等问题。

    仅在 first_contact（着地瞬间）给出奖励：
        reward = exp(-knee_pos² / std²) * first_contact

    knee_pos 使用原始关节角度（非相对于 default）。
    对于 Roban S14，knee 关节范围 [0, 2.618]，0=完全伸直。
    所以 knee_pos 越接近 0，奖励越大。

    amp_roban_share 参考值：weight=1.0, std=0.3
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]  # [batch, 2]

    asset: Articulation = env.scene[asset_cfg.name]
    joint_idxs = asset.find_joints(asset_cfg.joint_names)[0]

    # exp(-q²/σ²): 膝盖越伸直(q→0)奖励越大
    straight_knee = torch.exp(-torch.square(asset.data.joint_pos[:, joint_idxs] / std))
    straight_knee = straight_knee * first_contact  # 只在着地瞬间

    reward = torch.sum(straight_knee, dim=1)
    # 只在有速度指令时激活
    has_command = (
        torch.norm(env.command_generator.command[:, :2], dim=1)
        + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward * has_command


def contact_momentum(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.05,
) -> torch.Tensor:
    """Reward gentle (low vertical velocity) foot landing.

    与 amp_roban_share 等价的 contact_momentum：
    - 只考虑脚部在 z 方向的速度；
    - 在 first_contact 时，根据 v_z 大小给出 exp(-v_z^2 / std^2) 奖励；
    - 仅在有运动指令时激活。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]  # [N, feet]

    asset: Articulation = env.scene[asset_cfg.name]
    # 使用 sensor_cfg.body_ids 对齐脚部索引，避免与 asset_cfg.body_ids 维度不一致
    body_vel_z = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, 2]  # [N, feet]
    body_vel_z = torch.clamp(body_vel_z, min=-10.0, max=0.0)

    reward = torch.exp(-torch.square(body_vel_z / std)) * first_contact  # [N, feet]
    reward_sum = torch.sum(reward, dim=1)  # [N]

    has_command = (
        torch.norm(env.command_generator.command[:, :2], dim=1)
        + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward_sum * has_command.float()


def excessive_landing_force_penalty(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 200.0,
) -> torch.Tensor:
    """惩罚落地瞬间过大的接触力（落地冲击）。

    仅在脚刚着地（first_contact）的那一帧，若该脚接触力范数超过 threshold，则对超出部分施加惩罚；
    鼓励轻落地、减少冲击。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]  # [N, feet]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # [N, feet, 3]
    force_norm = torch.norm(net_forces, dim=-1)  # [N, feet]
    excess = torch.clamp(force_norm - threshold, min=0.0)
    penalty = (excess * first_contact.float()).sum(dim=1)
    return penalty


# ══════════════════════════════════════════════════════════════════════════════
# Stride / Step Frequency Control Rewards
# ══════════════════════════════════════════════════════════════════════════════


def feet_height_cycle(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_height_clip: float = 0.15,
) -> torch.Tensor:
    """Reward foot swing height per cycle, triggered only at touchdown.

    每个摆动周期只奖励一次：该周期中脚达到的**最大离地高度**。
    仅在脚刚着地（first_contact）的那一步触发奖励，其余步为 0。
    高度 clip 到 max_height_clip 防止过高抬脚。

    逻辑：
    1. 脚在空中时：持续更新该脚本周期的最大高度
    2. 脚刚着地时：输出该最大高度作为奖励，然后重置缓存
    3. 脚在地面时：奖励为 0（不重复给）

    效果：
    - 鼓励抬脚（避免拖地/擦地）
    - max_height_clip=0.15 限制抬脚上限（防止大幅甩腿导致大跨步）
    - 配合 step_frequency_penalty 一起使用效果最佳

    amp_roban_share 参考值：weight=12.0, max_height_clip=0.15
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    # 当前脚的世界坐标 z（离地高度）
    feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # [batch, 2]
    feet_height = torch.nan_to_num(feet_height, nan=0.0, posinf=0.1, neginf=0.0)

    # 是否在接触地面
    in_contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 10.0  # [batch, 2]
    # 本步是否刚着地
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]  # [batch, 2]

    # ── 初始化周期最大高度缓存 ──
    if not hasattr(env, "_feet_cycle_max_height"):
        env._feet_cycle_max_height = torch.zeros_like(feet_height)

    # ── episode 重置时清零 ──
    fresh = env.episode_length_buf <= 1
    if fresh.any():
        env._feet_cycle_max_height[fresh] = 0

    # ── 脚在空中 → 更新本周期最大高度 ──
    not_in_contact = ~in_contact
    env._feet_cycle_max_height = torch.where(
        not_in_contact,
        torch.maximum(env._feet_cycle_max_height, feet_height),
        env._feet_cycle_max_height,
    )

    # ── 刚着地 → 输出奖励 ──
    reward = torch.where(
        first_contact,
        env._feet_cycle_max_height,
        torch.zeros_like(feet_height),
    )

    # ── 刚着地 → 重置缓存（为下一个摆动周期准备）──
    env._feet_cycle_max_height = torch.where(
        first_contact,
        torch.zeros_like(env._feet_cycle_max_height),
        env._feet_cycle_max_height,
    )

    # 双脚求和，clip 上界
    reward = torch.sum(torch.clamp(reward, max=max_height_clip), dim=1)
    return reward


def feet_air_time_clip_biped(
    env: BaseEnv,
    threshold_min: float,
    threshold_max: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward appropriate air time with min/max clipping.

    与 feet_air_time_positive_biped 不同，本函数：
    1. 使用 last_air_time（上一次完整摆动相时长），而非瞬时 current_air_time
    2. 仅在**着地瞬间** (first_contact) 给出奖励
    3. 低于 threshold_min 无奖励，超过 threshold_max 也不再增长

    这样可以控制步幅：
    - threshold_min=0.1: 太短的摆动不算步（防止原地颤抖）
    - threshold_max=0.3: 超过 0.3s 的摆动不再有额外奖励（防止大跨步）

    amp_roban_share 参考值：threshold_min=0.25, threshold_max=0.45
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # first_contact: 本步刚着地的脚 (True)
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # last_air_time: 上一次完整空中相的时长
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    # 减去最小阈值，低于 min 的部分不奖励
    air_time = (last_air_time - threshold_min) * first_contact
    # 裁剪上限
    air_time = torch.clamp(air_time, min=0.0, max=threshold_max - threshold_min)
    reward = torch.sum(air_time, dim=1)
    # 无指令时不奖励
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1)
        + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward


def feet_air_time_clip(
    env: BaseEnv | TienKungEnv,
    command_name: str = "base_velocity",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    threshold_min: float = 0.25,
    threshold_max: float = 0.45,
) -> torch.Tensor:
    """与 amp_roban_share 等价的 feet_air_time_clip 实现。

    - 使用 last_air_time（上一次完整空中相时长）和 first_contact（刚着地）；
    - 对 (last_air_time - threshold_min) 做下截断，超过 threshold_max-threshold_min 的部分剪裁；
    - 仅在有速度命令时激活。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    air_time = (last_air_time - threshold_min) * first_contact
    air_time = torch.clamp(air_time, max=threshold_max - threshold_min)
    reward = torch.sum(air_time, dim=1)

    cmd = env.command_generator.command
    has_cmd = torch.norm(cmd[:, :3], dim=1) > 0.1
    return reward * has_cmd.float()


def feet_air_time_exp(
    env: BaseEnv,
    target_air_time: float = 0.2,
    std: float = 0.08,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
) -> torch.Tensor:
    """滞空时间的高斯核奖励（固定目标）：着地瞬间对上一段滞空时长给高斯奖励。

    reward = exp(-(last_air_time - target_air_time)^2 / std^2)，仅在着地瞬间、有指令时给奖。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    err_sq = torch.square(last_air_time - target_air_time)
    gaussian = torch.exp(-err_sq / (std**2)) * first_contact
    reward = torch.sum(gaussian, dim=1)

    has_command = (
        torch.norm(env.command_generator.command[:, :2], dim=1)
        + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward * has_command.float()


def feet_air_time_exp_speed_adaptive(
    env: BaseEnv,
    target_air_at_speed_low: float = 0.5,
    target_air_at_speed_high: float = 0.3,
    speed_low: float = 0.3,
    speed_high: float = 0.6,
    min_cmd_speed: float = 0.25,
    std: float = 0.08,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
) -> torch.Tensor:
    """滞空时间的高斯核奖励，目标滞空随指令速度线性变化（甜点区）。

    - 速度 = speed_low (0.3) 时目标滞空 = 0.5s；速度 = speed_high (0.6) 时目标滞空 = 0.3s。
    - 速度指令 < min_cmd_speed (0.25) 时不要求抬腿，不给奖励。
    - 抬腿时间超过目标（0.5 或 0.3）时，奖励封顶为目标对应的值（不再额外给，但给满目标分）。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    cmd_speed = torch.norm(env.command_generator.command[:, :2], dim=1)
    t = ((cmd_speed - speed_low) / (speed_high - speed_low)).clamp(0.0, 1.0)
    target_air_time = target_air_at_speed_low - (target_air_at_speed_low - target_air_at_speed_high) * t
    target_air_time = target_air_time.unsqueeze(1)  # [num_envs, 1]

    # 超过目标滞空时按目标算，奖励封顶为“目标滞空”对应的分数（等价于滞空=0.5或0.3时给满）
    effective_air_time = torch.minimum(last_air_time, target_air_time)
    err_sq = torch.square(effective_air_time - target_air_time)
    gaussian = torch.exp(-err_sq / (std**2)) * first_contact

    reward = torch.sum(gaussian, dim=1)
    has_command = cmd_speed >= min_cmd_speed
    return reward * has_command.float()


def step_frequency_penalty(
    env: BaseEnv,
    max_grounded_time: float = 0.2,
    penalty_scale: float = 2.0,
    velocity_threshold: float = 0.12,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
) -> torch.Tensor:
    """Penalize both feet being on the ground for too long.

    当机器人有速度指令时，如果双脚同时着地的时间超过 max_grounded_time，
    施加线性惩罚。这迫使机器人加快步频，而不是慢慢迈大步。

    amp_roban_share 的 turning_step_frequency 使用 max_grounded_time=0.2s。

    返回负值（惩罚），需要配合负权重或反转权重使用。
    注意：返回的是负值惩罚，配合 **正** weight 使用。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 检测双脚是否都在地面
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = net_forces.norm(dim=-1).max(dim=1)[0] > 1.0  # [batch, 2]
    both_feet_grounded = torch.all(in_contact, dim=1)  # [batch]

    # 维护双脚着地时间计数器
    if not hasattr(env, "_both_feet_grounded_time"):
        env._both_feet_grounded_time = torch.zeros(env.num_envs, device=in_contact.device)

    env._both_feet_grounded_time = torch.where(
        both_feet_grounded,
        env._both_feet_grounded_time + env.step_dt,
        torch.zeros_like(env._both_feet_grounded_time),
    )

    # episode 重置时清零
    fresh = env.episode_length_buf <= 1
    if fresh.any():
        env._both_feet_grounded_time[fresh] = 0

    # 超过 max_grounded_time 的部分施加惩罚
    penalty = torch.clamp(env._both_feet_grounded_time - max_grounded_time, min=0.0)

    # 只在有速度指令时激活
    has_command = (
        torch.norm(env.command_generator.command[:, :2], dim=1)
        + torch.abs(env.command_generator.command[:, 2])
    ) > velocity_threshold

    reward = torch.where(has_command, -penalty * penalty_scale, torch.zeros_like(penalty))
    return reward


def turning_step_frequency_reward(
    env: BaseEnv,
    command_name: str = "base_velocity",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    velocity_threshold: float = 0.12,
    max_grounded_time: float = 0.2,
    penalty_scale: float = 2.0,
) -> torch.Tensor:
    """与 amp_roban_share 等价的转弯步频奖励。

    amp_roban_share 中使用的 turning_step_frequency，本质上是对
    \"双脚同时着地时间过长\" 的惩罚，权重为正值（10）。

    这里直接复用 step_frequency_penalty 的实现与默认参数：
    - max_grounded_time = 0.2
    - penalty_scale = 2.0
    - velocity_threshold = 0.12

    注意：返回值为**负值惩罚**，需配合**正的 weight**（例如 10）使用。
    """
    # 复用前面的 step_frequency_penalty 实现，确保行为与注释保持一致
    return step_frequency_penalty(
        env,
        max_grounded_time=max_grounded_time,
        penalty_scale=penalty_scale,
        velocity_threshold=velocity_threshold,
        sensor_cfg=sensor_cfg,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Humanoid Locomotion: Contact Phase Rewards (no air-time shaping)
# Induce natural leg lifting through stable single support.
# ══════════════════════════════════════════════════════════════════════════════


def humanoid_flight_penalty(
    env: BaseEnv | TienKungEnv, threshold: float = 10.0, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor")
) -> torch.Tensor:
    """Strong penalty when both feet are off the ground (flight phase forbidden).

    Returns 1.0 when number_of_contacts == 0, else 0. Use with large negative weight.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = torch.norm(net_forces, dim=-1) > threshold  # [batch, 2]
    num_contacts = torch.sum(in_contact.int(), dim=1)
    return (num_contacts == 0).float()


def humanoid_double_support_penalty(
    env: BaseEnv | TienKungEnv, threshold: float = 10.0, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor")
) -> torch.Tensor:
    """Small penalty when both feet are on the ground (discourage persistent double support).

    Returns 1.0 when number_of_contacts == 2, else 0. Use with small negative weight.
    Only active when command is non-zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = torch.norm(net_forces, dim=-1) > threshold  # [batch, 2]
    num_contacts = torch.sum(in_contact.int(), dim=1)
    has_command = (
        torch.norm(env.command_generator.command[:, :2], dim=1)
        + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return ((num_contacts == 2) & has_command).float()


def humanoid_single_support_reward(
    env: BaseEnv | TienKungEnv, threshold: float = 10.0, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor")
) -> torch.Tensor:
    """Moderate positive reward when exactly one foot is in contact (stable single support).

    Returns 1.0 when number_of_contacts == 1, else 0. Use with moderate positive weight.
    Must be larger than double support penalty. Only active when command is non-zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = torch.norm(net_forces, dim=-1) > threshold  # [batch, 2]
    num_contacts = torch.sum(in_contact.int(), dim=1)
    has_command = (
        torch.norm(env.command_generator.command[:, :2], dim=1)
        + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return ((num_contacts == 1) & has_command).float()


def humanoid_swing_foot_height_reward(
    env: BaseEnv | TienKungEnv,
    threshold: float = 10.0,
    height_threshold: float = 0.05,
    max_height: float = 0.12,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Continuous swing foot height reward during single support only.

    - Only rewards the non-contact (swing) foot.
    - Positive reward for height above height_threshold (e.g. 5 cm).
    - Clipped above max_height (e.g. 10–15 cm) to prevent moonwalk/high kick.
    - Does NOT reward stance foot. Use moderate weight so it does not dominate velocity tracking.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = torch.norm(net_forces, dim=-1) > threshold  # [batch, 2]
    num_contacts = torch.sum(in_contact.int(), dim=1)
    single_support = num_contacts == 1
    has_command = (
        torch.norm(env.command_generator.command[:, :2], dim=1)
        + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1

    feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # [batch, 2] z in world
    feet_height = torch.nan_to_num(feet_height, nan=0.0, posinf=0.15, neginf=0.0)
    excess_height = (feet_height - height_threshold).clamp(min=0.0)
    clipped_height = excess_height.clamp(max=max_height - height_threshold)

    # Swing foot = the one NOT in contact; reward only swing foot height
    swing_height = torch.where(in_contact, torch.zeros_like(clipped_height), clipped_height)
    reward = torch.sum(swing_height, dim=1)
    return reward * single_support.float() * has_command.float()


def humanoid_swing_foot_forward_reward(
    env: BaseEnv | TienKungEnv,
    threshold: float = 10.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward forward displacement of swing foot during single support (anti-moonwalk).

    Swing foot forward offset = (swing_foot_pos - root_pos) dot forward_dir in yaw frame.
    Positive when swing foot is ahead of root. Use small positive weight.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = torch.norm(net_forces, dim=-1) > threshold  # [batch, 2]
    num_contacts = torch.sum(in_contact.int(), dim=1)
    single_support = num_contacts == 1

    # 仅在前行/后退有指令时激活；侧移或纯转向不使用“向前”奖励。
    cmd = env.command_generator.command
    cmd_forward = cmd[:, 0]
    has_forward_cmd = torch.abs(cmd_forward) > 0.1
    root_pos = asset.data.root_pos_w[:, :3]
    root_quat = asset.data.root_quat_w
    yaw_quat = math_utils.yaw_quat(root_quat)
    dev = root_pos.device
    forward_body = torch.tensor([1.0, 0.0, 0.0], device=dev, dtype=root_pos.dtype).unsqueeze(0).expand(env.num_envs, 3)
    forward_w = math_utils.quat_rotate(yaw_quat, forward_body)  # [batch, 3]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # [batch, 2, 3]
    rel_pos = feet_pos - root_pos.unsqueeze(1)  # [batch, 2, 3]
    forward_disp = torch.sum(rel_pos * forward_w.unsqueeze(1), dim=-1)  # [batch, 2]

    # 根据指令方向自适应：向前走奖励“向前”，向后走奖励“向后”。
    # sign(cmd_forward)=+1 → 前进，-1 → 后退；将位移乘以该符号后再截断为正值。
    sign_forward = torch.sign(cmd_forward).clamp(min=-1.0, max=1.0).unsqueeze(1)  # [batch,1]
    signed_disp = forward_disp * sign_forward
    signed_disp = signed_disp.clamp(min=0.0)

    swing_forward = torch.where(in_contact, torch.zeros_like(signed_disp), signed_disp)
    reward = torch.sum(swing_forward, dim=1)
    return reward * single_support.float() * has_forward_cmd.float()


def humanoid_swing_foot_lateral_penalty(
    env: BaseEnv | TienKungEnv,
    max_lateral_offset: float = 0.18,
    threshold: float = 10.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """单支撑时严重惩罚摆动脚横向偏离重心投影（不超过肩宽一半，防圆规步态）。

    - 仅在单支撑时生效；只约束当前摆动脚。
    - 重心投影 = root_pos 的 xy；在 yaw 系下横向 = y 分量（左右）。
    - 若 |swing_foot_y_in_yaw| > max_lateral_offset，则对超出部分做二次惩罚。
    - max_lateral_offset 建议约肩宽一半，如 0.18m。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = torch.norm(net_forces, dim=-1) > threshold  # [batch, 2], True=着地
    num_contacts = torch.sum(in_contact.int(), dim=1)
    single_support = num_contacts == 1
    has_command = (
        torch.norm(env.command_generator.command[:, :2], dim=1)
        + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1

    root_pos = asset.data.root_pos_w[:, :3]
    root_quat = asset.data.root_quat_w
    yaw_quat = math_utils.yaw_quat(root_quat)
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # [batch, 2, 3]
    rel_w = feet_pos - root_pos.unsqueeze(1)  # [batch, 2, 3] world
    rel_yaw_0 = math_utils.quat_rotate_inverse(yaw_quat, rel_w[:, 0, :])  # [batch, 3]
    rel_yaw_1 = math_utils.quat_rotate_inverse(yaw_quat, rel_w[:, 1, :])
    lateral = torch.stack([rel_yaw_0[:, 1], rel_yaw_1[:, 1]], dim=1)  # [batch, 2] 横向 = y in yaw
    excess = (torch.abs(lateral) - max_lateral_offset).clamp(min=0.0)  # [batch, 2]
    penalty_per_foot = torch.square(excess)
    swing_penalty = torch.where(in_contact, torch.zeros_like(penalty_per_foot), penalty_per_foot)
    penalty = torch.sum(swing_penalty, dim=1)
    return penalty * single_support.float() * has_command.float()


def humanoid_single_support_duration_penalty(
    env: BaseEnv | TienKungEnv,
    max_duration: float = 0.4,
    threshold: float = 10.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
) -> torch.Tensor:
    """Small penalty if single support lasts too long (encourage timely step transition).

    Tracks duration of current single-support phase; penalizes excess over max_duration.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = torch.norm(net_forces, dim=-1) > threshold
    num_contacts = torch.sum(in_contact.int(), dim=1)
    single_support = num_contacts == 1
    has_command = (
        torch.norm(env.command_generator.command[:, :2], dim=1)
        + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1

    if not hasattr(env, "_humanoid_single_support_time"):
        env._humanoid_single_support_time = torch.zeros(env.num_envs, device=net_forces.device)
    fresh = env.episode_length_buf <= 1
    if fresh.any():
        env._humanoid_single_support_time[fresh] = 0

    env._humanoid_single_support_time = torch.where(
        single_support & has_command,
        env._humanoid_single_support_time + env.step_dt,
        torch.zeros_like(env._humanoid_single_support_time),
    )
    excess = (env._humanoid_single_support_time - max_duration).clamp(min=0.0)
    return excess * single_support.float() * has_command.float()


# ══════════════════════════════════════════════════════════════════════════════
# Symmetry Reward Functions (for biped gait symmetry enforcement)
# ══════════════════════════════════════════════════════════════════════════════


def feet_contact_time_symmetry_exp(
    env: BaseEnv, sigma: float = 0.25, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor")
) -> torch.Tensor:
    """Reward symmetric stance durations between left and right feet (phase-aware).

    跟踪每只脚**最近一次完整支撑相**的持续时间（从着地到抬脚），
    然后比较左右脚的支撑时长是否一致。

    在交替步态中，两只脚在不同时刻接触地面，所以不能直接比较瞬时的
    contact_time（那样会鼓励双脚同步着地）。正确做法是比较"每次踩下去
    持续了多久"——对称步态中左右应该一样。

    检测逻辑：当 contact_time 从 >0 变为 0（抬脚瞬间），
    记录上一步的 contact_time 作为该脚的"最近支撑时长"。

    reward = exp(-(last_stance_left - last_stance_right)^2 / sigma^2)

    Only active when:
    - both feet have completed at least one stance phase
    - command is non-zero
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]  # [batch, 2]

    # ── 初始化状态跟踪缓冲区（首次调用时） ──
    if not hasattr(env, "_sym_last_stance_dur"):
        env._sym_last_stance_dur = torch.zeros(env.num_envs, 2, device=contact_time.device)
        env._sym_prev_contact_time = torch.zeros(env.num_envs, 2, device=contact_time.device)

    # ── 处理 episode 重置：新 episode 的前两步清空缓冲区 ──
    fresh = env.episode_length_buf <= 1
    if fresh.any():
        env._sym_last_stance_dur[fresh] = 0
        env._sym_prev_contact_time[fresh] = 0

    # ── 检测抬脚事件（liftoff）──
    # 上一步 contact_time > 0 且本步 contact_time == 0 → 脚刚抬起
    liftoff = (env._sym_prev_contact_time > 0) & (contact_time == 0)

    # 在抬脚瞬间，上一步的 contact_time 就是这次支撑相的总时长
    env._sym_last_stance_dur = torch.where(
        liftoff, env._sym_prev_contact_time, env._sym_last_stance_dur
    )

    # 更新 prev（为下一步检测做准备）
    env._sym_prev_contact_time = contact_time.clone()

    # ── 计算对称性奖励 ──
    diff = env._sym_last_stance_dur[:, 0] - env._sym_last_stance_dur[:, 1]
    reward = torch.exp(-torch.square(diff) / (sigma**2))

    # 只有两只脚都完成过至少一次完整支撑相时才有效
    valid = (env._sym_last_stance_dur[:, 0] > 0) & (env._sym_last_stance_dur[:, 1] > 0)
    reward = reward * valid.float()

    # 只在有速度指令时激活
    has_command = (
        torch.norm(env.command_generator.command[:, :2], dim=1)
        # + torch.abs(env.command_generator.command[:, 2])  #暂时不考虑转圈时的表现
    ) > 0.1
    return reward * has_command


def feet_distance_symmetry_per_cycle_exp(
    env: BaseEnv,
    sigma: float = 0.1,
    min_stride: float = 0.05,
) -> torch.Tensor:
    """奖励「一个步态周期内两只脚移动距离相近」。

    使用环境在步态周期回绕时记录的 _last_cycle_foot_distance（左右脚在上一个
    完整周期内的位移），计算 reward = exp(-(dist_L - dist_R)^2 / sigma^2)。
    仅在至少完成一个步态周期且两脚都有有效位移时生效；有速度指令时激活。

    适用于具有 _last_cycle_foot_distance、_feet_pos_at_cycle_start 等缓冲区的环境（如 RobanEnv）。
    """
    if not hasattr(env, "_last_cycle_foot_distance"):
        return torch.zeros(env.num_envs, device=env.device)

    dist = env._last_cycle_foot_distance
    diff = dist[:, 0] - dist[:, 1]
    reward = torch.exp(-torch.square(diff) / (sigma**2))
    valid = (dist[:, 0] >= min_stride) & (dist[:, 1] >= min_stride)
    reward = reward * valid.float()

    # 根据指令类型对奖励进行门控：
    # - 直行/后退：全额激活；
    # - 纯侧移：减半激活（左右步幅可以略不对称）；
    # - 原地转圈：关闭（由旋转类奖励负责）。
    cmd = env.command_generator.command
    lin_x = cmd[:, 0]
    lin_y = cmd[:, 1]
    ang_z = cmd[:, 2]

    forward_or_backward = torch.abs(lin_x) > 0.1
    side_only = (torch.abs(lin_y) > 0.1) & (torch.abs(lin_x) <= 0.1) & (torch.abs(ang_z) <= 0.1)
    turning_only = (torch.abs(ang_z) > 0.1) & (torch.norm(cmd[:, :2], dim=1) <= 0.1)

    factor = torch.zeros_like(lin_x)
    factor = torch.where(forward_or_backward, torch.ones_like(factor), factor)
    factor = torch.where(side_only, 0.5 * torch.ones_like(factor), factor)
    factor = torch.where(turning_only, torch.zeros_like(factor), factor)

    return reward * factor


def knee_joint_yaw_contact_only(
    env: BaseEnv | TienKungEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    joint_names: list[str] | None = None,
    angular_threshold: float = 0.1,
    linear_threshold: float = -0.01,
) -> torch.Tensor:
    """与 amp_roban_share 等价：只在接触期、直行/后退时惩罚膝关节 yaw 偏离默认位置。

    - 使用脚底 `contact_sensor` 判断是否接触；
    - 当 |yaw_cmd| 小或 x 方向速度越界（向后）时激活；
    - 对 `joint_names` 中关节的位置偏离做 L1 惩罚，仅在对应脚接触地面时生效。
    """
    # 1. 直行或后退模式
    cmd = env.command_generator.command
    lin_vel_x = cmd[:, 0]
    ang_vel_z = cmd[:, 2]
    is_valid_mode = (torch.abs(ang_vel_z) < angular_threshold) | (lin_vel_x < linear_threshold)

    # 2. 接触状态（脚部）
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )  # [N, num_feet]

    # 3. 关节偏移
    asset: Articulation = env.scene[asset_cfg.name]
    if joint_names is None:
        joint_names = ["leg_l3_joint", "leg_r3_joint"]

    joint_ids: list[int] = []
    for name in joint_names:
        ids, _ = asset.find_joints(name)
        if len(ids) > 0:
            joint_ids.append(ids[0])

    if len(joint_ids) == 0:
        return torch.zeros(env.num_envs, device=asset.data.joint_pos.device)

    current_pos = asset.data.joint_pos[:, joint_ids]
    default_pos = asset.data.default_joint_pos[:, joint_ids]
    deviation = torch.abs(current_pos - default_pos)

    # 将关节与脚一一对应（左脚-左膝，右脚-右膝）
    num_joints = deviation.shape[1]
    num_feet = contacts.shape[1]
    if num_joints == num_feet:
        contact_mask = contacts.float()
    else:
        # 若数量不一致，退化为“任一脚接触则所有相关关节惩罚”
        contact_mask = contacts.any(dim=1, keepdim=True).float().expand_as(deviation)

    penalty = deviation * contact_mask
    total_penalty = torch.sum(penalty, dim=1)
    return total_penalty * is_valid_mode.float()


def root_height_maintain(
    env: BaseEnv | TienKungEnv,
    min_height: float = 0.6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for maintaining root (base) height above a minimum threshold.

    当根节点高度 >= min_height 时奖励为 1（满分），
    低于 min_height 时按比例衰减：reward = (h / min_height)^2，
    高度越低惩罚越重，激励机器人保持直立行走姿态。

    返回 shape: (num_envs,)，值域 [0, 1]。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    root_height = asset.data.root_pos_w[:, 2]  # z 坐标

    reward = torch.where(
        root_height >= min_height,
        torch.ones_like(root_height),
        (root_height / min_height).clamp(min=0.0).square(),
    )
    return reward


def base_height_penalty(
    env: BaseEnv | TienKungEnv,
    min_height: float = 0.6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty when base height is below min_height (e.g. squatting or falling).

    penalty = (min_height - h)^2 when h < min_height, else 0.
    Use with negative weight so that below 0.6m incurs negative reward.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    h = asset.data.root_pos_w[:, 2]
    shortfall = (min_height - h).clamp(min=0.0)
    return torch.square(shortfall)


def flat_orientation_exp(
    env: BaseEnv | TienKungEnv,
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Positive Gaussian reward for flat base orientation (Roll=0, Pitch=0).

    reward = exp(-error / std^2), where error = sum(square(projected_gravity_b[:, :2])).
    High score when base is upright (gravity aligned with z in body frame).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return torch.exp(-error / (std**2))


# def joint_pos_symmetry_l2(
#     env: BaseEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     """L2 penalty for asymmetric joint positions between left and right limbs.

#     For each left-right joint pair, computes |q_left - sign * q_right|^2.
#     The sign accounts for joints whose zero position is mirrored (e.g., hip_roll).

#     This is a simpler alternative to FFT-based symmetry analysis.
#     Returns the mean squared deviation (use with negative weight to penalize).
#     """
#     asset: Articulation = env.scene[asset_cfg.name]
#     joint_pos = asset.data.joint_pos - asset.data.default_joint_pos

#     total_asym = torch.zeros(env.num_envs, device=joint_pos.device)

#     # Leg joints (6 pairs): sign pattern [-1, -1, -1, +1, +1, -1]
#     leg_signs = torch.tensor([-1.0, -1.0, -1.0, +1.0, +1.0, -1.0], device=joint_pos.device)
#     for i in range(6):
#         q_l = joint_pos[:, env.left_leg_ids[i]]
#         q_r = joint_pos[:, env.right_leg_ids[i]]
#         total_asym += torch.square(q_l - leg_signs[i] * q_r)

#     # Arm joints (4 pairs): sign pattern [+1, -1, -1, +1]
#     arm_signs = torch.tensor([+1.0, -1.0, -1.0, +1.0], device=joint_pos.device)
#     for i in range(4):
#         q_l = joint_pos[:, env.left_arm_ids[i]]
#         q_r = joint_pos[:, env.right_arm_ids[i]]
#         total_asym += torch.square(q_l - arm_signs[i] * q_r)

#     return total_asym / 10.0  # Normalize by number of pairs


# def joint_vel_symmetry_l2(
#     env: BaseEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     """L2 penalty for asymmetric joint velocities between left and right limbs.

#     Similar to joint_pos_symmetry_l2 but operates on joint velocities.
#     Symmetric gaits should have matching velocity patterns.
#     """
#     asset: Articulation = env.scene[asset_cfg.name]
#     joint_vel = asset.data.joint_vel

#     total_asym = torch.zeros(env.num_envs, device=joint_vel.device)

#     # Leg joints (6 pairs)
#     leg_signs = torch.tensor([-1.0, -1.0, -1.0, +1.0, +1.0, -1.0], device=joint_vel.device)
#     for i in range(6):
#         v_l = joint_vel[:, env.left_leg_ids[i]]
#         v_r = joint_vel[:, env.right_leg_ids[i]]
#         total_asym += torch.square(v_l - leg_signs[i] * v_r)

#     # Arm joints (4 pairs)
#     arm_signs = torch.tensor([+1.0, -1.0, -1.0, +1.0], device=joint_vel.device)
#     for i in range(4):
#         v_l = joint_vel[:, env.left_arm_ids[i]]
#         v_r = joint_vel[:, env.right_arm_ids[i]]
#         total_asym += torch.square(v_l - arm_signs[i] * v_r)

#     return total_asym / 10.0  # Normalize by number of pairs
