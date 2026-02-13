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


def energy(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )


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


def body_force(
    env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    zero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) < 0.1
    return torch.sum(torch.abs(angle), dim=1) * zero_flag


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

    着地瞬间脚部的垂直速度应该尽量小。砸地式落脚会产生冲击，
    不利于稳定行走，也不自然。

    仅在 first_contact 时给出奖励：
        reward = exp(-(v_z)² / std²) * first_contact

    v_z 被 clamp 到 [-10, 0]（只关注下落方向）。
    v_z ≈ 0 时奖励最大（轻柔落地）。

    amp_roban_share 参考值：weight=1.0(?), std=0.05
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]  # [batch, 2]

    asset: Articulation = env.scene[asset_cfg.name]
    # 脚部垂直速度（世界坐标系 z 方向）
    body_vel_z = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]  # [batch, 2]
    # 只关注下落方向（负速度）
    body_vel_z = torch.clamp(body_vel_z, min=-10.0, max=0.0)

    reward = torch.exp(-torch.square(body_vel_z / std)) * first_contact

    result = torch.sum(reward, dim=1)
    # 只在有速度指令时激活
    has_command = (
        torch.norm(env.command_generator.command[:, :2], dim=1)
        + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return result * has_command


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
