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
