# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# Modifications for PB_AMP: push with curriculum level scaling (amp_share 推力课程同款).
"""Event terms for curriculum-based push (external force level scaling)."""

from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

import isaaclab.envs.mdp.events as isaac_events


def push_by_setting_velocity_curriculum(
    env,
    env_ids,
    velocity_range: dict,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """与 amp_share 推力课程一致：按 env._external_force_torque_level 缩放推力速度。
    若 env 无 _external_force_torque_level 则退化为原始 push（缩放 1.0）。
    """
    asset = env.scene[asset_cfg.name]
    level = getattr(env, "_external_force_torque_level", None)
    if level is None:
        isaac_events.push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg)
        return
    # scale per env: (n_envs,) -> (n_envs, 1) for broadcasting to (n_envs, 6)
    scale = level[env_ids, 0, 0].unsqueeze(-1).to(asset.device)
    vel_w = asset.data.root_vel_w[env_ids].clone()
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    delta = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    vel_w += scale * delta
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)
