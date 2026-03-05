# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# Modifications for PB_AMP: push with curriculum level scaling (amp_share 推力课程同款).
"""Event terms for curriculum-based push (external force level scaling)."""

from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg, EventTermCfg, ManagerTermBase
from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import Articulation, RigidObject

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


def randomize_base_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_range: dict[str, tuple[float, float]],
    recompute_inertia: bool = False,
):
    """与 amp_roban_share 相同：随机化指定刚体（通常为 base_link）质心位置。"""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # 解析 env_ids（在 PhysX 视图里通常要用 CPU tensor）
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # 解析 body_ids
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # 当前 COM
    coms = asset.root_physx_view.get_coms()

    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0],
        ranges[:, 1],
        (len(env_ids), 1, 3),
        device=asset.device,
    )

    coms_default = coms[env_ids[:, None], body_ids, :3].clone()
    coms[env_ids[:, None], body_ids, :3] = coms_default + rand_samples.cpu()

    # 写回 PhysX
    asset.root_physx_view.set_coms(coms, env_ids)

    # inertia 目前直接透传（与 amp_share 一致，未做额外调整）
    if recompute_inertia:
        inertias = asset.root_physx_view.get_inertias()
        asset.root_physx_view.set_inertias(inertias, env_ids)


class apply_external_force_torque_stochastic(ManagerTermBase):
    """与 amp_roban_share 相同：对机器人 base 施加随机外力 / 外力矩，带课程因子。

    - 内部维护 `_external_force_torque_level`（[num_envs,1,1]），支持课程调节强度；
    - `EventTerm` 配置与 amp_share 的 `base_external_force_torque` 完全兼容。
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.low_level = 0.2
        self._external_force_torque_level = torch.ones(
            env.scene.num_envs, 1, 1, device=env.device
        ) * self.low_level

    def move_up_down(
        self,
        env_ids: torch.Tensor | None,
        move_up: torch.Tensor,
        move_down: torch.Tensor,
    ):
        # 更新外力等级
        self._external_force_torque_level[env_ids[move_up]] += 0.05
        self._external_force_torque_level[env_ids[move_down]] -= 0.05
        self._external_force_torque_level = torch.clip(
            self._external_force_torque_level,
            min=self.low_level,
            max=1.0,
        )

    def reset_all_env(self):
        self._external_force_torque_level[:] = self.low_level

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        force_range: dict = {
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
        },
        torque_range: dict = {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (-0.0, 0.0)},
        probability: float = 0.0,
    ):
        # 将输入的范围整理为 tensor
        force_range_t = torch.tensor(
            [
                [force_range["x"][0], force_range["y"][0], force_range["z"][0]],
                [force_range["x"][1], force_range["y"][1], force_range["z"][1]],
            ],
            device=env.device,
        )
        torque_range_t = torch.tensor(
            [
                [torque_range["x"][0], torque_range["y"][0], torque_range["z"][0]],
                [torque_range["x"][1], torque_range["y"][1], torque_range["z"][1]],
            ],
            device=env.device,
        )

        asset: RigidObject | Articulation = env.scene[asset_cfg.name]
        # 清空已有外力
        asset._external_force_b *= 0
        asset._external_torque_b *= 0

        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=asset.device)

        # 以概率 * level 触发外力
        random_values = torch.rand(env_ids.shape, device=env_ids.device)
        mask = random_values < probability * self._external_force_torque_level[env_ids, 0, 0]
        masked_env_ids = env_ids[mask]
        if len(masked_env_ids) == 0:
            return

        num_bodies = (
            len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies
        )

        size = (len(masked_env_ids), num_bodies, 3)

        # 按 level 缩放 force / torque 采样区间
        low_force = force_range_t[0, :].unsqueeze(0) * self._external_force_torque_level[masked_env_ids]
        high_force = force_range_t[1, :].unsqueeze(0) * self._external_force_torque_level[masked_env_ids]
        rand_force = torch.rand(size, device=force_range_t.device)
        sample_force = low_force + (high_force - low_force) * rand_force

        low_torque = torque_range_t[0, :].unsqueeze(0) * self._external_force_torque_level[masked_env_ids]
        high_torque = torque_range_t[1, :].unsqueeze(0) * self._external_force_torque_level[masked_env_ids]
        rand_torque = torch.rand(size, device=torque_range_t.device)
        sample_torque = low_torque + (high_torque - low_torque) * rand_torque

        asset.set_external_force_and_torque(
            sample_force,
            sample_torque,
            env_ids=masked_env_ids,
            body_ids=asset_cfg.body_ids,
        )
