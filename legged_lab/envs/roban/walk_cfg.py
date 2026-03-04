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
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# and is distributed under the BSD-3-Clause license.

import math
import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)

import legged_lab.mdp as mdp
from legged_lab.assets.roban_s14 import ROBAN_S14_CFG
from legged_lab.envs.base.base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    EventCfg,
    HeightScannerCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    PhysxCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG  # noqa:F401

# 与 amp_roban_share 一致的多轨迹 AMP 路径与权重（RobanS2MixEnvCfg.motion_mode["slowly_move_flip"]）
# 需保证 AMP 工作区下存在 amp_roban_share 及对应 npz 轨迹文件
def _get_amp_share_motions_dir():
    """AMP 仓库根目录下的 amp_roban_share motions 目录。"""
    _dir = os.path.dirname(os.path.abspath(__file__))
    # roban -> envs -> legged_lab -> PB_AMP -> AMP
    amp_root = os.path.abspath(os.path.join(_dir, "..", "..", "..", "..", ".."))
    return os.path.join(amp_root, "amp_roban_share", "source", "ext_roban", "ext_roban", "utils", "motions")


def get_amp_motion_files_multi_trajectory():
    """与 amp_share 完全一致的多轨迹与权重（dict {path: weight}），用于多轨迹 AMP 跟踪。
    若 amp_roban_share 的 motions 目录不存在，则回退为单轨迹 .txt。
    """
    MOTIONS_DIR = _get_amp_share_motions_dir()
    scaled_dir = os.path.join(MOTIONS_DIR, "scaled_shoulder_only")
    if not os.path.isdir(scaled_dir):
        return ["legged_lab/envs/roban/datasets/motion_amp_expert/walk_pb_add_root.txt"]
    return {
        os.path.join(MOTIONS_DIR, "scaled_shoulder_only", "直行_中速_小摆手_003_Skeleton.npz"): 1.5,
        os.path.join(MOTIONS_DIR, "scaled_shoulder_only", "直行_中速_小摆手_005_Skeleton.npz"): 1.5,
        os.path.join(MOTIONS_DIR, "scaled_shoulder_only", "直行_中速_小摆手_006_Skeleton.npz"): 1.5,
        os.path.join(MOTIONS_DIR, "scaled_shoulder_only", "直行_低速_小摆手_Skeleton.npz"): 2.0,
        os.path.join(MOTIONS_DIR, "scaled_shoulder_only", "后退_中速_小摆手_000_Skeleton.npz"): 2.0,
        os.path.join(MOTIONS_DIR, "scaled_shoulder_only", "后退_低速_小摆手_000_Skeleton.npz"): 2.0,
        os.path.join(MOTIONS_DIR, "scaled_shoulder_only", "后退_低速_小摆手_Skeleton.npz"): 2.0,
        os.path.join(MOTIONS_DIR, "scaled_shoulder_only", "弧线_小右_Skeleton.npz"): 1.0,
        os.path.join(MOTIONS_DIR, "scaled_shoulder_only", "弧线_中左_Skeleton.npz"): 1.0,
        os.path.join(MOTIONS_DIR, "scaled_shoulder_only", "弧线_中右_Skeleton.npz"): 1.0,
        os.path.join(MOTIONS_DIR, "scaled_shoulder_only", "椭圆_Skeleton.npz"): 1.0,
        os.path.join(MOTIONS_DIR, "scaled_shoulder_only", "圆_Skeleton.npz"): 1.0,
        os.path.join(MOTIONS_DIR, "scaled_shoulder_only", "原地转圈_低速_逆时针_Skeleton.npz"): 1.5,
        os.path.join(MOTIONS_DIR, "scaled_shoulder_only", "原地转圈_低速_顺时针_Skeleton.npz"): 1.5,
        os.path.join(MOTIONS_DIR, "scaled_shoulder_only", "原地转圈_加减速_逆时针_Skeleton.npz"): 1.5,
        os.path.join(MOTIONS_DIR, "scaled_shoulder_only", "原地转圈_加减速_顺时针_Skeleton.npz"): 1.5,
        os.path.join(MOTIONS_DIR, "BVH_retargeted_motion", "静止站立_LiuKe_Skeleton.npz"): 1.5,
    }


@configclass
class GaitCfg:
    gait_air_ratio_l: float = 0.5
    gait_air_ratio_r: float = 0.5
    gait_phase_offset_l: float = 0.0
    gait_phase_offset_r: float = 0.5
    gait_cycle: float = 1.0


@configclass
class TerrainForceCurriculumCfg:
    """与 amp_share 一致：地形课程 + 推力课程，切换条件为外力水平 >= stage_two_force_threshold 时应用 stage_2 权重。"""
    enable: bool = True
    # 推力课程：每 env 的 level，[low_level, 1]，步长 0.05
    force_level_low: float = 0.2
    force_level_step: float = 0.05
    # 当 mean(外力水平) >= 此值时触发 stage two（一次性重写奖励权重）
    stage_two_force_threshold: float = 0.9
    # stage two 时各 reward term 的目标权重（仅覆盖存在的项，与 amp_share stage_2_reward_weight 对应）
    stage_2_reward_weights: dict = None

    def __post_init__(self):
        if self.stage_2_reward_weights is None:
            self.stage_2_reward_weights = {
                # 完全对齐 amp_share 中 RobanS2MixEnvCfg.CurriculumCfg.stage_two 的 stage_2_reward_weight
                "track_lin_vel_xy_exp": 4.5,
                "track_ang_vel_z_exp": 2.25,
                "ang_vel_xy_l2": -4.0,
                "ang_acc_xy_l2": -0.0001,
                "dof_acc_l2": -1e-5,
                "action_rate_l2": -0.01,
                "action_smoothness_l2": -0.04,
                "fft_dof_symmetry": 0.01,
                "action_rate_l2_ankle_roll": -0.5,
                "joint_deviation_ankle_roll": -0.5,
                "stand_still_without_cmd": -1.0,
            }


@configclass
class VelocityCurriculumCfg:
    """Pragmatic Natural Walking 速度课程学习配置。

    设计理念："先动量，后精度"（Momentum First, Precision Later）
    物理直觉：接近零速度时维持静态平衡比动态小跑更难。
    因此优先解锁前进速度（动量辅助平衡），再逐步解锁转向和侧移。

    3 阶段倒金字塔结构：
      Level 0 (Ignition):   只前进，利用动量建立稳定步态
      Level 1 (Steering):   解锁转向，保持步态纯度
      Level 2 (Omni):       解锁侧移、后退、全速——最终形态

    每级有独立的升级条件（promote_criteria），避免一刀切阈值卡死升级。

    统一死区（Unified Deadband）：
      指令落在死区内视为"站立"，此时步态奖励归零，静止惩罚生效，
      防止机器人被迫原地踏步。
    """
    enable: bool = True

    # ── 升/降级阈值（episode 存活率）──
    promote_threshold: float = 0.75  # 平均能走 75% 最大时长 → 可升级
    demote_threshold: float = 0.35   # 平均只能走 35% → 降级

    # ── 每级独立的升级跟踪误差条件 ──
    # promote_criteria[i] 定义从 Level i → Level i+1 的误差上限。
    # 缺失的 key 不检查（等价于 +inf）。
    # 例如 Level 0 只检查 lin_vel（还没解锁转向，不看 ang_vel）。
    promote_criteria: list = None

    # ── 统计缓冲区 ──
    buffer_size: int = 2000

    # ── 冷却机制 ──
    cooldown_steps: int = 500

    # ── 统一死区（Unified Deadband）──
    # 指令在死区内 → is_command_active = False → 步态奖励归零 + 静止惩罚生效
    lin_vel_deadband: float = 0.25   # m/s, |cmd_x| > 此值才算有前进/后退指令
    lat_vel_deadband: float = 0.25  # m/s, |cmd_y| > 此值才算有侧移指令
    ang_vel_deadband: float = 0.25   # rad/s, |cmd_yaw| > 此值才算有转向指令

    # ── 各级别的速度范围（倒金字塔）──
    levels: list = None

    # ── 各级别的奖励权重乘数（Level-Based Dynamic Reward Scaling）──
    reward_multipliers: list = None

    def __post_init__(self):
        if self.levels is None:
            self.levels = [
                # Level 0 (Ignition): 强制进入"黄金速度区间"，动量辅助平衡
                # lin_vel_y 锁为 0，ang_vel_z 仅允许微小噪声
                {"lin_vel_x": (0.4, 0.5), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-0.5, 0.5)},
                # Level 1 (Steering): 拓宽前进范围，解锁转向
                # lin_vel_y 仍锁定，保持步态纯度
                {"lin_vel_x": (0.3, 0.6), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-0.8, 0.8)},
                # Level 2 (Omni & Refinement): 全方向运动——最终形态
                {"lin_vel_x": (-0.6, 0.6), "lin_vel_y": (-0.6, 0.6), "ang_vel_z": (-1.0, 1.0)},
            ]

        if self.promote_criteria is None:
            self.promote_criteria = [
                # L0 → L1: 宽松，只验证"它在动"（不检查 ang_vel，还没解锁转向）
                {"lin_vel_err_limit": 0.3},
                # L1 → L2: 严格，必须能精确转向
                {"lin_vel_err_limit": 0.2, "ang_vel_err_limit": 0.2},
            ]

        if self.reward_multipliers is None:
            self.reward_multipliers = [
                # Level 0 (Ignition): 聚焦前进与身体水平
                {
                    "track_lin_vel_xy_exp": 1.5,
                    "flat_orientation_exp": 1.0,
                    "base_height_penalty": 1.0,
                    "humanoid_single_support_reward": 1.0,
                    "humanoid_swing_foot_height": 1.0,
                    "feet_contact_time_symmetry": 1.0,
                    "feet_distance_symmetry_per_cycle": 1.0,
                    "feet_distance": 1.0,
                },
                # Level 1 (Steering): 聚焦转向精度
                {
                    "track_lin_vel_xy_exp": 1.0,
                    "track_ang_vel_z_exp": 1.5,
                    "flat_orientation_exp": 1.0,
                    "base_height_penalty": 1.0,
                    "humanoid_single_support_reward": 1.0,
                    "humanoid_swing_foot_height": 1.0,
                    "feet_contact_time_symmetry": 1.0,
                    "feet_distance_symmetry_per_cycle": 1.0,
                },
                # Level 2 (Omni & Refinement): 对齐 amp_share stage_two 权重逻辑
                # 进入全指令范围后加强跟踪与平滑/对称，对应 stage_2_reward_weight
                {
                    "track_lin_vel_xy_exp": 2.25,   # amp_share stage_two: 4.5 (≈ base 2.0 * 2.25)
                    "track_ang_vel_z_exp": 1.5,     # amp_share: 2.25 (≈ base 1.5 * 1.5)
                    "action_rate_l2": 2.0,          # 加强动作平滑惩罚 (amp: -0.01)
                    "dof_acc_l2": 4.0,              # 加强关节加速度惩罚 (amp: -1e-5)
                    "feet_contact_time_symmetry": 1.2,
                    "feet_distance_symmetry_per_cycle": 1.2,
                },
            ]


@configclass
class LiteRewardCfg:
    """Pragmatic Minimalist Core: Positive Gaussian Objectives + Negative Constraints."""

    # ─── 1. Objectives (POSITIVE Gaussian) ───
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.0, params={"std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.5, params={"std": 0.5}
    )
    base_height_penalty = RewTerm(
        func=mdp.base_height_penalty,
        weight=-1.0,
        params={"min_height": 0.6},
    )
    flat_orientation_exp = RewTerm(
        func=mdp.flat_orientation_exp, weight=1.0, params={"std": 0.25}
    )

    # ─── 2. Gait Stylers (POSITIVE) ───
    # (feet_air_time, feet_contact_time_symmetry defined in RobanLiteRewardCfg with robot body names)

    # ─── 3. Physics Constraints (NEGATIVE) ───
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_roll.*"),
        },
    )
    feet_orientation = RewTerm(
        func=mdp.feet_orientation_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="ankle_roll.*")},
    )
    feet_distance = RewTerm(
        func=mdp.feet_distance_penalty,
        weight=-0.2,
        params={
            "min_dist": 0.20,
            "max_dist": 0.32,
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_roll.*"),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_pitch.*", "pelvis"]
            ),
            "threshold": 1.0,
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    torques = RewTerm(func=mdp.energy, weight=-0.0001)

    # ─── 4. Safety Rails (NEGATIVE) ───
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)

    # ─── REMOVED/DISABLED (redundant or covered by above) ───
    # stand_still_penalty, dof_vel_limits, lin_vel_z_l2, ang_vel_xy_l2, stumble, feet_force


@configclass
class RobanLiteRewardCfg(LiteRewardCfg):
    """Pragmatic Minimalist Core with Roban S14 body/joint names."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=[
                    "leg_l1_link", "leg_l2_link", "leg_l3_link", "leg_l4_link", "leg_l5_link",
                    "leg_r1_link", "leg_r2_link", "leg_r3_link", "leg_r4_link", "leg_r5_link",
                    "base_link",
                    "zarm_l1_link", "zarm_l2_link", "zarm_l3_link", "zarm_l4_link",
                    "zarm_r1_link", "zarm_r2_link", "zarm_r3_link", "zarm_r4_link",
                ],
            ),
            "threshold": 1.0,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["leg_l6_link", "leg_r6_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["leg_l6_link", "leg_r6_link"]),
        },
    )

    feet_orientation = RewTerm(
        func=mdp.feet_orientation_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["leg_l6_link", "leg_r6_link"])},
    )

    feet_distance = RewTerm(
        func=mdp.feet_distance_penalty,
        weight=-0.2,
        params={
            "min_dist": 0.22,
            "max_dist": 0.30,
            "asset_cfg": SceneEntityCfg("robot", body_names=["leg_l6_link", "leg_r6_link"]),
        },
    )

    # ─── 2. Gait Stylers: Contact Phase (no air-time shaping) ───
    # Induce natural leg lifting through stable single support.
    humanoid_flight_penalty = RewTerm(
        func=mdp.humanoid_flight_penalty,
        weight=-5.0,  # strong: flight forbidden
        params={"threshold": 10.0, "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["leg_l6_link", "leg_r6_link"])},
    )
    humanoid_double_support_penalty = RewTerm(
        func=mdp.humanoid_double_support_penalty,
        weight=-0.1,  # small: discourage persistent double support
        params={"threshold": 10.0, "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["leg_l6_link", "leg_r6_link"])},
    )
    humanoid_single_support_reward = RewTerm(
        func=mdp.humanoid_single_support_reward,
        weight=0.5,  # moderate: single support > double penalty
        params={"threshold": 10.0, "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["leg_l6_link", "leg_r6_link"])},
    )
    humanoid_swing_foot_height = RewTerm(
        func=mdp.humanoid_swing_foot_height_reward,
        weight=3.5,  # moderate, does not dominate velocity tracking
        params={
            "threshold": 10.0,
            "height_threshold": 0.05,
            "max_height": 0.12,
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["leg_l6_link", "leg_r6_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["leg_l6_link", "leg_r6_link"]),
        },
    )
    humanoid_swing_foot_forward = RewTerm(
        func=mdp.humanoid_swing_foot_forward_reward,
        weight=5.0,  # 单支撑时奖励摆动脚向前（左脚支撑奖右脚向前，反之亦然）
        params={
            "threshold": 10.0,
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["leg_l6_link", "leg_r6_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["leg_l6_link", "leg_r6_link"]),
        },
    )
    # 单支撑时严重惩罚摆动脚横向偏离重心（不超过肩宽一半），防圆规步态
    humanoid_swing_foot_lateral_penalty = RewTerm(
        func=mdp.humanoid_swing_foot_lateral_penalty,
        weight=-5.0,
        params={
            "max_lateral_offset": 0.15,
            "threshold": 10.0,
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["leg_l6_link", "leg_r6_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["leg_l6_link", "leg_r6_link"]),
        },
    )
    humanoid_single_support_duration_penalty = RewTerm(
        func=mdp.humanoid_single_support_duration_penalty,
        weight=-0.5,  # small: encourage timely step transition
        params={"max_duration": 0.4, "threshold": 10.0, "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["leg_l6_link", "leg_r6_link"])},
    )

    feet_contact_time_symmetry = RewTerm(
        func=mdp.feet_contact_time_symmetry_exp,
        weight=1.0,
        params={
            "sigma": 0.25,
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["leg_l6_link", "leg_r6_link"]),
        },
    )
    # 一个步态周期内两脚移动距离相近
    feet_distance_symmetry_per_cycle = RewTerm(
        func=mdp.feet_distance_symmetry_per_cycle_exp,
        weight=1.0,
        params={"sigma": 0.1, "min_stride": 0.05},
    )

    # ─── REMOVED/DISABLED ───
    # feet_air_time (replaced by contact-phase rewards: single support + swing height)
    # stand_still_penalty, dof_vel_limits, lin_vel_z_l2, ang_vel_xy_l2, stumble, feet_force
    # step_frequency, feet_height, straight_knee_landing, soft_landing, root_height_maintain


@configclass
class RobanAmpShareRewardCfg:
    """与 amp_roban_share 的 RobanS2MixEnvCfg.RewardsCfg 对齐的奖励配置。

    注意：将 amp_share 中的 `contact_forces` 统一映射为本工程中的 `contact_sensor`。
    关节与刚体命名保持一致（waist_yaw, leg_l/r*, zarm_l/r*）。
    """

    # 终止惩罚
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # ---- 任务：速度跟踪 ----
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=3.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.35)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.55,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )

    # ---- 步态与接触形状 ----
    feet_height_cycle = RewTerm(
        func=mdp.feet_height_cycle,
        weight=12.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="leg_[l,r]6_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names="leg_[l,r]6_link"),
            "max_height_clip": 0.15,
        },
    )

    # ---- 物理代价 / 约束 ----
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=0.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-2.0)
    ang_acc_xy_l2 = RewTerm(func=mdp.ang_acc_xy_l2, weight=-0.00005)
    dof_power_l2 = RewTerm(func=mdp.joint_power_l2, weight=-2.0e-5)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-4.0e-6,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["leg_[l,r][1-5]_joint", "zarm_.*_joint"])
        },
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_mean_acc_l2,
        weight=-6e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01)

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=["leg_[l,r][1-5]_link", "base_link", "zarm_.*_link"]
            ),
            "threshold": 1.0,
        },
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    flat_orientation_l2_exp = RewTerm(
        func=mdp.orientation_l2,
        weight=0.5,
        params={
            "desired_gravity": (0.05, 0.0, -0.99875),
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "std": 0.0001,
        },
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-8.0)
    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-10.0,
        params={"soft_ratio": 0.9},
    )

    # 腰部偏移惩罚
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["waist_yaw_joint"],
            )
        },
    )

    # ---- 接触/力相关约束 ----
    contact_momentum = RewTerm(
        func=mdp.contact_momentum,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="leg_[l,r]6_link"),
        },
    )
    contact_force = RewTerm(
        func=mdp.contact_forces,
        weight=-0.001,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="leg_[l,r]6_link"),
            "threshold": 350.0,
            "violation_max": 300.0,
            "violation_min": 0.0,
        },
    )

    # ---- 静止与站立质量 ----
    gravity_aligned_when_stopping = RewTerm(
        func=mdp.gravity_aligned_when_stopping,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "std": 0.05,
        },
    )
    stand_still_without_cmd = RewTerm(
        func=mdp.stand_still_without_cmd,
        weight=-0.3,
        params={"command_name": "base_velocity"},
    )
    feet_parallel_when_standing = RewTerm(
        func=mdp.feet_parallel_when_standing,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names="leg_[l,r]6_link"),
            "std": 0.02,
        },
    )

    # ---- 步态风格 / 频率 / 对称性 ----
    turning_step_frequency = RewTerm(
        func=mdp.turning_step_frequency_reward,
        weight=10.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="leg_[l,r]6_link"),
            "velocity_threshold": 0.12,
        },
    )
    contact_ground_straight_knee = RewTerm(
        func=mdp.contact_ground_straight_knee,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="leg_[l,r]6_link"),
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["leg_[l,r]4_joint"],
            ),
            "std": 0.3,
        },
    )
    joint_deviation_ankle_roll = RewTerm(
        func=mdp.joint_deviation_l1_straight_only,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["leg_[l,r]6_joint"]),
        },
    )
    action_rate_l2_ankle_roll = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_[l,r]6_joint"])},
    )
    fft_dof_symmetry = RewTerm(
        func=mdp.fft_dof_symmetry,
        weight=0.001,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names_pairs": [
                "leg_[l,r]6_joint",
                "leg_[l,r]5_joint",
                "leg_[l,r]4_joint",
                "leg_[l,r]3_joint",
                "zarm_[l,r]1_joint",
                "zarm_[l,r]4_joint",
            ],
            "angular_threshold": 0.8,
            "command_name": "base_velocity",
        },
    )
    feet_y_distance = RewTerm(func=mdp.feet_y_distance, weight=-1.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_clip,
        weight=10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="leg_[l,r]6_link"),
            "command_name": "base_velocity",
            "threshold_min": 0.25,
            "threshold_max": 0.45,
        },
    )
    joint_deviation_hip_roll = RewTerm(
        func=mdp.joint_deviation_l1_straight_only,
        weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_[l,r]2_joint"])},
    )
    action_rate_l2_hip_roll = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_[l,r]2_joint"])},
    )
    joint_deviation_knee_yaw = RewTerm(
        func=mdp.joint_deviation_l1_straight_only,
        weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_[l,r]3_joint"])},
    )
    action_rate_l2_hip_yaw = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_[l,r]3_joint"])},
    )
    feet_slide_vel = RewTerm(
        func=mdp.feet_slide,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="leg_[l,r]6_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names="leg_[l,r]6_link"),
        },
    )
    feet_slide_yaw = RewTerm(
        func=mdp.feet_slide_yaw,
        weight=-0.8,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="leg_[l,r]6_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names="leg_[l,r]6_link"),
            "command_name": "base_velocity",
        },
    )
    feet_pos_acc_l2 = RewTerm(
        func=mdp.feet_pos_acc_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["leg_[l,r]6_link"])},
    )
    knee_pos_acc_l2 = RewTerm(
        func=mdp.knee_pos_acc_l2,
        weight=-0.0003,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["leg_[l,r]4_link"])},
    )
    knee_pos_z_limit = RewTerm(
        func=mdp.knee_pos_z_limit,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["leg_[l,r]4_link"]), "knee_z_limit": 0.58},
    )
    knee_ankle_y_position_difference = RewTerm(
        func=mdp.knee_ankle_y_position_difference,
        weight=-1.0,
        params={
            "knee_cfg": SceneEntityCfg("robot", body_names=["leg_[l,r]4_link"]),
            "ankle_cfg": SceneEntityCfg("robot", body_names=["leg_[l,r]6_link"]),
        },
    )
    foot_lift_during_rotation = RewTerm(
        func=mdp.foot_lift_height_during_rotation,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="leg_[l,r]6_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names="leg_[l,r]6_link"),
            "target_height": 0.08,
            "height_tolerance": 0.02,
            "rotation_threshold": 0.1,
            "linear_threshold": 0.1,
            "max_air_time": 0.2,
        },
    )
    feet_contact_time_symmetry_exp = RewTerm(
        func=mdp.feet_contact_time_symmetry_exp,
        weight=1.6,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="leg_[l,r]6_link"),
            "command_name": "base_velocity",
            "degree": 5,
            "std": 0.1,
        },
    )
    knee_joint_yaw_contact_only = RewTerm(
        func=mdp.knee_joint_yaw_contact_only,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="leg_[l,r]6_link"),
            "joint_names": ["leg_l3_joint", "leg_r3_joint"],
            "angular_threshold": 0.1,
            "linear_threshold": 0.1,
        },
    )
    stand_static_vel_without_cmd = RewTerm(
        func=mdp.stand_static_vel_without_cmd,
        weight=-0.1,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "cmd_threshold": 0.05,
            "ext_force_threshold": 10.0,
        },
    )


@configclass
class RobanWalkFlatEnvCfg:
    """与 amp_share (RobanS2MixEnvCfg) 一致：地形课程 + 推力课程，无速度课程；外力>=0.9 时 stage two 权重重写。"""
    amp_motion_files_display = ["legged_lab/envs/roban/datasets/motion_visualization/walk_pb_easy_nowrist.txt"]
    amp_num_joints = 21  # Roban no-wrist: waist(1)+legs(12)+arms(8)=21
    device: str = "cuda:0"
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=ROBAN_S14_CFG,
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="base_link",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),
        ),
    )
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        action_scale=0.5, # 也许可以尝试使用公式计算并替代
        # 与 amp_roban_share 一致：仅躯干碰地硬终止，膝盖/手臂碰撞用 undesired_contacts 软惩罚
        terminate_contacts_body_names=["base_link"],
        feet_body_names=["leg_l6_link", "leg_r6_link"],
        # 与 amp_roban_share 一致：根部高度低于 0.35m 时终止（真正摔倒）
        terminate_min_height=0.35,
    )
    # 使用与 amp_share RobanS2MixEnvCfg.RewardsCfg 对齐的奖励配置
    reward = RobanAmpShareRewardCfg()
    gait = GaitCfg()
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0,
            ang_vel=1.0,
            projected_gravity=1.0,
            commands=1.0,
            joint_pos=1.0,
            joint_vel=1.0,
            actions=1.0,
            height_scan=1.0,
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5,
    )
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(0.5, 5.0),
        rel_standing_envs=0.5,
        rel_heading_envs=0.8,
        heading_command=True,
        heading_control_stiffness=0.3,
        debug_vis=True,
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.6, 0.6),
            lin_vel_y=(-0.6, 0.6),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )
    # 与 amp_share 对齐：本任务不启用速度课程（仅使用地形+推力课程）
    velocity_curriculum: VelocityCurriculumCfg | None = None
    terrain_force_curriculum: TerrainForceCurriculumCfg = TerrainForceCurriculumCfg()
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(
            lin_vel=0.2,
            ang_vel=0.2,
            projected_gravity=0.05,
            # 与 amp_share Policy 观测噪声量级对齐（joint_pos ≈ ±0.05）
            joint_pos=0.05,
            joint_vel=1.5,
            height_scan=0.1,
        ),
    )
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=EventCfg(
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    # 对齐 amp_share：更宽的摩擦 / 恢复系数范围
                    "static_friction_range": (0.3, 1.0),
                    "dynamic_friction_range": (0.2, 1.0),
                    "restitution_range": (0.0, 0.5),
                    "num_buckets": 64,
                },
            ),
            # 对齐 amp_share：腿/臂 link 质量缩放，而不是只给 base_link 加质量
            scale_link_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", body_names=["leg_.*_link", "zarm_.*_link"]
                    ),
                    "mass_distribution_params": (0.8, 1.2),
                    "operation": "scale",
                },
            ),
            randomize_rigid_body_com=EventTerm(
                func=mdp.randomize_base_body_com,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                    "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
                },
            ),
            scale_actuator_gains=EventTerm(
                func=mdp.randomize_actuator_gains,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
                    "stiffness_distribution_params": (0.8, 1.2),
                    "damping_distribution_params": (0.8, 1.2),
                    "operation": "scale",
                },
            ),
            scale_joint_parameters=EventTerm(
                func=mdp.randomize_joint_parameters,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
                    "friction_distribution_params": (1.0, 1.0),
                    "armature_distribution_params": (0.5, 1.5),
                    "operation": "scale",
                },
            ),
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    # 对齐 amp_share：位置范围略大，增加 pitch/roll 随机
                    "pose_range": {
                        "x": (-0.7, 0.7),
                        "y": (-0.7, 0.7),
                        "yaw": (-3.14, 3.14),
                        "pitch": (-0.1, 0.1),
                        "roll": (-0.1, 0.1),
                    },
                    "velocity_range": {
                        "x": (-0.3, 0.3),
                        "y": (-0.3, 0.3),
                        "z": (-0.3, 0.3),
                        "roll": (-0.3, 0.3),
                        "pitch": (-0.3, 0.3),
                        "yaw": (-0.3, 0.3),
                    },
                },
            ),
            reset_robot_joints=EventTerm(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (0.5, 1.5),
                    "velocity_range": (0.0, 0.0),
                },
            ),
            # Interval 外力事件：完全对齐 amp_share EventCfg.base_external_force_torque
            base_external_force_torque=EventTerm(
                func=mdp.apply_external_force_torque_stochastic,
                mode="interval",
                interval_range_s=(0.01, 0.3),
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                    "force_range": {
                        "x": (-100.0, 100.0),
                        "y": (-100.0, 100.0),
                        "z": (-100.0, 100.0),
                    },
                    "torque_range": {
                        "x": (-50.0, 50.0),
                        "y": (-50.0, 50.0),
                        "z": (-50.0, 50.0),
                    },
                    "probability": 0.01,
                },
            ),
        ),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}),
    )
    sim: SimCfg = SimCfg(dt=0.005, decimation=4, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))


@configclass
class RobanWalkAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50000
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            data_augmentation_func="legged_lab.mdp.symmetry:roban_symmetry_augmentation",
            mirror_loss_coeff=3.0,
        ),
        rnd_cfg=None,  # RslRlRndCfg()
    )
    clip_actions = None
    save_interval = 100
    runner_class_name = "AmpOnPolicyRunner"
    experiment_name = "walk"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "walk"
    wandb_project = "walk"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # amp parameter
    # 多轨迹 AMP 跟踪：与 amp_share 完全相同的动作轨迹与权重（dict {path: weight}）
    # 使用 amp_roban_share 的 npz 轨迹与权重；若未放置 npz，可改为单轨迹：
    # amp_motion_files = ["legged_lab/envs/roban/datasets/motion_amp_expert/walk_pb_add_root.txt"]
    amp_motion_files = get_amp_motion_files_multi_trajectory()
    amp_num_preload_transitions = 200000
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 64  # AMP obs dim (amp_share 同款): 21 joint_pos + 21 joint_vel + 1 root_height + 3 root_gravity + 3 root_lin_vel + 3 root_ang_vel + 12 end_effector
    
    # 向后兼容参数（如果使用旧机制）
    amp_reward_coef = 0.3  # 仅用于向后兼容，新机制不使用
    amp_task_reward_lerp = 0.0  # 设置为0以禁用旧机制
    
    # 新奖励组合参数（与 amp_roban_share 一致）
    task_reward_weight = 1.0              # Task 奖励权重（对应 amp_roban_share 的 task_reward_weight）
    style_reward_weight = 0.02            # AMP 风格奖励权重（对应 amp_roban_share 的 style_reward_weight）
    discriminator_reward_scale = 5.0      # 判别器奖励缩放（对应 amp_roban_share 的 discriminator_reward_scale）
    # 最终奖励 = 1.0 * task_reward + 0.02 * (discriminator_reward_scale * style_reward)
    # AMP 贡献最大 = 0.02 * 2.0 = 0.04（当 style_reward = 1.0 时）

    # ===== 对称性额外参数 (传递给 AMPPPO via runner) =====
    # Critic 对称性损失系数：V(obs) ≈ V(mirror(obs))
    critic_mirror_loss_coeff: float = 3
    # Discriminator 对称性损失模式：0=关闭, 1=仅 policy, 2=仅 expert, 3=两者都用
    disc_sym_loss_mode: int = 3
    # Discriminator 对称性损失系数
    disc_mirror_loss_coeff: float = 3
