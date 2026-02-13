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


@configclass
class GaitCfg:
    gait_air_ratio_l: float = 0.5
    gait_air_ratio_r: float = 0.5
    gait_phase_offset_l: float = 0.0
    gait_phase_offset_r: float = 0.5
    gait_cycle: float = 1.0


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
                },
                # Level 2 (Omni & Refinement): 原始权重
                {},
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
        weight=5.0,  # small: anti-moonwalk
        params={
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

    # ─── REMOVED/DISABLED ───
    # feet_air_time (replaced by contact-phase rewards: single support + swing height)
    # stand_still_penalty, dof_vel_limits, lin_vel_z_l2, ang_vel_xy_l2, stumble, feet_force
    # step_frequency, feet_height, straight_knee_landing, soft_landing, root_height_maintain


@configclass
class RobanWalkFlatEnvCfg:
    amp_motion_files_display = ["legged_lab/envs/roban/datasets/motion_visualization/walk_pb_easy_nowrist.txt"]
    amp_num_joints = 21  # Roban no-wrist: waist(1)+legs(12)+arms(8)=21
    device: str = "cuda:0"
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=ROBAN_S14_CFG,
        terrain_type="generator",
        terrain_generator=GRAVEL_TERRAINS_CFG,
        max_init_terrain_level=5,
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
    reward = RobanLiteRewardCfg()
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
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.6, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.57, 1.57), heading=(-math.pi, math.pi)
        ),
    )
    velocity_curriculum: VelocityCurriculumCfg = VelocityCurriculumCfg()
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(
            lin_vel=0.2,
            ang_vel=0.2,
            projected_gravity=0.05,
            joint_pos=0.01,
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
                    "static_friction_range": (0.6, 1.0),
                    "dynamic_friction_range": (0.4, 0.8),
                    "restitution_range": (0.0, 0.005),
                    "num_buckets": 64,
                },
            ),
            add_base_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                    "mass_distribution_params": (-5.0, 5.0),
                    "operation": "add",
                },
            ),
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                    "velocity_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (-0.5, 0.5),
                        "roll": (-0.5, 0.5),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-0.5, 0.5),
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
            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(10.0, 15.0),
                params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
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
        learning_rate=1.0e-3,
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
            mirror_loss_coeff=1.5,
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
    # 使用与 amp_roban_share 一致的奖励计算机制
    # 奖励组合方式：combined_reward = task_reward_weight * task_reward + style_reward_weight * style_reward
    amp_motion_files = ["legged_lab/envs/roban/datasets/motion_amp_expert/walk_pb_add_root.txt"]
    amp_num_preload_transitions = 200000
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 67  # AMP obs dim: 21 joint_pos + 21 joint_vel + 1 root_height + 6 root_rotation + 3 root_lin_vel + 3 root_ang_vel + 12 end_effector
    
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
    critic_mirror_loss_coeff: float = 1.5
    # Discriminator 对称性损失模式：0=关闭, 1=仅 policy, 2=仅 expert, 3=两者都用
    disc_sym_loss_mode: int = 3
    # Discriminator 对称性损失系数
    disc_mirror_loss_coeff: float = 1.5
