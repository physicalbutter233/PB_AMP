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
    """速度指令课程学习配置。

    训练开始时只给小速度指令（先学慢走），随着策略稳定后逐步扩大速度范围。

    升级条件（必须同时满足）：
      1. 平均 episode 长度 > promote_threshold × max_episode_length
      2. 平均线速度跟踪误差 < promote_lin_vel_err_limit
      3. 平均角速度跟踪误差 < promote_ang_vel_err_limit
    这避免了"站桩生存偏差"——机器人站着不动也能升级的问题。

    降级条件：平均 episode 长度 < demote_threshold × max_episode_length

    冷却机制：级别变更后，缓冲区中残留着旧级别指令下的数据。
    通过 cooldown_steps 忽略掉在旧级别下开始的 episode，避免"火箭跳级"。
    """
    enable: bool = True

    # ── 升/降级阈值 ──
    promote_threshold: float = 0.75  # 平均能走 75% 最大时长 → 升级
    demote_threshold: float = 0.35   # 平均只能走 35% → 降级

    # ── 跟踪误差限制（防止站桩生存偏差）──
    promote_lin_vel_err_limit: float = 0.2   # m/s，线速度 RMSE 上限
    promote_ang_vel_err_limit: float = 0.2   # rad/s，角速度 RMSE 上限

    # ── 统计缓冲区 ──
    buffer_size: int = 2000  # 大规模并行环境需要更大的样本量

    # ── 冷却机制 ──
    # 级别变更后，忽略在旧级别下启动的 episode（单位：env steps）
    cooldown_steps: int = 500

    # ── 各级别的速度范围 ──
    levels: list = None

    def __post_init__(self):
        if self.levels is None:
            self.levels = [
                # Level 0: 只前进慢走，几乎不转弯
                {"lin_vel_x": (0.0, 0.3), "lin_vel_y": (-0.1, 0.1), "ang_vel_z": (-0.3, 0.3)},
                # Level 1: 加入后退和中速
                {"lin_vel_x": (-0.2, 0.5), "lin_vel_y": (-0.2, 0.2), "ang_vel_z": (-0.8, 0.8)},
                # Level 2: 接近完整范围
                {"lin_vel_x": (-0.4, 0.8), "lin_vel_y": (-0.3, 0.3), "ang_vel_z": (-1.2, 1.2)},
                # Level 3: 完整范围
                {"lin_vel_x": (-0.6, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.57, 1.57)},
            ]


@configclass
class LiteRewardCfg:
    """简化的奖励配置，与kuavo对应"""
    # 速度跟踪 (对应kuavo: tracking_lin_vel=1.2, tracking_ang_vel=1.1)
    # 大幅增加权重以使 AMP 占比降至 4%（Task 占比 96%）
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.5, params={"std": 0.5})  # 2.5 → 10.0
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5})         # 2.0 → 8.0
    
    # 速度不匹配惩罚 (对应kuavo: vel_mismatch_exp=0.5)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    
    # 姿态控制 (对应kuavo: orientation=1.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    
    # 能量消耗 (对应kuavo: torques=-1e-5, dof_vel=-5e-4)
    energy = RewTerm(func=mdp.energy, weight=-1e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1e-7)
    
    # 动作平滑度 (对应kuavo: action_smoothness=-0.002)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
    
    # 碰撞惩罚 (对应kuavo: collision=-1.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-3.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_pitch.*", "pelvis"]
            ),
            "threshold": 1.0,
        },
    )
    
    # 脚部滑动 (对应kuavo: foot_slip=-0.05)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_roll.*"),
        },
    )
    
    # 脚部接触力 (对应kuavo: feet_contact_forces=-0.01)
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-0.01,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )


@configclass
class RobanLiteRewardCfg(LiteRewardCfg):
    """Reward 与 LiteRewardCfg 相同，但 body/joint 名称改为 Roban S14。"""
    # 碰撞惩罚 - 使用Roban S14的body名称
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=["leg_l4_link", "leg_r4_link", "zarm_l2_link", "zarm_r2_link", "zarm_l4_link", "zarm_r4_link", "base_link"],
            ),
            "threshold": 1.0,
        },
    )
    
    # 脚部滑动 - 使用Roban S14的body名称
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["leg_l6_link", "leg_r6_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["leg_l6_link", "leg_r6_link"]),
        },
    )
    
    # 脚部接触力 - 使用Roban S14的body名称
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-0.01,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["leg_l6_link", "leg_r6_link"]),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    
    # ========== 滞空时间奖励（带裁剪） ==========
    # 使用 last_air_time + first_contact 触发，仅在着地瞬间给奖励
    # threshold_min=0.1: 低于 0.1s 的摆动不算步（防原地颤抖）
    # threshold_max=0.3: 超过 0.3s 不再有额外奖励（防大跨步）
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_clip_biped,
        weight=20.0,
        params={
            "threshold_min": 0.25,
            "threshold_max": 0.4,
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["leg_l6_link", "leg_r6_link"]),
        },
    )

    # ========== 步频惩罚 ==========
    # 惩罚双脚同时着地超过 0.2s，迫使机器人快速迈步
    # 返回负惩罚值，用正 weight
    step_frequency = RewTerm(
        func=mdp.step_frequency_penalty,
        weight=10.0,
        params={
            "max_grounded_time": 0.2,
            "penalty_scale": 2.0,
            "velocity_threshold": 0.12,
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["leg_l6_link", "leg_r6_link"]),
        },
    )

    # ========== 抬脚高度奖励 ==========
    # 每个摆动周期只在着地瞬间奖励一次，奖励该周期的最大抬脚高度
    # max_height_clip=0.15: 超过 15cm 不再有额外奖励（防止甩腿过高→大跨步）
    # amp_roban_share 参考值：weight=12.0, max_height_clip=0.15
    feet_height = RewTerm(
        func=mdp.feet_height_cycle,
        weight=12.0,
        params={
            "max_height_clip": 0.15,
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["leg_l6_link", "leg_r6_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["leg_l6_link", "leg_r6_link"]),
        },
    )

    # ========== 自然落地奖励 ==========
    # 着地瞬间膝盖伸直（人类走路落地腿近乎伸直）
    straight_knee_landing = RewTerm(
        func=mdp.contact_ground_straight_knee,
        weight=15.0,
        params={
            "std": 0.3,
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["leg_l6_link", "leg_r6_link"]),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["leg_l4_joint", "leg_r4_joint"]),
        },
    )

    # 着地瞬间脚部垂直速度小（轻柔落地，不砸地）
    soft_landing = RewTerm(
        func=mdp.contact_momentum,
        weight=1.0,
        params={
            "std": 0.05,
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["leg_l6_link", "leg_r6_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["leg_l6_link", "leg_r6_link"]),
        },
    )

    ## 以下是为了防止诡异行走而设置

    
    # ========== 髋关节 Roll (L2/R2) 防劈叉惩罚 ==========
    # 由于 L1/R1 的旋转轴 (0, ±0.707, -0.707) 没有 X 分量,
    # L2/R2 (roll, 轴=X) 在正常前后行走时不需要补偿 L1 的运动.
    # 因此可以较严格地约束 L2/R2, 防止双腿诡异岔开.
    #
    # # 速度条件版本: 前向行走时 deadzone=0.1rad(约5.7°),
    # # 侧向行走时自动放宽到 0.3rad(约17.2°)
    # hip_roll_penalty = RewTerm(
    #     func=mdp.hip_roll_conditional_penalty,
    #     weight=-5.0,
    #     params={
    #         "deadzone_base": 0.2,    # 前向行走时的死区 (rad)
    #         "deadzone_max": 0.4,     # 侧向行走时的死区 (rad)
    #         "vel_threshold": 0.3,    # 侧向速度阈值 (m/s)
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=["leg_l2_joint", "leg_r2_joint"],
    #         ),
    #     },
    # )
    
    # # ========== 髋关节 Yaw (L3/R3) 防内八惩罚 ==========
    # # L3/R3 是髋关节偏航, axis=(0,0,1), 默认值 L3=-0.287, R3=0.287
    # # 转弯时需要较大幅度运动, 其他情况保持默认位置附近
    # # 内八方向死区更紧: 正常方向 deadzone, 内八方向 deadzone * 0.4
    # #   L3 内八 = 偏差<0 (toein_sign=-1), R3 内八 = 偏差>0 (toein_sign=+1)
    # hip_yaw_penalty = RewTerm(
    #     func=mdp.hip_yaw_conditional_penalty,
    #     weight=-5.0,
    #     params={
    #         "deadzone_base": 0.2,            # 直行时的基础死区 (rad, ≈8.6°)
    #         "deadzone_max": 0.5,              # 转弯时的最大死区 (rad, ≈28.6°)
    #         "vel_threshold": 0.8,             # 角速度阈值 (rad/s), 达到此值死区最大
    #         "toein_deadzone_ratio": 0.4,      # 内八方向死区 = 基础死区 × 0.4 (更严格)
    #         "toein_signs": [-1.0, 1.0],       # L3 内八=-1, R3 内八=+1
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=["leg_l3_joint", "leg_r3_joint"],
    #         ),
    #     },
    # )
    
    # # ========== 膝盖 Pitch (L4/R4) 限位惩罚 ==========
    # # 膝盖关节 axis=(0,1,0), 限位=[0.0, 2.618], 默认=0.5rad
    # # 正常走路时膝盖在默认值附近 ±0.5rad (0.0~1.0rad) 范围内弯曲/伸展
    # # 超出此范围则惩罚，防止膝盖过度弯曲或完全伸直锁死
    # knee_pitch_penalty = RewTerm(
    #     func=mdp.deadzone_penalty,
    #     weight=-5.0,
    #     params={
    #         "deadzone": 0.3,  # 允许默认值±0.5rad(约28.6°)的偏移
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=["leg_l4_joint", "leg_r4_joint"],
    #         ),
    #     },
    # )
    
    # ========== 两脚距离惩罚 ==========
    # 惩罚两脚之间 3D 距离 < 20cm（太近/交叉）或 > 32cm（太远/劈叉）
    # 在 [20cm, 32cm] 范围内无惩罚
    feet_distance = RewTerm(
        func=mdp.feet_distance_penalty,
        weight=-30.0,
        params={
            "min_dist": 0.22,  # 最小允许距离 (m)
            "max_dist": 0.30,  # 最大允许距离 (m)
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["leg_l6_link", "leg_r6_link"],
            ),
        },
    )

    # ========== 对称性奖励 (Symmetry Rewards) ==========
    # 左右脚触地时间对称性：鼓励左右脚触地时间一致，防止瘸腿
    feet_contact_time_symmetry = RewTerm(
        func=mdp.feet_contact_time_symmetry_exp,
        weight=1.6,
        params={
            "sigma": 0.25,
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["leg_l6_link", "leg_r6_link"]),
        },
    )

    # # 关节位置对称性：惩罚左右关节位置不对称（代替 FFT 对称性分析的简化版本）
    # joint_pos_symmetry = RewTerm(
    #     func=mdp.joint_pos_symmetry_l2,
    #     weight=-0.5,
    #     params={},
    # )

    # # 关节速度对称性：惩罚左右关节速度不对称
    # joint_vel_symmetry = RewTerm(
    #     func=mdp.joint_vel_symmetry_l2,
    #     weight=-0.001,
    #     params={},
    # )


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
        action_scale=0.25, # 也许可以尝试使用公式计算并替代
        terminate_contacts_body_names=["leg_l4_link", "leg_r4_link", "zarm_l4_link", "zarm_r4_link", "base_link"],
        feet_body_names=["leg_l6_link", "leg_r6_link"],
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
