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
    gait_air_ratio_l: float = 0.38
    gait_air_ratio_r: float = 0.38
    gait_phase_offset_l: float = 0.38
    gait_phase_offset_r: float = 0.88
    gait_cycle: float = 0.85


@configclass
class LiteRewardCfg:
    """简化的奖励配置，与kuavo对应"""
    # 速度跟踪 (对应kuavo: tracking_lin_vel=1.2, tracking_ang_vel=1.1)
    # 大幅增加权重以使 AMP 占比降至 4%（Task 占比 96%）
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=5.0, params={"std": 0.5})  # 2.5 → 10.0
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=4.0, params={"std": 0.5})         # 2.0 → 8.0
    
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
        weight=-1.0,
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
    
    ## 以下是为了防止诡异行走而设置
    # 全身关节位置跟踪（指数形式）- 奖励关节接近默认位置
    # 权重建议：0.1 到 1.0，std 控制敏感度（越小越严格）
    joint_pos_tracking_exp_all = RewTerm(
        func=mdp.joint_pos_tracking_exp,
        weight=0.3,
        params={
            "std": 0.5,  # 标准差，控制奖励衰减速度
            "asset_cfg": SceneEntityCfg("robot"),  # 跟踪所有关节
        },
    )
    
    # 关节速度跟踪 - 鼓励平滑运动
    # 权重建议：-0.01 到 -0.1
    joint_vel_tracking_all = RewTerm(
        func=mdp.joint_vel_tracking_l2,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),  # 跟踪所有关节
        },
    )
    


@configclass
class RobanWalkFlatEnvCfg:
    amp_motion_files_display = ["/home/kyxmb/mkh/AMP/PB_AMP/legged_lab/envs/roban/datasets/motion_visualization/walk_pb_easy_nowrist.txt"]
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
        symmetry_cfg=None,  # RslRlSymmetryCfg()
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
    amp_motion_files = ["legged_lab/envs/roban/datasets/motion_amp_expert/walk_pb_easy.txt"]
    amp_num_preload_transitions = 200000
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 54  # AMP obs dim: 21 joint_pos + 21 joint_vel + 12 end_effector
    
    # 向后兼容参数（如果使用旧机制）
    amp_reward_coef = 0.3  # 仅用于向后兼容，新机制不使用
    amp_task_reward_lerp = 0.0  # 设置为0以禁用旧机制
    
    # 新奖励组合参数（与 amp_roban_share 一致）
    task_reward_weight = 1.0              # Task 奖励权重（对应 amp_roban_share 的 task_reward_weight）
    style_reward_weight = 0.02            # AMP 风格奖励权重（对应 amp_roban_share 的 style_reward_weight）
    discriminator_reward_scale = 2.0      # 判别器奖励缩放（对应 amp_roban_share 的 discriminator_reward_scale）
    # 最终奖励 = 1.0 * task_reward + 0.02 * (discriminator_reward_scale * style_reward)
    # AMP 贡献最大 = 0.02 * 2.0 = 0.04（当 style_reward = 1.0 时）
