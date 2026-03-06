# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# and is distributed under the BSD-3-Clause license.

"""Configuration for Roban S2 humanoid robot.

Joint layout: 21 DoF — 1 waist (waist_yaw) + 6×2 legs (leg_l1..6, leg_r1..6) + 4×2 arms (zarm_l1..4, zarm_r1..4).
与 amp_roban_share RobanS2 一致：使用 URDF 载入（UrdfFileCfg），保证 rigid bodies 与 contact sensors 正确。
"""

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.actuators.actuator_cfg import DelayedPDActuatorCfg_RobanS2
from legged_lab.assets import ISAAC_ASSET_DIR

ROBAN_S2_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAAC_ASSET_DIR}/roban_s2/urdf/urdf/biped_s221.urdf",
        fix_base=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.68),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "waist_yaw_joint": 0.0,
            "leg_l1_joint": -0.412,
            "leg_l2_joint": -0.0437,
            "leg_l3_joint": -0.287,
            "leg_l4_joint": 0.5,
            "leg_l5_joint": -0.2,
            "leg_l6_joint": 0.0,
            "leg_r1_joint": 0.412,
            "leg_r2_joint": 0.0437,
            "leg_r3_joint": 0.287,
            "leg_r4_joint": 0.5,
            "leg_r5_joint": -0.2,
            "leg_r6_joint": 0.0,
            "zarm_l1_joint": 0.2,
            "zarm_l2_joint": 0.16,
            "zarm_l3_joint": -0.4,
            "zarm_l4_joint": -0.5,
            "zarm_r1_joint": 0.2,
            "zarm_r2_joint": -0.16,
            "zarm_r3_joint": 0.4,
            "zarm_r4_joint": -0.5,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "motor": DelayedPDActuatorCfg_RobanS2(
            joint_names_expr=[
                "waist_yaw_joint",
                "leg_.*",
                "zarm_.*"
            ],
            effort_limit_sim={
                "waist_yaw_joint": 89.0,
                "leg_[l,r]1_joint": 89.0,
                "leg_[l,r]2_joint": 63.0,
                "leg_[l,r]3_joint": 89.0,
                "leg_[l,r]4_joint": 63.0,
                "leg_[l,r]5_joint": 24.0,
                "leg_[l,r]6_joint": 24.0,
                "zarm_[l,r]1_joint": 14.0,
                "zarm_[l,r]2_joint": 14.0,
                "zarm_[l,r]3_joint": 14.0,
                "zarm_[l,r]4_joint": 14.0,
            },
            velocity_limit={
                "waist_yaw_joint": 6.0,
                "leg_[l,r]1_joint": 6.0,
                "leg_[l,r]2_joint": 9.0,
                "leg_[l,r]3_joint": 6.0,
                "leg_[l,r]4_joint": 9.0,
                "leg_[l,r]5_joint": 9.0,
                "leg_[l,r]6_joint": 9.0,
                "zarm_[l,r]1_joint": 9.0,
                "zarm_[l,r]2_joint": 9.0,
                "zarm_[l,r]3_joint": 9.0,
                "zarm_[l,r]4_joint": 9.0,
            },
            stiffness={
                "waist_yaw_joint": 50.0,
                "leg_[l,r]1_joint": 50.0,
                "leg_[l,r]2_joint": 50.0,
                "leg_[l,r]3_joint": 50.0,
                "leg_[l,r]4_joint": 75.0,
                "leg_[l,r]5_joint": 15.0,
                "leg_[l,r]6_joint": 15.0,
                "zarm_[l,r]1_joint": 20.0,
                "zarm_[l,r]2_joint": 20.0,
                "zarm_[l,r]3_joint": 20.0,
                "zarm_[l,r]4_joint": 20.0,
            },
            damping={
                "waist_yaw_joint": 4.0,
                "leg_[l,r]1_joint": 4.0,
                "leg_[l,r]2_joint": 4.0,
                "leg_[l,r]3_joint": 4.0,
                "leg_[l,r]4_joint": 8.0,
                "leg_[l,r]5_joint": 2.0,
                "leg_[l,r]6_joint": 2.0,
                "zarm_[l,r]1_joint": 3.0,
                "zarm_[l,r]2_joint": 3.0,
                "zarm_[l,r]3_joint": 3.0,
                "zarm_[l,r]4_joint": 3.0,
            },
            armature={
                "waist_yaw_joint": 0.05,
                "leg_[l,r]1_joint": 0.05,
                "leg_[l,r]2_joint": 0.025,
                "leg_[l,r]3_joint": 0.025,
                "leg_[l,r]4_joint": 0.05,
                "leg_[l,r]5_joint": 0.05,
                "leg_[l,r]6_joint": 0.05,
                "zarm_[l,r]1_joint": 0.025,
                "zarm_[l,r]2_joint": 0.02,
                "zarm_[l,r]3_joint": 0.02,
                "zarm_[l,r]4_joint": 0.02,
            },
            friction=0,
            min_delay=0,
            max_delay=4,
            friction_static={
                "waist_yaw_joint": 1.0,
                "leg_[l,r]1_joint": 1.0,
                "leg_[l,r]2_joint": 0.5,
                "leg_[l,r]3_joint": 0.5,
                "leg_[l,r]4_joint": 1.0,
                "leg_[l,r]5_joint": 0.2,
                "leg_[l,r]6_joint": 0.2,
                "zarm_[l,r]1_joint": 0.5,
                "zarm_[l,r]2_joint": 0.3,
                "zarm_[l,r]3_joint": 0.2,
                "zarm_[l,r]4_joint": 0.3,
            },
            activation_vel=0.1,
            friction_dynamic=0,
        ),
    },
)
