# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# and is distributed under the BSD-3-Clause license.

"""Configuration for Roban S14 humanoid robot.

Joint layout: 21 DoF total — 1 waist (waist_yaw) + 6×2 legs (leg_l1..6, leg_r1..6) + 4×2 arms (zarm_l1..4, zarm_r1..4).

* :obj:`ROBAN_S14_CFG`: Roban S14 biped with 21 joints.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.actuators.actuator_cfg import DelayedPDActuatorCfg_RobanS2
from legged_lab.assets import ISAAC_ASSET_DIR

ROBAN_S14_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/roban_s14/usd/biped_s14.usd",
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
            enabled_self_collisions=True, # 关键！之前忘记开了！！！
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=(0.0, 0.0, 0.68),
        pos=(0.0, 0.0, 0.68),
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
                "waist_yaw_joint": 80.0,
                "leg_[l,r]1_joint": 80.0,
                "leg_[l,r]2_joint": 63.0,
                "leg_[l,r]3_joint": 80.0,
                "leg_[l,r]4_joint": 63.0,
                "leg_[l,r]5_joint": 25.0,
                "leg_[l,r]6_joint": 25.0,
                "zarm_[l,r]1_joint": 25.0,
                "zarm_[l,r]2_joint": 25.0,
                "zarm_[l,r]3_joint": 25.0,
                "zarm_[l,r]4_joint": 25.0,
            },
            velocity_limit={
                "waist_yaw_joint": 10.8,
                "leg_[l,r]1_joint": 10.8,
                "leg_[l,r]2_joint": 10.8,
                "leg_[l,r]3_joint": 10.8,
                "leg_[l,r]4_joint": 10.8,
                "leg_[l,r]5_joint": 10.2,
                "leg_[l,r]6_joint": 10.2,
                "zarm_[l,r]1_joint": 9.0,
                "zarm_[l,r]2_joint": 9.0,
                "zarm_[l,r]3_joint": 9.0,
                "zarm_[l,r]4_joint": 9.0,
            },
            stiffness={
                "waist_yaw_joint": 40.1792,
                "leg_[l,r]1_joint": 40.1792,
                "leg_[l,r]2_joint": 99.0984,
                "leg_[l,r]3_joint": 40.1792,
                "leg_[l,r]4_joint": 99.0984,
                "leg_[l,r]5_joint": 14.2506,
                "leg_[l,r]6_joint": 14.2506,
                "zarm_[l,r]1_joint": 14.2506,
                "zarm_[l,r]2_joint": 14.2506,
                "zarm_[l,r]3_joint": 14.2506,
                "zarm_[l,r]4_joint": 14.2506,
            },
            damping={
                "waist_yaw_joint": 2.5579,
                "leg_[l,r]1_joint": 2.5579,
                "leg_[l,r]2_joint": 6.3088,
                "leg_[l,r]3_joint": 2.5579,
                "leg_[l,r]4_joint": 6.3088,
                "leg_[l,r]5_joint": 0.9072,
                "leg_[l,r]6_joint": 0.9072,
                "zarm_[l,r]1_joint": 0.9072,
                "zarm_[l,r]2_joint": 0.9072,
                "zarm_[l,r]3_joint": 0.9072,
                "zarm_[l,r]4_joint": 0.9072,
            },
            armature={
                "waist_yaw_joint": 0.01017752,
                "leg_[l,r]1_joint": 0.01017752,
                "leg_[l,r]2_joint": 0.025101925,
                "leg_[l,r]3_joint": 0.01017752,
                "leg_[l,r]4_joint": 0.025101925,
                "leg_[l,r]5_joint": 0.003609725,
                "leg_[l,r]6_joint": 0.003609725,
                "zarm_[l,r]1_joint": 0.003609725,
                "zarm_[l,r]2_joint": 0.003609725,
                "zarm_[l,r]3_joint": 0.003609725,
                "zarm_[l,r]4_joint": 0.003609725,
            },
            friction=0,
            min_delay=0,
            max_delay=4,
            friction_static={
                "waist_yaw_joint": 0.2,
                "leg_[l,r]1_joint": 0.2,
                "leg_[l,r]2_joint": 0.2,
                "leg_[l,r]3_joint": 0.2,
                "leg_[l,r]4_joint": 0.2,
                "leg_[l,r]5_joint": 0.2,
                "leg_[l,r]6_joint": 0.2,
                "zarm_[l,r]1_joint": 0.2,
                "zarm_[l,r]2_joint": 0.2,
                "zarm_[l,r]3_joint": 0.2,
                "zarm_[l,r]4_joint": 0.2,
            },
            activation_vel=0.1,
            friction_dynamic=0,
        ),
    },
)
