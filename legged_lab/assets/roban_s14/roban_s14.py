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
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

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
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            "waist_yaw_joint": 0.0,
            "leg_l1_joint": 0.0,
            "leg_l2_joint": 0.0,
            "leg_l3_joint": 0.0,
            "leg_l4_joint": 0.0,
            "leg_l5_joint": 0.0,
            "leg_l6_joint": 0.0,
            "leg_r1_joint": 0.0,
            "leg_r2_joint": 0.0,
            "leg_r3_joint": 0.0,
            "leg_r4_joint": 0.0,
            "leg_r5_joint": 0.0,
            "leg_r6_joint": 0.0,
            "zarm_l1_joint": 0.0,
            "zarm_l2_joint": 0.0,
            "zarm_l3_joint": 0.0,
            "zarm_l4_joint": 0.0,
            "zarm_r1_joint": 0.0,
            "zarm_r2_joint": 0.0,
            "zarm_r3_joint": 0.0,
            "zarm_r4_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim={".*": 50.0},
            velocity_limit_sim={".*": 8.0},
            stiffness={".*": 30.0},
            damping={".*": 2.0},
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["leg_.*_joint"],
            effort_limit_sim={".*": 300.0},
            velocity_limit_sim={".*": 24.0},
            stiffness={".*": 200.0},
            damping={".*": 5.0},
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["zarm_.*_joint"],
            effort_limit_sim={".*": 12.0},
            velocity_limit_sim={".*": 14.0},
            stiffness={".*": 20.0},
            damping={".*": 1.5},
        ),
    },
)
