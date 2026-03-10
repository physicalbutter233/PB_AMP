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

import argparse
import os
import sys

import mujoco
import mujoco_viewer
import numpy as np
import torch
from pynput import keyboard
import time

# 策略观测/动作的 21 维顺序，必须与 Isaac Lab 中 robot.data.joint_pos 顺序一致（roban_envs 里
# compute_current_observations 用的就是该顺序）；若不一致会导致 policy 输出错位。
ISAAC_JOINT_NAMES_21 = (
    "waist_yaw_joint",
    "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint", "leg_l6_joint",
    "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint", "leg_r6_joint",
    "zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint",
    "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint", "zarm_r4_joint",
)


class SimToSimCfg:
    """Configuration class for sim2sim parameters.

    Must be kept consistent with the training configuration (walk_cfg.py).
    Robot init state (default_dof_pos, base height 0.68) follows
    legged_lab/assets/roban_s14/roban_s14.py ROBAN_S14_CFG.
    """

    class sim:
        sim_duration = 100.0
        num_action = 21  # Roban S14: waist(1) + legs(12) + arms(8) = 21
        num_obs_per_step = 78  # 3(ang_vel) + 3(gravity) + 3(command) + 21(joint_pos) + 21(joint_vel) + 21(action) + 2(sin) + 2(cos) + 2(phase_ratio)
        actor_obs_history_length = 10
        dt = 0.005
        decimation = 4
        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25  # 与 walk_cfg 的 robot.action_scale 对齐

    class robot:
        gait_air_ratio_l: float = 0.38
        gait_air_ratio_r: float = 0.38
        gait_phase_offset_l: float = 0.38
        gait_phase_offset_r: float = 0.88
        gait_cycle: float = 0.85
        # PD 增益：与 roban_s14.py DelayedPDActuatorCfg_RobanS2 的 stiffness/damping 一致（Isaac Lab 关节顺序，21 维）
        stiffness: tuple = (
            40.1792, 40.1792, 99.0984, 40.1792, 99.0984, 14.2506, 14.2506,  # waist, leg_l1-6
            40.1792, 99.0984, 40.1792, 99.0984, 14.2506, 14.2506,          # leg_r1-6
            14.2506, 14.2506, 14.2506, 14.2506,                            # zarm_l1-4
            14.2506, 14.2506, 14.2506, 14.2506,                           # zarm_r1-4
        )
        damping: tuple = (
            2.5579, 2.5579, 6.3088, 2.5579, 6.3088, 0.9072, 0.9072,
            2.5579, 6.3088, 2.5579, 6.3088, 0.9072, 0.9072,
            0.9072, 0.9072, 0.9072, 0.9072,
            0.9072, 0.9072, 0.9072, 0.9072,
        )
        kp_scale_sim2sim: float = 1  # MuJoCo 下缩小 Kp 避免过冲蜷缩，Kd 按 sqrt(scale) 缩放


class MujocoRunner:
    """
    Sim2Sim runner that loads a policy and a MuJoCo model
    to run real-time humanoid control simulation for Roban S14.

    Args:
        cfg (SimToSimCfg): Configuration object for simulation.
        policy_path (str): Path to the TorchScript exported policy.
        model_path (str): Path to the MuJoCo XML model.
    """

    def __init__(self, cfg: SimToSimCfg, policy_path, model_path, zero_action: bool = False, right_flip: bool = False, left_flip: bool = False, flip_knees: bool = False, flip_arm_end: bool = False, hold_pose: bool = False, skip_viewer: bool = False):
        self.cfg = cfg
        self.zero_action = zero_action
        self.right_flip = right_flip
        self.left_flip = left_flip
        self.flip_knees = flip_knees
        self.flip_arm_end = flip_arm_end
        self.hold_pose = hold_pose  # 每步强制 qpos/qvel=default、ctrl=0，不积分，仅看 default 在 MuJoCo 里长什么样
        self.skip_viewer = skip_viewer  # 仅打印关节顺序等时可不创建 viewer
        network_path = policy_path
        
        # Check if the provided path is a checkpoint (model_*.pt) instead of exported policy
        if "model_" in os.path.basename(network_path) and network_path.endswith(".pt"):
            # Try to find the corresponding exported policy
            checkpoint_dir = os.path.dirname(network_path)
            exported_policy = os.path.join(checkpoint_dir, "exported", "policy.pt")
            if os.path.isfile(exported_policy):
                print(f"[INFO] 检测到 checkpoint 文件，自动使用导出的策略: {exported_policy}")
                network_path = exported_policy
            else:
                print(f"[WARNING] 提供的文件看起来是 checkpoint ({network_path})")
                print(f"[WARNING] 请使用导出的策略文件: {exported_policy}")
                print(f"[WARNING] 如果文件不存在，请先运行 play.py 导出策略")
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.cfg.sim.dt

        if zero_action or hold_pose or skip_viewer:
            self.policy = None
        else:
            try:
                self.policy = torch.jit.load(network_path)
            except RuntimeError as e:
                if "constants.pkl" in str(e) or "file not found" in str(e):
                    print(f"[ERROR] 无法加载策略文件: {network_path}")
                    print(f"[ERROR] 这看起来不是一个有效的 TorchScript 导出文件")
                    checkpoint_dir = os.path.dirname(network_path)
                    exported_policy = os.path.join(checkpoint_dir, "exported", "policy.pt")
                    if os.path.isfile(exported_policy):
                        print(f"[INFO] 找到导出的策略文件: {exported_policy}")
                        print(f"[INFO] 请使用: --policy {exported_policy}")
                    else:
                        print(f"[INFO] 请先运行 play.py 导出策略文件")
                    raise
                else:
                    raise
        self.data = mujoco.MjData(self.model)
        if not skip_viewer:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.viewer._render_every_frame = False
        else:
            self.viewer = None
        self.init_variables()

    def init_variables(self) -> None:
        """Initialize simulation variables and joint index mappings."""
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.dof_pos = np.zeros(self.cfg.sim.num_action)
        self.dof_vel = np.zeros(self.cfg.sim.num_action)
        self.action = np.zeros(self.cfg.sim.num_action)

        self.episode_length_buf = 0
        self.gait_phase = np.zeros(2)
        self.gait_cycle = self.cfg.robot.gait_cycle
        self.phase_ratio = np.array([self.cfg.robot.gait_air_ratio_l, self.cfg.robot.gait_air_ratio_r])
        self.phase_offset = np.array([self.cfg.robot.gait_phase_offset_l, self.cfg.robot.gait_phase_offset_r])

        # MuJoCo joint order (from XML): waist_yaw, leg_l1-6, leg_r1-6, zarm_l1-4, left_wrist, zarm_r1-4, right_wrist
        # Isaac Lab joint order: waist_yaw, leg_l1-6, leg_r1-6, zarm_l1-4, zarm_r1-4 (21 joints, no wrists)
        # Mapping: mujoco_to_isaac_idx[i] = mujoco joint index for isaac lab joint i
        self.mujoco_to_isaac_idx = [  # value是mujoco关节在isaac lab的id，注释是isaac lab的顺序
            0,   # waist_yaw_joint (MuJoCo index 0)
            1,   # leg_l1_joint (MuJoCo index 1)
            2,   # leg_l2_joint (MuJoCo index 2)
            3,   # leg_l3_joint (MuJoCo index 3)
            4,   # leg_l4_joint (MuJoCo index 4)
            5,   # leg_l5_joint (MuJoCo index 5)
            6,   # leg_l6_joint (MuJoCo index 6)
            7,   # leg_r1_joint (MuJoCo index 7)
            8,   # leg_r2_joint (MuJoCo index 8)
            9,   # leg_r3_joint (MuJoCo index 9)
            10,  # leg_r4_joint (MuJoCo index 10)
            11,  # leg_r5_joint (MuJoCo index 11)
            12,  # leg_r6_joint (MuJoCo index 12)
            13,  # zarm_l1_joint (MuJoCo index 13)
            14,  # zarm_l2_joint (MuJoCo index 14)
            15,  # zarm_l3_joint (MuJoCo index 15)
            16,  # zarm_l4_joint (MuJoCo index 16)
            # Skip left_wrist_joint (MuJoCo index 17)
            18,  # zarm_r1_joint (MuJoCo index 18, but skip wrist so -1)
            19,  # zarm_r2_joint (MuJoCo index 19, but skip wrist so -1)
            20,  # zarm_r3_joint (MuJoCo index 20, but skip wrist so -1)
            21,  # zarm_r4_joint (MuJoCo index 21, but skip wrist so -1)
            # Skip right_wrist_joint (MuJoCo index 22)
        ]

        # Isaac Lab to MuJoCo mapping: isaac_to_mujoco_idx[i] = isaac lab joint index for mujoco joint i
        self.isaac_to_mujoco_idx = [  # value是isaac lab关节在mujoco的id，注释是mujoco的顺序
            0,   # waist_yaw_joint
            1,   # leg_l1_joint
            2,   # leg_l2_joint
            3,   # leg_l3_joint
            4,   # leg_l4_joint
            5,   # leg_l5_joint
            6,   # leg_l6_joint
            7,   # leg_r1_joint
            8,   # leg_r2_joint
            9,   # leg_r3_joint
            10,  # leg_r4_joint
            11,  # leg_r5_joint
            12,  # leg_r6_joint
            13,  # zarm_l1_joint
            14,  # zarm_l2_joint
            15,  # zarm_l3_joint
            16,  # zarm_l4_joint
            # Skip left_wrist_joint (MuJoCo index 17)
            18,  # zarm_r1_joint
            19,  # zarm_r2_joint
            20,  # zarm_r3_joint
            21,  # zarm_r4_joint
            # Skip right_wrist_joint (MuJoCo index 22)
        ]

        # 初始关节位置：与 legged_lab/assets/roban_s14/roban_s14.py 中
        # ROBAN_S14_CFG.init_state.joint_pos 严格一致（Isaac Lab 关节顺序）。
        # MuJoCo 控制时按 isaac_to_mujoco_idx 映射，手腕关节在 XML 中存在则跳过。
        self.default_dof_pos = np.array([
            0.0,    # waist_yaw_joint
            -0.412, # leg_l1_joint
            -0.0437,# leg_l2_joint
            -0.287, # leg_l3_joint
            0.5,    # leg_l4_joint
            -0.2,   # leg_l5_joint
            0.0,    # leg_l6_joint
            0.412,  # leg_r1_joint
            0.0437, # leg_r2_joint
            0.287,  # leg_r3_joint
            0.5,    # leg_r4_joint
            -0.2,   # leg_r5_joint
            0.0,    # leg_r6_joint
            0.2,    # zarm_l1_joint
            0.16,   # zarm_l2_joint
            -0.4,   # zarm_l3_joint
            -0.5,   # zarm_l4_joint
            0.2,    # zarm_r1_joint
            -0.16,  # zarm_r2_joint
            0.4,    # zarm_r3_joint
            -0.5,   # zarm_r4_joint
        ])

        # 验证映射
        assert len(self.mujoco_to_isaac_idx) == self.cfg.sim.num_action
        assert len(self.isaac_to_mujoco_idx) == self.cfg.sim.num_action

        # PD 增益：与 roban_s14.py 一致，但 MuJoCo 步长下易过冲，默认缩放到 0.5
        kp_scale = getattr(self.cfg.robot, "kp_scale_sim2sim", 0.5)
        self.stiffness = np.array(self.cfg.robot.stiffness, dtype=np.float64) * kp_scale
        self.damping = np.array(self.cfg.robot.damping, dtype=np.float64) * np.sqrt(kp_scale)
        assert len(self.stiffness) == self.cfg.sim.num_action and len(self.damping) == self.cfg.sim.num_action

        # 关节传感器在 sensordata 中的索引（与 get_obs 一致，用于每步更新 dof_pos/dof_vel）
        self._joint_pos_sensor_slice = slice(16, 16 + 21)   # 21 个 joint_pos，顺序已是 waist,leg_l,leg_r,zarm_l,zarm_r
        self._joint_vel_sensor_slice = slice(37, 37 + 21)   # 21 个 joint_vel，但 XML 里顺序为 leg_l,leg_r,waist,zarm_l,zarm_r
        # 将 MuJoCo joint_vel 传感器顺序 → Isaac 顺序 (waist, leg_l1-6, leg_r1-6, zarm_l1-4, zarm_r1-4)
        self._joint_vel_sensor_to_isaac = np.array(
            [12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20], dtype=np.intp
        )
        # 左/右腿、左/右臂符号：默认不翻转；若 MuJoCo 轴与 Isaac 相反可加 --left-flip / --right-flip
        self._joint_sign_isaac_from_mujoco = np.ones(self.cfg.sim.num_action, dtype=np.float64)
        if self.left_flip:
            self._joint_sign_isaac_from_mujoco[1:7] = -1.0    # left leg
            self._joint_sign_isaac_from_mujoco[13:17] = -1.0  # left arm
        if self.right_flip:
            self._joint_sign_isaac_from_mujoco[7:13] = -1.0   # right leg
            self._joint_sign_isaac_from_mujoco[17:21] = -1.0  # right arm
        if self.flip_knees:
            self._joint_sign_isaac_from_mujoco[4] *= -1.0   # leg_l4 膝
            self._joint_sign_isaac_from_mujoco[10] *= -1.0  # leg_r4 膝
        if self.flip_arm_end:
            self._joint_sign_isaac_from_mujoco[15] *= -1.0   # zarm_l3
            self._joint_sign_isaac_from_mujoco[16] *= -1.0  # zarm_l4
            self._joint_sign_isaac_from_mujoco[19] *= -1.0   # zarm_r3
            self._joint_sign_isaac_from_mujoco[20] *= -1.0  # zarm_r4

        # 直接从 data.qpos / data.qvel 按关节顺序读取（Isaac 顺序，跳过手腕）
        # MuJoCo: id 0=freejoint, 1=waist, 2-7=leg_l, 8-13=leg_r, 14-17=zarm_l, 18=left_wrist, 19-22=zarm_r, 23=right_wrist
        # 21 关节取：1,2..13, 14,15,16,17, 19,20,21,22（跳过 18/23 手腕）
        mujoco_joint_ids_isaac = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22]
        self._qpos_adr = np.array([self.model.jnt_qposadr[jid] for jid in mujoco_joint_ids_isaac], dtype=np.intp)
        self._qvel_adr = np.array([self.model.jnt_dofadr[jid] for jid in mujoco_joint_ids_isaac], dtype=np.intp)

        # Initial command vel
        self.command_vel = np.array([0.0, 0.0, 0.0])
        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,), dtype=np.float32
        )

    def _read_dof_from_state(self) -> None:
        """从 data.qpos / data.qvel 按 Isaac 顺序读取 21 个关节的位置与速度（跳过手腕）。"""
        for i in range(self.cfg.sim.num_action):
            self.dof_pos[i] = self.data.qpos[self._qpos_adr[i]]
            self.dof_vel[i] = self.data.qvel[self._qvel_adr[i]]

    def _reset_to_default_pose(self) -> None:
        """将仿真状态设为 default 站立姿，避免从全 0 关节被 PD 拉崩。"""
        # 基座：位置 (0, 0, 0.68)；MuJoCo freejoint 四元数为 (w,x,y,z)，单位 (1,0,0,0)
        self.data.qpos[0:3] = [0.0, 0.0, 0.68]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[0:6] = 0.0
        # 21 个关节：Isaac default 转为 MuJoCo 约定后写入
        default_mujoco = self.default_dof_pos * self._joint_sign_isaac_from_mujoco
        for i in range(self.cfg.sim.num_action):
            self.data.qpos[self._qpos_adr[i]] = default_mujoco[i]
            self.data.qvel[self._qvel_adr[i]] = 0.0
        # 手腕关节保持 0（MuJoCo id 18=left_wrist, 23=right_wrist）
        wrist_jids = [18, 23]
        if all(jid < self.model.njnt for jid in wrist_jids):
            for jid in wrist_jids:
                adr = self.model.jnt_qposadr[jid]
                dofadr = self.model.jnt_dofadr[jid]
                self.data.qpos[adr] = 0.0
                self.data.qvel[dofadr] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _pin_base_to_default(self) -> None:
        """仅将基座位置/姿态/速度复位到默认，用于零策略模式下固定基座，避免整机倾倒。"""
        self.data.qpos[0:3] = [0.0, 0.0, 0.68]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[0:6] = 0.0

    def get_obs(self) -> np.ndarray:
        """
        Compute current observation vector from MuJoCo sensors and internal state.

        Returns:
            np.ndarray: Normalized and clipped observation history.
        """
        # 从 data.qpos/qvel 按关节顺序读取，与 Isaac 顺序一致，避免传感器顺序问题
        self._read_dof_from_state()

        # 获取角速度和方向（从传感器）
        ang_vel = self.data.sensor("BodyGyro").data.astype(np.double).copy()
        # MuJoCo framequat 已是 (x,y,z,w)，与 quat_rotate_inverse 一致，勿重排
        body_quat = self.data.sensor("BodyQuat").data.astype(np.double).copy()

        # 观测用 Isaac 约定：右腿/右臂从 MuJoCo 取反
        joint_pos_isaac = self.dof_pos * self._joint_sign_isaac_from_mujoco
        joint_vel_isaac = self.dof_vel * self._joint_sign_isaac_from_mujoco
        obs = np.concatenate(
            [
                ang_vel,  # 3
                self.quat_rotate_inverse(body_quat, np.array([0, 0, -1])),  # 3
                self.command_vel,  # 3
                (joint_pos_isaac - self.default_dof_pos),  # 21
                joint_vel_isaac,  # 21
                np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions),  # 21
                np.sin(2 * np.pi * self.gait_phase),  # 2
                np.cos(2 * np.pi * self.gait_phase),  # 2
                self.phase_ratio,  # 2
            ],
            axis=0,
        ).astype(np.float32)

        # Update observation history
        self.obs_history = np.roll(self.obs_history, shift=-self.cfg.sim.num_obs_per_step)
        self.obs_history[-self.cfg.sim.num_obs_per_step :] = obs.copy()

        return np.clip(self.obs_history, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations)

    def position_control(self) -> np.ndarray:
        """
        PD 位置控制：tau = Kp*(q_des - q) - Kd*qd，与 Isaac Lab 训练时一致（roban_s14.py 的 stiffness/damping）。
        目标位置由策略输出（Isaac 约定）转为 MuJoCo 约定：右腿/右臂取反。

        Returns:
            np.ndarray: 关节力矩，MuJoCo 执行器顺序（23 维，手腕为 0）。
        """
        actions_scaled = self.action * self.cfg.sim.action_scale
        q_des_isaac = actions_scaled + self.default_dof_pos  # 目标位置，Isaac 顺序 21
        q_des_mujoco = q_des_isaac * self._joint_sign_isaac_from_mujoco  # 转为 MuJoCo 轴约定
        q = self.dof_pos   # 当前位置，MuJoCo 传感器顺序
        qd = self.dof_vel  # 当前速度，已重排为 Isaac 顺序，数值仍是 MuJoCo，与 q 同约定
        tau = self.stiffness * (q_des_mujoco - q) - self.damping * qd  # 21 维，全 MuJoCo 约定

        # 策略输出为 Isaac 顺序 21 维；此处按相同顺序算 tau，再按 MuJoCo 执行器顺序写入（与 XML actuator 一致）
        mujoco_ctrl = np.zeros(self.model.nu)
        mujoco_ctrl[0] = tau[0]
        mujoco_ctrl[1:7] = tau[1:7]
        mujoco_ctrl[7:13] = tau[7:13]
        mujoco_ctrl[13:17] = tau[13:17]
        # mujoco_ctrl[17] = 0  # left_wrist
        mujoco_ctrl[18:22] = tau[17:21]
        # mujoco_ctrl[22] = 0  # right_wrist
        return mujoco_ctrl

    def print_joint_order(self) -> None:
        """打印当前使用的 21 关节顺序（与 Isaac 策略观测/动作一致），便于与 Isaac 的 robot.data.joint_pos 顺序对照。"""
        mujoco_joint_ids_isaac = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22]
        try:
            # MuJoCo 3.x: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, id)
            obj_type = getattr(mujoco.mjtObj, "mjOBJ_JOINT", 2)
        except AttributeError:
            obj_type = 2
        names_from_mujoco = []
        for jid in mujoco_joint_ids_isaac:
            n = mujoco.mj_id2name(self.model, obj_type, jid)
            names_from_mujoco.append(n if n else f"(jid={jid})")
        print("[sim2sim] 21 关节顺序（观测/策略动作/PD 目标均按此顺序，须与 Isaac robot.data.joint_pos 一致）：")
        for i, (expected, actual) in enumerate(zip(ISAAC_JOINT_NAMES_21, names_from_mujoco)):
            match = "  OK" if expected == actual else "  <-- 与预期不符"
            print(f"  [{i:2d}] {actual}{match}")
        print("[sim2sim] 若 Isaac 顺序不同，请在 Isaac 中打印 robot 的 joint 顺序并调整本脚本的映射。")

    def run(self) -> None:
        """
        Run the simulation loop with keyboard-controlled commands.
        """
        self._reset_to_default_pose()
        self.setup_keyboard_listener()
        self.listener.start()

        while self.data.time < self.cfg.sim.sim_duration:
            self.obs_history = self.get_obs()
            if not self.zero_action and not self.hold_pose:
                current_obs = self.obs_history[-self.cfg.sim.num_obs_per_step:]
                obs_tensor = torch.tensor(current_obs, dtype=torch.float32).unsqueeze(0)
                self.action[:] = self.policy(obs_tensor).detach().numpy()[0, : self.cfg.sim.num_action]
                self.action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)
            else:
                self.action[:] = 0.0  # 零动作或保持姿态时不用策略

            for sim_update in range(self.cfg.sim.decimation):
                step_start_time = time.time()
                if self.hold_pose:
                    # 不仿真：每步强制 default 姿态、零力矩，仅做 mj_forward 看姿态是否正确
                    self._reset_to_default_pose()
                    self.data.ctrl[:] = 0.0
                    mujoco.mj_forward(self.model, self.data)
                else:
                    self._read_dof_from_state()
                    if self.zero_action:
                        self._pin_base_to_default()  # 零策略时固定基座，只验证关节 PD，避免整机倾倒
                    mujoco_ctrl = self.position_control()
                    self.data.ctrl[:] = mujoco_ctrl
                    mujoco.mj_step(self.model, self.data)
                self.viewer.render()

                elapsed = time.time() - step_start_time
                sleep_time = self.cfg.sim.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            self.episode_length_buf += 1
            self.calculate_gait_para()

        self.listener.stop()
        self.viewer.close()

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Rotate a vector by the inverse of a quaternion.

        Args:
            q (np.ndarray): Quaternion (x, y, z, w) format.
            v (np.ndarray): Vector to rotate.

        Returns:
            np.ndarray: Rotated vector.
        """
        q_w = q[-1]
        q_vec = q[:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0

        return a - b + c

    def calculate_gait_para(self) -> None:
        """
        Update gait phase parameters based on simulation time and offset.
        """
        t = self.episode_length_buf * self.dt / self.gait_cycle
        self.gait_phase[0] = (t + self.phase_offset[0]) % 1.0
        self.gait_phase[1] = (t + self.phase_offset[1]) % 1.0

    def adjust_command_vel(self, idx: int, increment: float) -> None:
        """
        Adjust command velocity vector.

        Args:
            idx (int): Index of velocity component (0=x, 1=y, 2=yaw).
            increment (float): Value to increment.
        """
        self.command_vel[idx] += increment
        self.command_vel[idx] = np.clip(self.command_vel[idx], -1.0, 1.0)  # vel clip

    def setup_keyboard_listener(self) -> None:
        """
        Set up keyboard event listener for user control input.
        """

        def on_press(key):
            try:
                if key.char == "8":  # NumPad 8      x += 0.2
                    self.adjust_command_vel(0, 0.2)
                elif key.char == "2":  # NumPad 2      x -= 0.2
                    self.adjust_command_vel(0, -0.2)
                elif key.char == "4":  # NumPad 4      y -= 0.2
                    self.adjust_command_vel(1, -0.2)
                elif key.char == "6":  # NumPad 6      y += 0.2
                    self.adjust_command_vel(1, 0.2)
                elif key.char == "7":  # NumPad 7      yaw += 0.2
                    self.adjust_command_vel(2, -0.2)
                elif key.char == "9":  # NumPad 9      yaw -= 0.2
                    self.adjust_command_vel(2, 0.2)
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press)


if __name__ == "__main__":
    LEGGED_LAB_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    parser = argparse.ArgumentParser(description="Run sim2sim Mujoco controller for Roban S14.")
    parser.add_argument(
        "--task",
        type=str,
        default="walk",
        help="Task name, e.g., walk",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Path to policy.pt. If not specified, it will be set automatically based on --task",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(LEGGED_LAB_ROOT_DIR, "legged_lab/assets/roban_s14/biped_s14.xml"),
        help="Path to model.xml",
    )
    parser.add_argument("--duration", type=float, default=100.0, help="Simulation duration in seconds")
    parser.add_argument("--zero-action", action="store_true", help="不跑策略，动作恒为 0，仅用 PD 维持 default 姿态，用于调试")
    parser.add_argument("--left-flip", action="store_true", help="对左腿/左臂关节做符号翻转")
    parser.add_argument("--right-flip", action="store_true", help="对右腿/右臂关节做符号翻转")
    parser.add_argument("--flip-knees", action="store_true", help="仅对膝盖(leg_l4/leg_r4)符号取反，膝反了时用")
    parser.add_argument("--flip-arm-end", action="store_true", help="仅对小臂/手(zarm_l3/l4, zarm_r3/r4)符号取反")
    parser.add_argument("--kp-scale", type=float, default=None, help="PD 刚度缩放，默认 0.5；可试 0.25 若仍蜷缩")
    parser.add_argument("--hold-pose", action="store_true", help="每步强制 default 姿态、不积分，用于确认 default 在 MuJoCo 里是否站姿正确")
    parser.add_argument("--print-joint-order", action="store_true", help="仅打印 21 关节顺序（与 Isaac 策略观测/动作一致），用于核对是否与 Isaac robot.data.joint_pos 一致，然后退出")
    args = parser.parse_args()

    if args.policy is None:
        # Try to find exported policy in logs directory
        logs_dir = os.path.join(LEGGED_LAB_ROOT_DIR, "logs", args.task)
        exported_policy = None
        
        # Look for the most recent exported policy.pt
        if os.path.isdir(logs_dir):
            # Find all subdirectories (timestamped)
            subdirs = [d for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))]
            subdirs.sort(reverse=True)  # Most recent first
            
            for subdir in subdirs:
                exported_path = os.path.join(logs_dir, subdir, "exported", "policy.pt")
                if os.path.isfile(exported_path):
                    exported_policy = exported_path
                    break
        
        if exported_policy:
            args.policy = exported_policy
            print(f"[INFO] Auto-detected exported policy: {args.policy}")
        else:
            # Fallback to Exported_policy directory
            args.policy = os.path.join(LEGGED_LAB_ROOT_DIR, "Exported_policy", f"{args.task}.pt")

    if not args.print_joint_order and not args.zero_action and not args.hold_pose and not os.path.isfile(args.policy):
        print(f"[ERROR] Policy file not found: {args.policy}")
        print(f"[INFO] 请确保已通过 play.py 导出策略文件，或使用 --policy 指定正确的路径")
        print(f"[INFO] 导出的策略文件应该在: logs/{args.task}/*/exported/policy.pt")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"[ERROR] MuJoCo model file not found: {args.model}")
        sys.exit(1)

    print(f"[INFO] Loaded task preset: {args.task.upper()}")
    print(f"[INFO] Loaded policy: {args.policy}")
    print(f"[INFO] Loaded model: {args.model}")

    sim_cfg = SimToSimCfg()
    sim_cfg.sim.sim_duration = args.duration
    if args.kp_scale is not None:
        sim_cfg.robot.kp_scale_sim2sim = args.kp_scale
        print(f"[INFO] PD 刚度缩放: {args.kp_scale}")

    # Set gait parameters according to task
    if args.task == "walk":
        sim_cfg.robot.gait_air_ratio_l = 0.38
        sim_cfg.robot.gait_air_ratio_r = 0.38
        sim_cfg.robot.gait_phase_offset_l = 0.38
        sim_cfg.robot.gait_phase_offset_r = 0.88
        sim_cfg.robot.gait_cycle = 0.85
    elif args.task == "run":
        sim_cfg.robot.gait_air_ratio_l = 0.6
        sim_cfg.robot.gait_air_ratio_r = 0.6
        sim_cfg.robot.gait_phase_offset_l = 0.6
        sim_cfg.robot.gait_phase_offset_r = 0.1
        sim_cfg.robot.gait_cycle = 0.5

    runner = MujocoRunner(
        cfg=sim_cfg,
        policy_path=args.policy or "",
        model_path=args.model,
        zero_action=args.zero_action or args.print_joint_order,
        right_flip=args.right_flip,
        left_flip=args.left_flip,
        flip_knees=args.flip_knees,
        flip_arm_end=args.flip_arm_end,
        hold_pose=args.hold_pose,
        skip_viewer=args.print_joint_order,
    )
    if args.print_joint_order:
        runner.print_joint_order()
        sys.exit(0)
    if args.zero_action:
        print("[INFO] 零动作模式：仅 PD 维持 default 姿态，不调用策略；基座已固定，避免整机倾倒")
    if args.hold_pose:
        print("[INFO] 保持姿态模式：每步强制 default、不积分，用于检查 default 在 MuJoCo 中是否正确")
    if args.left_flip:
        print("[INFO] 已开启左腿/左臂符号翻转")
    if args.right_flip:
        print("[INFO] 已开启右腿/右臂符号翻转")
    if args.flip_knees:
        print("[INFO] 已开启膝盖(leg_l4/leg_r4)符号翻转")
    if args.flip_arm_end:
        print("[INFO] 已开启小臂/手(zarm_l3/l4, zarm_r3/r4)符号翻转")
    runner.run()
