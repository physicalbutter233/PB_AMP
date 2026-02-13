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

class SimToSimCfg:
    """Configuration class for sim2sim parameters.

    Must be kept consistent with the training configuration.
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
        action_scale = 0.25  # Roban uses 0.25

    class robot:
        gait_air_ratio_l: float = 0.38
        gait_air_ratio_r: float = 0.38
        gait_phase_offset_l: float = 0.38
        gait_phase_offset_r: float = 0.88
        gait_cycle: float = 0.85


class MujocoRunner:
    """
    Sim2Sim runner that loads a policy and a MuJoCo model
    to run real-time humanoid control simulation for Roban S14.

    Args:
        cfg (SimToSimCfg): Configuration object for simulation.
        policy_path (str): Path to the TorchScript exported policy.
        model_path (str): Path to the MuJoCo XML model.
    """

    def __init__(self, cfg: SimToSimCfg, policy_path, model_path):
        self.cfg = cfg
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
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer._render_every_frame = False
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

        # 直接硬编码初始关节位置（与 roban_s14.py 对齐）
        # 这里的是isaac lab顺序，要转成mujoco顺序（即用isaac_to_mujoco_idx）
        # 注意：MuJoCo 模型可能包含手腕关节，需要跳过
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

        # Initial command vel
        self.command_vel = np.array([0.0, 0.0, 0.0])
        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,), dtype=np.float32
        )

    def get_obs(self) -> np.ndarray:
        """
        Compute current observation vector from MuJoCo sensors and internal state.

        Returns:
            np.ndarray: Normalized and clipped observation history.
        """
        # 从 MuJoCo 传感器数据中获取关节位置和速度
        # 传感器顺序：BodyAcc(3), BodyVel(3), BodyGyro(3), BodyPos(3), BodyQuat(4), 
        #           21个joint_pos, 21个joint_vel
        # 传感器索引：0-2: BodyAcc, 3-5: BodyVel, 6-8: BodyGyro, 9-11: BodyPos, 12-15: BodyQuat,
        #            16-36: joint_pos (21个), 37-57: joint_vel (21个)
        
        # 获取关节位置（21个，不含手腕）
        joint_pos_sensor_start = 16  # BodyAcc(3) + BodyVel(3) + BodyGyro(3) + BodyPos(3) + BodyQuat(4) = 16
        joint_pos_sensor_end = joint_pos_sensor_start + 21
        self.dof_pos = self.data.sensordata[joint_pos_sensor_start:joint_pos_sensor_end]
        
        # 获取关节速度（21个，不含手腕）
        joint_vel_sensor_start = 37  # 16 + 21 = 37
        joint_vel_sensor_end = joint_vel_sensor_start + 21
        self.dof_vel = self.data.sensordata[joint_vel_sensor_start:joint_vel_sensor_end]

        # 获取角速度和方向（从传感器）
        ang_vel = self.data.sensor("BodyGyro").data.astype(np.double)
        body_quat = self.data.sensor("BodyQuat").data[[1, 2, 3, 0]].astype(np.double)  # 转换为 x,y,z,w 格式

        obs = np.concatenate(
            [
                ang_vel,  # 3
                self.quat_rotate_inverse(body_quat, np.array([0, 0, -1])),  # 3
                self.command_vel,  # 3
                (self.dof_pos - self.default_dof_pos),  # 21
                self.dof_vel,  # 21
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
        Apply position control using scaled action.

        Returns:
            np.ndarray: Target joint positions in MuJoCo actuator order.
        """
        actions_scaled = self.action * self.cfg.sim.action_scale
        processed_actions = actions_scaled + self.default_dof_pos
        
        # MuJoCo 执行器顺序：waist_yaw, leg_l1-6, leg_r1-6, zarm_l1-4, left_wrist, zarm_r1-4, right_wrist (23个)
        # Isaac Lab 顺序：waist_yaw, leg_l1-6, leg_r1-6, zarm_l1-4, zarm_r1-4 (21个)
        # 需要将 21 个关节映射到 23 个执行器（跳过手腕）
        mujoco_ctrl = np.zeros(self.model.nu)  # 执行器数量
        
        # 映射 21 个关节到执行器
        # 执行器索引：0-12 (waist + legs), 13-16 (zarm_l1-4), 17 (left_wrist跳过), 18-21 (zarm_r1-4), 22 (right_wrist跳过)
        mujoco_ctrl[0] = processed_actions[0]   # waist_yaw
        mujoco_ctrl[1:7] = processed_actions[1:7]   # leg_l1-6
        mujoco_ctrl[7:13] = processed_actions[7:13]  # leg_r1-6
        mujoco_ctrl[13:17] = processed_actions[13:17]  # zarm_l1-4
        # mujoco_ctrl[17] = 0  # left_wrist (跳过)
        mujoco_ctrl[18:22] = processed_actions[17:21]  # zarm_r1-4
        # mujoco_ctrl[22] = 0  # right_wrist (跳过)
        
        return mujoco_ctrl

    def run(self) -> None:
        """
        Run the simulation loop with keyboard-controlled commands.
        """
        self.setup_keyboard_listener()
        self.listener.start()

        while self.data.time < self.cfg.sim.sim_duration:
            self.obs_history = self.get_obs()
            self.action[:] = self.policy(torch.tensor(self.obs_history, dtype=torch.float32)).detach().numpy()[:self.cfg.sim.num_action]
            self.action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)

            for sim_update in range(self.cfg.sim.decimation):
                step_start_time = time.time()

                mujoco_ctrl = self.position_control()
                # 设置 MuJoCo 控制目标（位置控制）
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

    if not os.path.isfile(args.policy):
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
        policy_path=args.policy,
        model_path=args.model,
    )
    runner.run()
