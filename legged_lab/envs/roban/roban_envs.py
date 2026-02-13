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

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor
from isaaclab.sim import PhysxCfg, SimulationContext
from collections import deque

from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_rotate
from scipy.spatial.transform import Rotation

from legged_lab.envs.roban.walk_cfg import RobanWalkFlatEnvCfg

from legged_lab.utils.env_utils.scene import SceneCfg
from rsl_rl.env import VecEnv
from rsl_rl.utils import AMPLoaderDisplay


class RobanEnv(VecEnv):
    def __init__(
        self,
        cfg: (
            RobanWalkFlatEnvCfg
        ),
        headless,
    ):
        self.cfg: (
            RobanWalkFlatEnvCfg
        )

        self.cfg = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)

        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]

        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)
        self.reward_manager = RewardManager(self.cfg.reward, self)

        self.init_buffers()

        # ── 速度课程学习初始化（必须在 reset 之前，reset 会调用 _update_velocity_curriculum）──
        self._init_velocity_curriculum()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(env_ids)

        amp_num_joints = getattr(self.cfg, "amp_num_joints", None)
        self.amp_loader_display = AMPLoaderDisplay(
            motion_files=self.cfg.amp_motion_files_display,
            device=self.device,
            time_between_frames=self.physics_dt,
            num_joints=amp_num_joints,
        )
        self.motion_len = self.amp_loader_display.trajectory_num_frames[0]

    def init_buffers(self):
        self.extras = {}

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        self.action_scale = self.cfg.robot.action_scale
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)

        # Roban S14: 脚踝/脚 body 为 leg_l6_link, leg_r6_link
        self.feet_body_ids, _ = self.robot.find_bodies(
            name_keys=["leg_l6_link", "leg_r6_link"], preserve_order=True
        )
        # 手臂末端（用于 hand 位置）zarm_l4_link, zarm_r4_link
        self.elbow_body_ids, _ = self.robot.find_bodies(
            name_keys=["zarm_l4_link", "zarm_r4_link"], preserve_order=True
        )
        # 左腿 6 关节: leg_l1..leg_l6_joint
        self.left_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "leg_l1_joint",
                "leg_l2_joint",
                "leg_l3_joint",
                "leg_l4_joint",
                "leg_l5_joint",
                "leg_l6_joint",
            ],
            preserve_order=True,
        )
        self.right_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "leg_r1_joint",
                "leg_r2_joint",
                "leg_r3_joint",
                "leg_r4_joint",
                "leg_r5_joint",
                "leg_r6_joint",
            ],
            preserve_order=True,
        )
        # 左臂 4 关节: zarm_l1..zarm_l4_joint
        self.left_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "zarm_l1_joint",
                "zarm_l2_joint",
                "zarm_l3_joint",
                "zarm_l4_joint",
            ],
            preserve_order=True,
        )
        self.right_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "zarm_r1_joint",
                "zarm_r2_joint",
                "zarm_r3_joint",
                "zarm_r4_joint",
            ],
            preserve_order=True,
        )
        self.ankle_joint_ids, _ = self.robot.find_joints(
            name_keys=["leg_l5_joint", "leg_r5_joint", "leg_l6_joint", "leg_r6_joint"],
            preserve_order=True,
        )
        self.waist_ids, _ = self.robot.find_joints(
            name_keys=["waist_yaw_joint"],
            preserve_order=True,
        )

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.left_arm_local_vec = torch.tensor([0.0, 0.0, -0.3], device=self.device).repeat((self.num_envs, 1))
        self.right_arm_local_vec = torch.tensor([0.0, 0.0, -0.3], device=self.device).repeat((self.num_envs, 1))

        # Init gait parameter
        self.gait_phase = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.gait_cycle = torch.full(
            (self.num_envs,), self.cfg.gait.gait_cycle, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.phase_ratio = torch.tensor(
            [self.cfg.gait.gait_air_ratio_l, self.cfg.gait.gait_air_ratio_r], dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)
        self.phase_offset = torch.tensor(
            [self.cfg.gait.gait_phase_offset_l, self.cfg.gait.gait_phase_offset_r],
            dtype=torch.float,
            device=self.device,
        ).repeat(self.num_envs, 1)
        self.action = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.init_obs_buffer()

    def visualize_motion(self, time):
        """
        Update the robot simulation state based on the AMP motion capture data at a given time.

        This function sets the joint positions and velocities, root position and orientation,
        and linear/angular velocities according to the AMP motion frame at the specified time,
        then steps the simulation and updates the scene.

        Args:
            time (float): The time (in seconds) at which to fetch the AMP motion frame.

        Returns:
            None
        """
        visual_motion_frame = self.amp_loader_display.get_full_frame_at_time(0, time)
        device = self.device
        loader = self.amp_loader_display
        num_j = loader.num_joints

        dof_pos = torch.zeros((self.num_envs, self.robot.num_joints), device=device)
        dof_vel = torch.zeros((self.num_envs, self.robot.num_joints), device=device)

        # motion 布局: root(3)+euler(3)+dof | root_lin_vel(3)+root_ang_vel(3)+dof_vel
        # Roban 21 关节: waist(1)+leg_l(6)+leg_r(6)+zarm_l(4)+zarm_r(4)
        if num_j == 21:
            dof_pos[:, self.waist_ids] = visual_motion_frame[6:7]
            dof_pos[:, self.left_leg_ids] = visual_motion_frame[7:13]
            dof_pos[:, self.right_leg_ids] = visual_motion_frame[13:19]
            dof_pos[:, self.left_arm_ids] = visual_motion_frame[19:23]
            dof_pos[:, self.right_arm_ids] = visual_motion_frame[23:27]
            vel_off = loader.pose_end_idx + 6
            dof_vel[:, self.waist_ids] = visual_motion_frame[vel_off : vel_off + 1]
            dof_vel[:, self.left_leg_ids] = visual_motion_frame[vel_off + 1 : vel_off + 7]
            dof_vel[:, self.right_leg_ids] = visual_motion_frame[vel_off + 7 : vel_off + 13]
            dof_vel[:, self.left_arm_ids] = visual_motion_frame[vel_off + 13 : vel_off + 17]
            dof_vel[:, self.right_arm_ids] = visual_motion_frame[vel_off + 17 : vel_off + 21]
        else:
            # TienKung 20 关节: leg_l(6)+leg_r(6)+arm_l(4)+arm_r(4)，无 waist
            dof_pos[:, self.left_leg_ids] = visual_motion_frame[6:12]
            dof_pos[:, self.right_leg_ids] = visual_motion_frame[12:18]
            dof_pos[:, self.left_arm_ids] = visual_motion_frame[18:22]
            dof_pos[:, self.right_arm_ids] = visual_motion_frame[22:26]
            dof_vel[:, self.left_leg_ids] = visual_motion_frame[32:38]
            dof_vel[:, self.right_leg_ids] = visual_motion_frame[38:44]
            dof_vel[:, self.left_arm_ids] = visual_motion_frame[44:48]
            dof_vel[:, self.right_arm_ids] = visual_motion_frame[48:52]

        self.robot.write_joint_position_to_sim(dof_pos)
        self.robot.write_joint_velocity_to_sim(dof_vel)

        env_ids = torch.arange(self.num_envs, device=device)

        root_pos = visual_motion_frame[:3].clone()
        root_pos[2] += -0.02

        euler = visual_motion_frame[3:6].cpu().numpy()
        quat_xyzw = Rotation.from_euler("XYZ", euler, degrees=False).as_quat()  # [x, y, z, w]
        quat_wxyz = torch.tensor(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=torch.float32, device=device
        )

        lin_vel = visual_motion_frame[loader.pose_end_idx : loader.pose_end_idx + 3].clone()
        ang_vel = torch.zeros_like(lin_vel)

        # root state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        root_state = torch.zeros((self.num_envs, 13), device=device)
        root_state[:, 0:3] = torch.tile(root_pos.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 3:7] = torch.tile(quat_wxyz.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 7:10] = torch.tile(lin_vel.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 10:13] = torch.tile(ang_vel.unsqueeze(0), (self.num_envs, 1))

        self.robot.write_root_state_to_sim(root_state, env_ids)
        self.sim.render()
        self.sim.step()
        self.scene.update(dt=self.step_dt)

        left_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[0], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_rotate(self.robot.data.body_state_w[:, self.elbow_body_ids[0], 3:7], self.left_arm_local_vec)
        )
        right_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[1], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_rotate(self.robot.data.body_state_w[:, self.elbow_body_ids[1], 3:7], self.right_arm_local_vec)
        )
        left_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_hand_pos)
        right_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_hand_pos)
        left_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        right_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        left_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_foot_pos)
        right_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_foot_pos)

        self.left_leg_dof_pos = dof_pos[:, self.left_leg_ids]
        self.right_leg_dof_pos = dof_pos[:, self.right_leg_ids]
        self.left_leg_dof_vel = dof_vel[:, self.left_leg_ids]
        self.right_leg_dof_vel = dof_vel[:, self.right_leg_ids]
        self.left_arm_dof_pos = dof_pos[:, self.left_arm_ids]
        self.right_arm_dof_pos = dof_pos[:, self.right_arm_ids]
        self.left_arm_dof_vel = dof_vel[:, self.left_arm_ids]
        self.right_arm_dof_vel = dof_vel[:, self.right_arm_ids]
        # 输出与 get_amp_obs_for_expert_trans 一致的 67 维 AMP expert 格式
        # 21 joint pos: waist, left_leg, right_leg, left_arm, right_arm
        # 21 joint vel: 同上
        # 1 root height + 6 root rotation + 3 root lin_vel + 3 root ang_vel
        # 12 end effector: left_hand, right_hand, left_foot, right_foot
        if num_j == 21:
            waist_pos = dof_pos[:, self.waist_ids]
            waist_vel = dof_vel[:, self.waist_ids]
            joint_pos_21 = torch.cat(
                (waist_pos, self.left_leg_dof_pos, self.right_leg_dof_pos, self.left_arm_dof_pos, self.right_arm_dof_pos),
                dim=-1,
            )
            joint_vel_21 = torch.cat(
                (waist_vel, self.left_leg_dof_vel, self.right_leg_dof_vel, self.left_arm_dof_vel, self.right_arm_dof_vel),
                dim=-1,
            )
            
            # Root state information
            root_quat = self.robot.data.root_state_w[:, 3:7]  # Quaternion (w, x, y, z)
            root_height = self.robot.data.root_state_w[:, 2:3]  # Root height (z coordinate)
            
            # Convert quaternion to tangent and normal vectors (6D)
            root_rotation = self._quaternion_to_tangent_and_normal(root_quat)
            
            # Root linear and angular velocities (in root frame)
            root_lin_vel_w = self.robot.data.root_state_w[:, 7:10]  # World frame
            root_ang_vel_w = self.robot.data.root_state_w[:, 10:13]  # World frame
            
            # Transform velocities to root frame
            root_lin_vel_b = quat_apply(quat_conjugate(root_quat), root_lin_vel_w)
            root_ang_vel_b = quat_apply(quat_conjugate(root_quat), root_ang_vel_w)
            
            # Return 67-dim AMP observation (same format as get_amp_obs_for_expert_trans)
            return torch.cat(
                (
                    joint_pos_21,
                    joint_vel_21,
                    root_height,
                    root_rotation,
                    root_lin_vel_b,
                    root_ang_vel_b,
                    left_hand_pos,
                    right_hand_pos,
                    left_foot_pos,
                    right_foot_pos,
                ),
                dim=-1,
            )
        else:
            # TienKung 20 关节，无 waist
            return torch.cat(
                (
                    self.right_arm_dof_pos,
                    self.left_arm_dof_pos,
                    self.right_leg_dof_pos,
                    self.left_leg_dof_pos,
                    self.right_arm_dof_vel,
                    self.left_arm_dof_vel,
                    self.right_leg_dof_vel,
                    self.left_leg_dof_vel,
                    left_hand_pos,
                    right_hand_pos,
                    left_foot_pos,
                    right_foot_pos,
                ),
                dim=-1,
            )

    def compute_current_observations(self):
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        root_lin_vel = robot.data.root_lin_vel_b
        feet_contact = (torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5).float()

        # 构建基础观察值（不包含步态信息）
        base_obs = [
            ang_vel * self.obs_scales.ang_vel,  # 3
            projected_gravity * self.obs_scales.projected_gravity,  # 3
            command * self.obs_scales.commands,  # 3
            joint_pos * self.obs_scales.joint_pos,  # 21 (waist+legs+arms)
            joint_vel * self.obs_scales.joint_vel,  # 21
            action * self.obs_scales.actions,  # 21
        ]
        
        # 根据配置决定是否包含步态信息
        if getattr(self.cfg.robot, 'include_gait_in_obs', True):
            base_obs.extend([
                torch.sin(2 * torch.pi * self.gait_phase),  # 2
                torch.cos(2 * torch.pi * self.gait_phase),  # 2
                self.phase_ratio,  # 2
            ])
        
        current_actor_obs = torch.cat(base_obs, dim=-1)
        current_critic_obs = torch.cat([current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact], dim=-1)

        return current_actor_obs, current_critic_obs

    def compute_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return

        # Reset buffer
        self.avg_feet_force_per_step[env_ids] = 0.0
        self.avg_feet_speed_per_step[env_ids] = 0.0

        # ── 速度课程学习 ──
        vel_cur_extras = self._update_velocity_curriculum(env_ids)

        self.extras["log"] = dict()
        if vel_cur_extras:
            self.extras["log"].update(vel_cur_extras)
        if self.cfg.scene.terrain_generator is not None:
            if self.cfg.scene.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf

        self.command_generator.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0

        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):
        delayed_actions = self.action_buffer.compute(actions)
        self.action = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)

        processed_actions = self.action * self.action_scale + self.robot.data.default_joint_pos

        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

            self.avg_feet_force_per_step += torch.norm(
                self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :3], dim=-1
            )
            self.avg_feet_speed_per_step += torch.norm(self.robot.data.body_lin_vel_w[:, self.feet_body_ids, :], dim=-1)

        self.avg_feet_force_per_step /= self.cfg.sim.decimation
        self.avg_feet_speed_per_step /= self.cfg.sim.decimation

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        self._calculate_gait_para()

        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # ── 累积速度跟踪误差（供课程学习使用）──
        if self._vel_curriculum_enabled:
            self._accumulate_tracking_error()

        # ── 统一死区（Unified Deadband）──
        # 计算 is_command_active，供步态奖励门控和 stand_still_penalty 使用。
        # 必须在 reward_manager.compute() 之前计算，因为奖励函数会读取此属性。
        self._compute_command_deadband()

        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(self.reset_env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}

        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        # 接触力终止（仅 base_link 碰地）
        reset_buf = torch.any(
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
                    dim=-1,
                ),
                dim=1,
            )[0]
            > 1.0,
            dim=1,
        )

        # 与 amp_roban_share 一致：根部高度低于阈值时终止（真正摔倒）
        min_height = getattr(self.cfg.robot, "terminate_min_height", 0.0)
        if min_height > 0.0:
            root_height = self.robot.data.root_pos_w[:, 2]
            reset_buf |= root_height < min_height

        time_out_buf = self.episode_length_buf >= self.max_episode_length
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            noise_vec[:3] = noise_scales.lin_vel * self.obs_scales.lin_vel
            noise_vec[3:6] = noise_scales.ang_vel * self.obs_scales.ang_vel
            noise_vec[6:9] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            noise_vec[9:12] = 0
            noise_vec[12 : 12 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
            noise_vec[12 + self.num_actions : 12 + self.num_actions * 2] = (
                noise_scales.joint_vel * self.obs_scales.joint_vel
            )
            noise_vec[12 + self.num_actions * 2 : 12 + self.num_actions * 3] = 0.0
            noise_vec[12 + self.num_actions * 3 : 18 + self.num_actions * 3] = 0.0
            self.noise_scale_vec = noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Unified Deadband Logic
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_command_deadband(self):
        """计算统一死区 mask: is_command_active。

        指令"有效"的条件（OR 逻辑）：
          |cmd_x| > lin_vel_deadband  OR  |cmd_y| > lat_vel_deadband  OR  |cmd_yaw| > ang_vel_deadband

        效果：
          - is_command_active = True  → 步态奖励正常生效
          - is_command_active = False → 步态奖励归零 + stand_still_penalty 生效
        """
        cur_cfg = getattr(self.cfg, "velocity_curriculum", None)
        lin_db = getattr(cur_cfg, "lin_vel_deadband", 0.25) if cur_cfg else 0.25
        lat_db = getattr(cur_cfg, "lat_vel_deadband", 0.2) if cur_cfg else 0.2
        ang_db = getattr(cur_cfg, "ang_vel_deadband", 0.25) if cur_cfg else 0.25

        cmd = self.command_generator.command
        self.is_command_active = (
            torch.abs(cmd[:, 0]) > lin_db
        ) | (
            torch.abs(cmd[:, 1]) > lat_db
        ) | (
            torch.abs(cmd[:, 2]) > ang_db
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Velocity Curriculum Learning
    # ══════════════════════════════════════════════════════════════════════════

    def _init_velocity_curriculum(self):
        """初始化速度课程学习状态。"""
        cur_cfg = getattr(self.cfg, "velocity_curriculum", None)
        self._vel_curriculum_enabled = cur_cfg is not None and getattr(cur_cfg, "enable", False)
        if not self._vel_curriculum_enabled:
            return

        self._vel_cur_cfg = cur_cfg
        self._vel_cur_level = 0
        self._vel_cur_max_level = len(cur_cfg.levels) - 1

        # 统计缓冲区：episode 长度 + 跟踪误差
        self._vel_cur_ep_lengths = deque(maxlen=cur_cfg.buffer_size)
        self._vel_cur_ep_lin_errs = deque(maxlen=cur_cfg.buffer_size)
        self._vel_cur_ep_ang_errs = deque(maxlen=cur_cfg.buffer_size)

        # 每步跟踪误差累积器（per-env）
        self._vel_cur_lin_err_acc = torch.zeros(self.num_envs, device=self.device)
        self._vel_cur_ang_err_acc = torch.zeros(self.num_envs, device=self.device)

        # 冷却机制：记录级别变更时的 env-step 时间戳
        # 只有在该时间戳之后才开始的 episode 才会被记录
        self._vel_cur_level_change_step = 0  # 初始时为 0，所有 episode 都有效

        # ── 动态奖励缩放：备份原始权重 ──
        # 从 RewardManager 中读取每个 reward term 的原始权重并保存，
        # 后续 _apply_reward_scales 通过 base_weight × multiplier 计算，避免复合乘法漂移。
        self._reward_base_weights: dict[str, float] = {}
        for name in self.reward_manager._term_names:
            cfg = self.reward_manager.get_term_cfg(name)
            self._reward_base_weights[name] = cfg.weight

        # 应用初始级别（Level 0）— 同时更新速度范围和奖励权重
        self._apply_velocity_level(0)

        # ── 启动信息 ──
        print(f"[VelocityCurriculum] Pragmatic Natural Walking — {len(cur_cfg.levels)} 级")
        level_names = {0: "Ignition", 1: "Steering", 2: "Omni & Refinement"}
        for i, lv in enumerate(cur_cfg.levels):
            name = level_names.get(i, f"Level {i}")
            print(f"  Level {i} ({name}): {lv}")
        print(f"  promote_threshold: ep > {cur_cfg.promote_threshold*100:.0f}% max_episode_length")
        print(f"  demote_threshold:  ep < {cur_cfg.demote_threshold*100:.0f}% max_episode_length")
        criteria_list = getattr(cur_cfg, "promote_criteria", None)
        if criteria_list:
            for i, c in enumerate(criteria_list):
                print(f"  L{i}→L{i+1} criteria: {c}")
        print(f"  Deadband: lin_vel={cur_cfg.lin_vel_deadband}m/s, ang_vel={cur_cfg.ang_vel_deadband}rad/s")
        if getattr(cur_cfg, "reward_multipliers", None):
            print(f"  [DynamicRewardScaling] Level 0 multipliers: {cur_cfg.reward_multipliers[0]}")

    def _apply_velocity_level(self, level: int):
        """将指定级别的速度范围写入 command_generator，并更新奖励权重缩放。"""
        ranges = self._vel_cur_cfg.levels[level]
        cmd_ranges = self.command_generator.cfg.ranges
        cmd_ranges.lin_vel_x = tuple(ranges["lin_vel_x"])
        cmd_ranges.lin_vel_y = tuple(ranges["lin_vel_y"])
        cmd_ranges.ang_vel_z = tuple(ranges["ang_vel_z"])

        # 动态奖励缩放
        self._apply_reward_scales(level)

    def _apply_reward_scales(self, level: int):
        """根据课程级别，用 base_weight × multiplier 更新 RewardManager 中各 term 的权重。

        - reward_multipliers[level] 中列出的 term：使用指定乘数
        - 未列出的 term：乘数默认为 1.0（保持原始权重）
        - 始终从 _reward_base_weights（原始权重备份）出发计算，避免复合乘法漂移
        """
        multipliers_list = getattr(self._vel_cur_cfg, "reward_multipliers", None)
        if not multipliers_list:
            return  # 未配置动态缩放，跳过

        # 获取当前级别的乘数表（超出范围则用空 dict → 全部 1.0）
        if level < len(multipliers_list):
            level_multipliers = multipliers_list[level]
        else:
            level_multipliers = {}

        for name in self.reward_manager._term_names:
            base_w = self._reward_base_weights[name]
            multiplier = level_multipliers.get(name, 1.0)
            new_weight = base_w * multiplier

            # 直接修改 term_cfg.weight（RewardManager.compute 每步读取此值）
            term_cfg = self.reward_manager.get_term_cfg(name)
            term_cfg.weight = new_weight

    def _accumulate_tracking_error(self):
        """每个 env step 调用，累积线速度和角速度跟踪误差。

        在 step() 中调用，用于在 episode 结束时计算平均跟踪误差。
        """
        # 线速度误差：在 yaw frame 下计算 ||cmd_xy - vel_xy||
        vel_yaw = math_utils.quat_rotate_inverse(
            math_utils.yaw_quat(self.robot.data.root_quat_w),
            self.robot.data.root_lin_vel_w[:, :3],
        )
        lin_err = torch.norm(
            self.command_generator.command[:, :2] - vel_yaw[:, :2], dim=1
        )
        # 角速度误差：|cmd_yaw - actual_yaw_rate|
        ang_err = torch.abs(
            self.command_generator.command[:, 2] - self.robot.data.root_ang_vel_w[:, 2]
        )

        self._vel_cur_lin_err_acc += lin_err
        self._vel_cur_ang_err_acc += ang_err

    def _update_velocity_curriculum(self, env_ids) -> dict:
        """在 reset 时调用，记录 episode 统计并判断是否升/降级。

        改进点（相比原版）：
        1. 防止站桩升级：升级需要同时满足 episode 长度 + 跟踪误差两个条件
        2. 冷却机制：级别变更后，忽略在旧级别下开始的 episode
        3. 更大的缓冲区：默认 2000（适配大规模并行环境）
        """
        if not self._vel_curriculum_enabled:
            return {}

        # 当前全局 env step（不是 sim step）
        global_env_step = self.sim_step_counter // self.cfg.sim.decimation

        # ── 记录刚结束的 episode 统计（附冷却过滤）──
        ep_lengths = self.episode_length_buf[env_ids]  # 刚结束的 episode 长度（steps）
        for i, env_id in enumerate(env_ids):
            ep_len = ep_lengths[i].item()
            if ep_len <= 0:
                continue  # 排除初始 reset

            # 冷却过滤：该 episode 是否在新级别生效之后才开始？
            episode_start_step = global_env_step - ep_len
            if episode_start_step < self._vel_cur_level_change_step:
                # 该 episode 在旧级别下启动，丢弃（避免数据滞后）
                continue

            ep_length_s = ep_len * self.step_dt
            # 计算该 episode 的平均跟踪误差（RMSE 近似：累积值 / 步数）
            mean_lin_err = self._vel_cur_lin_err_acc[env_id].item() / max(ep_len, 1)
            mean_ang_err = self._vel_cur_ang_err_acc[env_id].item() / max(ep_len, 1)

            self._vel_cur_ep_lengths.append(ep_length_s)
            self._vel_cur_ep_lin_errs.append(mean_lin_err)
            self._vel_cur_ep_ang_errs.append(mean_ang_err)

        # ── 重置已结束 episode 的累积器 ──
        self._vel_cur_lin_err_acc[env_ids] = 0
        self._vel_cur_ang_err_acc[env_ids] = 0

        # ── 检查是否有足够样本做决策 ──
        min_samples = self._vel_cur_cfg.buffer_size // 4
        if len(self._vel_cur_ep_lengths) < min_samples:
            extras = {"Curriculum/velocity_level": float(self._vel_cur_level)}
            extras.update(self._get_reward_scaling_extras())
            return extras

        # ── 计算统计量 ──
        mean_ep_length = sum(self._vel_cur_ep_lengths) / len(self._vel_cur_ep_lengths)
        mean_lin_err = sum(self._vel_cur_ep_lin_errs) / len(self._vel_cur_ep_lin_errs)
        mean_ang_err = sum(self._vel_cur_ep_ang_errs) / len(self._vel_cur_ep_ang_errs)

        promote_thresh = self._vel_cur_cfg.promote_threshold * self.max_episode_length_s
        demote_thresh = self._vel_cur_cfg.demote_threshold * self.max_episode_length_s

        # ── 升级判断（per-level 条件 + 存活时间）──
        # 使用 promote_criteria[current_level] 获取当前级别的升级误差上限。
        # 缺失的 key 不检查（等价于 +inf）。
        criteria_list = getattr(self._vel_cur_cfg, "promote_criteria", None)
        can_promote = self._vel_cur_level < self._vel_cur_max_level and mean_ep_length > promote_thresh
        if can_promote and criteria_list and self._vel_cur_level < len(criteria_list):
            criteria = criteria_list[self._vel_cur_level]
            lin_limit = criteria.get("lin_vel_err_limit", float("inf"))
            ang_limit = criteria.get("ang_vel_err_limit", float("inf"))
            can_promote = mean_lin_err < lin_limit and mean_ang_err < ang_limit
        elif can_promote:
            # 无 promote_criteria 配置时，不额外检查误差（向后兼容）
            lin_limit = float("inf")
            ang_limit = float("inf")

        if can_promote:
            self._vel_cur_level += 1
            self._apply_velocity_level(self._vel_cur_level)
            # 清空缓冲区 + 设置冷却时间戳
            self._vel_cur_ep_lengths.clear()
            self._vel_cur_ep_lin_errs.clear()
            self._vel_cur_ep_ang_errs.clear()
            self._vel_cur_level_change_step = global_env_step + self._vel_cur_cfg.cooldown_steps
            level_names = {0: "Ignition", 1: "Steering", 2: "Omni"}
            lv_name = level_names.get(self._vel_cur_level, f"Level {self._vel_cur_level}")
            print(
                f"[VelocityCurriculum] ↑ 升级到 Level {self._vel_cur_level} ({lv_name}): "
                f"{self._vel_cur_cfg.levels[self._vel_cur_level]}  "
                f"(ep={mean_ep_length:.1f}s>{promote_thresh:.1f}s, "
                f"lin_err={mean_lin_err:.3f}, ang_err={mean_ang_err:.3f})"
            )

        # ── 降级判断（只看存活时间，摔太多就降级）──
        elif self._vel_cur_level > 0 and mean_ep_length < demote_thresh:
            self._vel_cur_level -= 1
            self._apply_velocity_level(self._vel_cur_level)
            self._vel_cur_ep_lengths.clear()
            self._vel_cur_ep_lin_errs.clear()
            self._vel_cur_ep_ang_errs.clear()
            self._vel_cur_level_change_step = global_env_step + self._vel_cur_cfg.cooldown_steps
            level_names = {0: "Ignition", 1: "Steering", 2: "Omni"}
            lv_name = level_names.get(self._vel_cur_level, f"Level {self._vel_cur_level}")
            print(
                f"[VelocityCurriculum] ↓ 降级到 Level {self._vel_cur_level} ({lv_name}): "
                f"{self._vel_cur_cfg.levels[self._vel_cur_level]}  "
                f"(ep={mean_ep_length:.1f}s<{demote_thresh:.1f}s)"
            )

        extras = {
            "Curriculum/velocity_level": float(self._vel_cur_level),
            "Curriculum/mean_episode_length": mean_ep_length,
            "Curriculum/mean_lin_vel_error": mean_lin_err,
            "Curriculum/mean_ang_vel_error": mean_ang_err,
        }
        extras.update(self._get_reward_scaling_extras())
        return extras

    def _get_reward_scaling_extras(self) -> dict:
        """返回当前动态奖励缩放状态的 TensorBoard 日志条目。"""
        extras = {}
        if not hasattr(self, "_reward_base_weights"):
            return extras
        # 记录关键奖励的当前有效权重，方便在 TensorBoard 中验证动态缩放是否生效
        tracked_names = (
            "track_lin_vel_xy_exp", "track_ang_vel_z_exp",
            "flat_orientation_exp", "base_height_penalty",
            "feet_air_time", "feet_contact_time_symmetry", "feet_distance",
        )
        for name in tracked_names:
            if name in self._reward_base_weights:
                current_cfg = self.reward_manager.get_term_cfg(name)
                extras[f"Curriculum/reward_weight/{name}"] = current_cfg.weight
        return extras

    def update_terrain_levels(self, env_ids):
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {}
        extras["Curriculum/terrain_levels"] = torch.mean(self.scene.terrain.terrain_levels.float())
        return extras

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, self.extras

    def get_amp_obs_for_expert_trans(self):
        """AMP obs for 21-DoF Roban with root state: 
        joint_pos(21) + joint_vel(21) + root_height(1) + 
        root_rotation(6) + root_lin_vel(3) + root_ang_vel(3) + 
        end_effector_pos(12) = 67.
        Order: waist, left_leg, right_leg, left_arm, right_arm (pos then vel), 
        then root state, then hand/foot positions.
        """
        left_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[0], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_rotate(self.robot.data.body_state_w[:, self.elbow_body_ids[0], 3:7], self.left_arm_local_vec)
        )
        right_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[1], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_rotate(self.robot.data.body_state_w[:, self.elbow_body_ids[1], 3:7], self.right_arm_local_vec)
        )
        left_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_hand_pos)
        right_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_hand_pos)
        left_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        right_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        left_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_foot_pos)
        right_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_foot_pos)
        waist_pos = self.robot.data.joint_pos[:, self.waist_ids]
        waist_vel = self.robot.data.joint_vel[:, self.waist_ids]
        self.left_leg_dof_pos = self.robot.data.joint_pos[:, self.left_leg_ids]
        self.right_leg_dof_pos = self.robot.data.joint_pos[:, self.right_leg_ids]
        self.left_leg_dof_vel = self.robot.data.joint_vel[:, self.left_leg_ids]
        self.right_leg_dof_vel = self.robot.data.joint_vel[:, self.right_leg_ids]
        self.left_arm_dof_pos = self.robot.data.joint_pos[:, self.left_arm_ids]
        self.right_arm_dof_pos = self.robot.data.joint_pos[:, self.right_arm_ids]
        self.left_arm_dof_vel = self.robot.data.joint_vel[:, self.left_arm_ids]
        self.right_arm_dof_vel = self.robot.data.joint_vel[:, self.right_arm_ids]
        # 21 joint pos: waist, left_leg, right_leg, left_arm, right_arm
        joint_pos_21 = torch.cat(
            (waist_pos, self.left_leg_dof_pos, self.right_leg_dof_pos, self.left_arm_dof_pos, self.right_arm_dof_pos),
            dim=-1,
        )
        joint_vel_21 = torch.cat(
            (waist_vel, self.left_leg_dof_vel, self.right_leg_dof_vel, self.left_arm_dof_vel, self.right_arm_dof_vel),
            dim=-1,
        )
        
        # Root state information
        root_quat = self.robot.data.root_state_w[:, 3:7]  # Quaternion (w, x, y, z)
        root_height = self.robot.data.root_state_w[:, 2:3]  # Root height (z coordinate)
        
        # Convert quaternion to tangent and normal vectors (6D)
        root_rotation = self._quaternion_to_tangent_and_normal(root_quat)
        
        # Root linear and angular velocities (in root frame)
        root_lin_vel_w = self.robot.data.root_state_w[:, 7:10]  # World frame
        root_ang_vel_w = self.robot.data.root_state_w[:, 10:13]  # World frame
        
        # Transform velocities to root frame
        root_lin_vel_b = quat_apply(quat_conjugate(root_quat), root_lin_vel_w)
        root_ang_vel_b = quat_apply(quat_conjugate(root_quat), root_ang_vel_w)
        
        return torch.cat(
            (
                joint_pos_21,
                joint_vel_21,
                root_height,
                root_rotation,
                root_lin_vel_b,
                root_ang_vel_b,
                left_hand_pos,
                right_hand_pos,
                left_foot_pos,
                right_foot_pos,
            ),
            dim=-1,
        )
    
    @staticmethod
    def _quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to tangent and normal vectors.
        
        Args:
            q: Quaternion tensor (w, x, y, z) with shape (..., 4)
        
        Returns:
            Concatenated tangent and normal vectors with shape (..., 6)
        """
        ref_tangent = torch.zeros_like(q[..., :3])
        ref_normal = torch.zeros_like(q[..., :3])
        ref_tangent[..., 0] = 1  # Forward direction (x-axis)
        ref_normal[..., -1] = 1   # Up direction (z-axis)
        
        tangent = quat_rotate(q, ref_tangent)
        normal = quat_rotate(q, ref_normal)
        
        return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)

    def _calculate_gait_para(self) -> None:
        """
        Update gait phase parameters based on simulation time and offset.
        """
        t = self.episode_length_buf * self.step_dt / self.gait_cycle
        self.gait_phase[:, 0] = (t + self.phase_offset[:, 0]) % 1.0
        self.gait_phase[:, 1] = (t + self.phase_offset[:, 1]) % 1.0
