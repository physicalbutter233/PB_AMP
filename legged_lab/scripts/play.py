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
import copy
import os

import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Start camera rendering
if "sensor" in args_cli.task:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg


def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    
    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # 首先创建运行环境（用于加载训练时的模型）
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.ranges.lin_vel_x = (0.5, 0.5)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)

    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)  # 禁止旋转，只向前走
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"

    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.terrain_generator.difficulty_range = (0.4, 0.4)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    runner_class: OnPolicyRunner | AmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # For AMP runners, skip discriminator if dimension mismatch (play doesn't need discriminator)
    if isinstance(runner, AmpOnPolicyRunner):
        runner.load(resume_path, load_optimizer=False, skip_discriminator_on_mismatch=True)
    else:
        runner.load(resume_path, load_optimizer=False)

    # 导出模型：创建适配层将单帧观察值转换为训练格式
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    
    # 获取当前环境的观察空间维度
    test_obs, _ = env.get_observations()
    actual_obs_dim = test_obs.shape[1]
    history_length = env.cfg.robot.actor_obs_history_length
    single_frame_dim = actual_obs_dim // history_length
    
    # 检查是否包含步态信息
    include_gait = getattr(env.cfg.robot, 'include_gait_in_obs', True)
    base_obs_dim = 3 + 3 + 3 + env.num_actions + env.num_actions + env.num_actions  # ang_vel + gravity + command + joint_pos + joint_vel + action
    gait_dim = 6 if include_gait else 0  # sin(2) + cos(2) + phase_ratio(2)
    expected_single_frame_dim = base_obs_dim + gait_dim
    
    print(f"[INFO] Current observation space:")
    print(f"  - Total obs dimension: {actual_obs_dim}")
    print(f"  - History length: {history_length}")
    print(f"  - Single frame dimension: {single_frame_dim}")
    print(f"  - Expected single frame dimension: {expected_single_frame_dim}")
    print(f"  - Includes gait: {include_gait}")
    print(f"  - Number of actions: {env.num_actions}")
    
    # 创建适配层：将单帧观察值转换为训练时的格式（10帧历史）
    class ObsAdapterActor(torch.nn.Module):
        """适配层Actor：将单帧观察值转换为训练时的格式"""
        def __init__(self, original_actor, obs_normalizer=None, single_frame_dim=None, history_length=None):
            super().__init__()
            self.original_actor = original_actor
            self.obs_normalizer = obs_normalizer
            self.single_frame_dim = single_frame_dim
            self.history_length = history_length
            
            # 为了兼容ONNX导出函数，需要让这个模块看起来像Sequential
            # 创建一个假的Linear层（输入维度，输出维度，实际上不会被使用）
            self._dummy_layer = torch.nn.Linear(single_frame_dim, single_frame_dim)
            
        def __getitem__(self, idx):
            """模拟Sequential的行为，用于ONNX导出"""
            if idx == 0:
                return self._dummy_layer
            raise IndexError(f"Index {idx} out of range")
            
        def forward(self, obs_single_frame):
            """
            Args:
                obs_single_frame: [batch_size, single_frame_dim] - 单帧观察值
            Returns:
                actions: [batch_size, num_actions] - 动作
            """
            batch_size = obs_single_frame.shape[0]
            
            # 单帧 -> 多帧历史（重复当前帧history_length次）
            obs_history = obs_single_frame.unsqueeze(1).repeat(1, self.history_length, 1)  # [batch, history_length, single_frame_dim]
            obs_history = obs_history.reshape(batch_size, -1)  # [batch, history_length * single_frame_dim]
            
            # 归一化（如果启用）
            if self.obs_normalizer is not None:
                obs_history = self.obs_normalizer(obs_history)
            
            # 直接通过actor模块
            return self.original_actor(obs_history)
    
    class ObsAdapter(torch.nn.Module):
        """适配层：包装ObsAdapterActor以符合导出函数的接口"""
        is_recurrent = False  # 导出函数需要的属性
        
        def __init__(self, original_policy, obs_normalizer=None, single_frame_dim=None, history_length=None):
            super().__init__()
            # 创建独立的actor模块，避免循环引用
            self.actor = ObsAdapterActor(original_policy, obs_normalizer, single_frame_dim, history_length)
            
        def forward(self, obs_single_frame):
            """转发到actor模块"""
            return self.actor(obs_single_frame)
    
    # 创建适配后的策略
    adapted_policy = ObsAdapter(
        runner.alg.policy.actor, 
        runner.obs_normalizer if runner.cfg.get("empirical_normalization", False) else None,
        single_frame_dim=single_frame_dim,
        history_length=history_length
    )
    adapted_policy.eval()
    
    # 创建测试输入
    test_obs_single = torch.zeros(1, single_frame_dim, device=agent_cfg.device)
    with torch.inference_mode():
        test_action = adapted_policy(test_obs_single)
    print(f"[INFO] Export observation dimension: {single_frame_dim} (single-frame)")
    print(f"[INFO] Test export policy: input {test_obs_single.shape} -> output {test_action.shape}")
    
    # 导出适配后的模型
    export_policy_as_jit(adapted_policy, None, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        adapted_policy, normalizer=None, path=export_model_dir, filename="policy.onnx"
    )
    print(f"[INFO] Model exported successfully to {export_model_dir}")
    
    # 继续使用运行环境进行play
    policy = runner.get_inference_policy(device=env.device)

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard

        keyboard = Keyboard(env)  # noqa:F841

    obs, _ = env.get_observations()

    while simulation_app.is_running():

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)


if __name__ == "__main__":
    play()
    simulation_app.close()
