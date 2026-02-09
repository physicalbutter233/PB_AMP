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
    env_cfg.commands.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)

    env_cfg.commands.ranges.ang_vel_z = (2.0, 2.0)  # 禁止旋转，只向前走
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
    runner.load(resume_path, load_optimizer=False)

    # 导出模型：创建适配层将单帧、无步态信息转换为训练格式
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    print("[INFO] Exporting model with single-frame observations (no gait info)...")
    
    # 创建适配层：将72维（单帧，无步态）转换为780维（10帧历史，有步态）
    # 单帧无步态：72维 = 3(ang_vel) + 3(gravity) + 3(command) + 21(joint_pos) + 21(joint_vel) + 21(action)
    # 单帧有步态：78维 = 72 + 6（步态信息：2(sin) + 2(cos) + 2(phase_ratio)）
    # 10帧历史：780维 = 78 * 10
    
    class ObsAdapterActor(torch.nn.Module):
        """适配层Actor：将单帧、无步态信息的观察值转换为训练时的格式"""
        def __init__(self, original_actor, obs_normalizer=None):
            super().__init__()
            # 只保存actor模块，避免TorchScript访问policy的其他属性
            self.original_actor = original_actor
            self.obs_normalizer = obs_normalizer
            
            # 为了兼容ONNX导出函数，需要让这个模块看起来像Sequential
            # 创建一个假的Sequential，包含一个Linear层（输入72维，输出72维，实际上不会被使用）
            # 导出函数会访问 self.actor[0].in_features
            self._dummy_layer = torch.nn.Linear(72, 72)
            
        def __getitem__(self, idx):
            """模拟Sequential的行为，用于ONNX导出"""
            if idx == 0:
                return self._dummy_layer
            raise IndexError(f"Index {idx} out of range")
            
        def forward(self, obs_single_frame_no_gait):
            """
            Args:
                obs_single_frame_no_gait: [batch_size, 72] - 单帧，无步态信息
            Returns:
                actions: [batch_size, 21] - 动作
            """
            # 72维单帧无步态 -> 78维单帧有步态（添加6维零步态信息）
            batch_size = obs_single_frame_no_gait.shape[0]
            gait_info = torch.zeros(batch_size, 6, device=obs_single_frame_no_gait.device, dtype=obs_single_frame_no_gait.dtype)
            obs_single_frame_with_gait = torch.cat([obs_single_frame_no_gait, gait_info], dim=1)  # [batch, 78]
            
            # 78维单帧 -> 780维10帧历史（重复当前帧10次）
            obs_history = obs_single_frame_with_gait.unsqueeze(1).repeat(1, 10, 1)  # [batch, 10, 78]
            obs_history = obs_history.reshape(batch_size, -1)  # [batch, 780]
            
            # 归一化（如果启用）
            if self.obs_normalizer is not None:
                obs_history = self.obs_normalizer(obs_history)
            
            # 直接通过actor模块（而不是policy.act_inference）
            return self.original_actor(obs_history)
    
    class ObsAdapter(torch.nn.Module):
        """适配层：包装ObsAdapterActor以符合导出函数的接口"""
        is_recurrent = False  # 导出函数需要的属性
        
        def __init__(self, original_policy, obs_normalizer=None):
            super().__init__()
            # 创建独立的actor模块，避免循环引用
            self.actor = ObsAdapterActor(original_policy, obs_normalizer)
            
        def forward(self, obs_single_frame_no_gait):
            """转发到actor模块"""
            return self.actor(obs_single_frame_no_gait)
    
    # 创建适配后的策略（只传递actor模块，避免TorchScript访问policy的其他属性）
    adapted_policy = ObsAdapter(runner.alg.policy.actor, runner.obs_normalizer if runner.cfg["empirical_normalization"] else None)
    adapted_policy.eval()
    
    # 创建测试输入（72维）
    test_obs = torch.zeros(1, 72, device=agent_cfg.device)
    with torch.inference_mode():
        test_action = adapted_policy(test_obs)
    print(f"[INFO] Export observation dimension: 72 (single-frame, no gait)")
    print(f"[INFO] Test export policy: input {test_obs.shape} -> output {test_action.shape}")
    
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
