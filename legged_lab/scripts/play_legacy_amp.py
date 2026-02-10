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

"""
play_legacy_amp.py - 支持加载旧版本（108维）AMP模型的play脚本

该脚本会自动检测检查点中判别器的输入维度，并创建匹配的判别器以支持旧版本模型。
"""

import argparse
import copy
import os

import numpy as np
import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner
from rsl_rl.modules import Discriminator
from rsl_rl.utils import AMPLoader, Normalizer

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


class LegacyAmpOnPolicyRunner(AmpOnPolicyRunner):
    """
    支持旧版本（108维）AMP模型的Runner。
    
    在加载检查点之前，先检查判别器的输入维度，然后创建匹配的判别器。
    """
    
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device="cpu", checkpoint_path: str | None = None):
        """
        Args:
            checkpoint_path: 检查点路径，用于检测判别器维度
        """
        self.checkpoint_path = checkpoint_path
        # 先调用父类初始化，但会在load时重新创建判别器
        super().__init__(env, train_cfg, log_dir, device)
    
    def _create_discriminator_with_dim(self, input_dim: int, train_cfg: dict, device):
        """根据指定的输入维度创建判别器"""
        task_reward_lerp = train_cfg.get("amp_task_reward_lerp", 0.0)
        amp_reward_coef = train_cfg.get("amp_reward_coef", 0.3)
        
        discriminator = Discriminator(
            input_dim,
            amp_reward_coef,
            train_cfg["amp_discr_hidden_dims"],
            device,
            task_reward_lerp,
        ).to(device)
        
        print(f"[LegacyAmpOnPolicyRunner] 创建判别器，输入维度: {input_dim}")
        return discriminator
    
    def load(self, path: str, load_optimizer: bool = True):
        """加载检查点，自动检测并匹配判别器维度"""
        # 先加载检查点以检测判别器维度
        loaded_dict = torch.load(path, weights_only=False, map_location=self.device)
        
        dimension_mismatch = False
        checkpoint_input_dim = None
        
        # 检测判别器的输入维度
        if "discriminator_state_dict" in loaded_dict:
            disc_state = loaded_dict["discriminator_state_dict"]
            # 判别器第一层的权重形状是 [hidden_dim, input_dim]
            if "trunk.0.weight" in disc_state:
                checkpoint_input_dim = disc_state["trunk.0.weight"].shape[1]
                print(f"[LegacyAmpOnPolicyRunner] 检测到检查点判别器输入维度: {checkpoint_input_dim}")
                
                # 如果当前判别器维度不匹配，需要重新创建
                current_input_dim = self.alg.discriminator.input_dim
                if checkpoint_input_dim != current_input_dim:
                    dimension_mismatch = True
                    print(f"[LegacyAmpOnPolicyRunner] 维度不匹配！当前: {current_input_dim}, 检查点: {checkpoint_input_dim}")
                    print(f"[LegacyAmpOnPolicyRunner] 重新创建匹配的判别器...")
                    
                    # 重新创建判别器
                    new_discriminator = self._create_discriminator_with_dim(
                        checkpoint_input_dim, 
                        self.cfg, 
                        self.device
                    )
                    
                    # 替换算法中的判别器
                    self.alg.discriminator = new_discriminator
                    
                    # 更新amp_storage的维度（如果需要）
                    if hasattr(self.alg, 'amp_storage'):
                        # amp_storage存储的是单个state，所以维度是input_dim的一半
                        single_state_dim = checkpoint_input_dim // 2
                        from rsl_rl.storage import ReplayBuffer
                        self.alg.amp_storage = ReplayBuffer(
                            single_state_dim, 
                            self.alg.amp_storage.buffer_size, 
                            self.device
                        )
                    
                    # 如果维度是108（54*2），说明是旧版本，需要调整AMP数据加载
                    if checkpoint_input_dim == 108:
                        print("[LegacyAmpOnPolicyRunner] 检测到旧版本模型（108维），调整AMP数据加载...")
                        # 旧版本的AMP观测维度是54（21 joint_pos + 21 joint_vel + 12 end_effector）
                        # 需要确保AMP数据也使用54维
                        # 注意：这里假设motion文件已经是54维格式
                        pass
        
        # 如果维度不匹配，需要手动加载状态字典（避免父类load方法中的维度检查失败）
        if dimension_mismatch:
            print("[LegacyAmpOnPolicyRunner] 手动加载状态字典（维度已匹配）...")
            # 加载策略
            resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
            # 加载判别器
            self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
            # 加载AMP normalizer
            if "amp_normalizer" in loaded_dict:
                single_state_dim = checkpoint_input_dim // 2
                # 检查当前normalizer的维度
                current_dim = None
                if self.alg.amp_normalizer is not None and hasattr(self.alg.amp_normalizer, 'mean'):
                    # Normalizer的mean属性是numpy数组，其长度就是维度
                    mean = self.alg.amp_normalizer.mean
                    if isinstance(mean, np.ndarray):
                        current_dim = len(mean) if mean.ndim == 1 else mean.shape[0]
                    else:
                        current_dim = len(mean) if hasattr(mean, '__len__') else None
                
                # 如果维度不匹配，创建新的normalizer
                if self.alg.amp_normalizer is None or current_dim != single_state_dim:
                    print(f"[LegacyAmpOnPolicyRunner] 创建新的AMP normalizer，维度: {single_state_dim}")
                    self.alg.amp_normalizer = Normalizer(single_state_dim)
                
                # 加载normalizer（直接赋值，因为检查点中存储的就是Normalizer对象）
                try:
                    loaded_normalizer = loaded_dict["amp_normalizer"]
                    # 检查加载的normalizer维度是否匹配
                    if hasattr(loaded_normalizer, 'mean'):
                        loaded_dim = len(loaded_normalizer.mean) if isinstance(loaded_normalizer.mean, np.ndarray) else loaded_normalizer.mean.shape[0]
                        if loaded_dim == single_state_dim:
                            self.alg.amp_normalizer = loaded_normalizer
                            print("[LegacyAmpOnPolicyRunner] 已加载检查点中的AMP normalizer")
                        else:
                            print(f"[LegacyAmpOnPolicyRunner] 警告：检查点normalizer维度({loaded_dim})不匹配，使用新的normalizer")
                    else:
                        self.alg.amp_normalizer = loaded_normalizer
                        print("[LegacyAmpOnPolicyRunner] 已加载检查点中的AMP normalizer")
                except Exception as e:
                    print(f"[LegacyAmpOnPolicyRunner] 警告：无法加载检查点中的AMP normalizer: {e}，使用新的normalizer")
            # 加载RND（如果存在）
            if self.alg.rnd and "rnd_state_dict" in loaded_dict:
                self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
            # 加载观察normalizer（如果使用）
            if self.empirical_normalization:
                if resumed_training:
                    self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                    self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
                else:
                    self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            # 加载优化器（如果需要）
            if load_optimizer and "optimizer_state_dict" in loaded_dict:
                self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            print("[LegacyAmpOnPolicyRunner] 状态字典加载完成")
        else:
            # 维度匹配，调用父类的load方法
            super().load(path, load_optimizer)


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
    env_cfg.commands.ranges.lin_vel_x = (2.0, 2.0)
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

    # 使用LegacyAmpOnPolicyRunner来支持旧版本模型
    runner_class_name = agent_cfg.runner_class_name
    if runner_class_name == "AmpOnPolicyRunner":
        print("[INFO] 使用 LegacyAmpOnPolicyRunner 以支持旧版本AMP模型")
        runner = LegacyAmpOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, checkpoint_path=resume_path)
    else:
        runner_class: OnPolicyRunner | AmpOnPolicyRunner = eval(runner_class_name)
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
