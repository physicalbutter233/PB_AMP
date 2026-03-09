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
parser.add_argument(
    "--export_full_obs",
    action="store_true",
    help="Export ONNX with full observation (10 frames, 720 dim). Deploy with frameStack=10 to align with PB_AMP training.",
)

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

    # 首先创建运行环境（用于加载训练时的模型，与 amp_share Roban AMP play 对齐）
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.ranges.lin_vel_x = (0.6, 0.6)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)  # 禁止旋转，只向前走
    # 禁用速度课程：否则 _apply_velocity_level(0) 会覆盖上面的 ranges
    if hasattr(env_cfg, "velocity_curriculum") and env_cfg.velocity_curriculum is not None:
        env_cfg.velocity_curriculum.enable = False
    # 禁用地形+推力课程（play 不需要课程更新，与 amp_share PLAY 一致）
    if hasattr(env_cfg, "terrain_force_curriculum") and env_cfg.terrain_force_curriculum is not None:
        env_cfg.terrain_force_curriculum.enable = False
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

    # 导出模型
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    test_obs, _ = env.get_observations()
    actual_obs_dim = test_obs.shape[1]
    history_length = env.cfg.robot.actor_obs_history_length
    single_frame_dim = actual_obs_dim // history_length

    include_gait = getattr(env.cfg.robot, 'include_gait_in_obs', True)
    base_obs_dim = 3 + 3 + 3 + env.num_actions + env.num_actions + env.num_actions
    gait_dim = 6 if include_gait else 0
    expected_single_frame_dim = base_obs_dim + gait_dim

    print(f"[INFO] Current observation space:")
    print(f"  - Total obs dimension: {actual_obs_dim}")
    print(f"  - History length: {history_length}")
    print(f"  - Single frame dimension: {single_frame_dim}")
    print(f"  - Includes gait: {include_gait}")
    print(f"  - Export full obs (10 frames): {getattr(args_cli, 'export_full_obs', False)}")

    export_full_obs = getattr(args_cli, "export_full_obs", False)

    # 导出器要求 policy 有 .is_recurrent 和 .actor，用薄包装满足接口
    class RawPolicyWrapper(torch.nn.Module):
        is_recurrent = False
        def __init__(self, actor):
            super().__init__()
            self.actor = actor
        def forward(self, x):
            return self.actor(x)

    if export_full_obs:
        # 10 帧 720 维，部署端 frameStack=10
        policy_to_export = RawPolicyWrapper(runner.alg.policy.actor)
        policy_to_export.eval()
        test_input = torch.zeros(1, actual_obs_dim, device=agent_cfg.device)
        with torch.inference_mode():
            _ = policy_to_export(test_input)
        export_policy_as_jit(policy_to_export, None, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_to_export, normalizer=None, path=export_model_dir, filename="policy.onnx")
        print(f"[INFO] Exported full-obs policy: input {actual_obs_dim} (10 frames). Deploy with frameStack=10, numSingleObs={single_frame_dim}")
        policy = runner.get_inference_policy(device=env.device)
    elif history_length == 1:
        # 训练已是 1 帧 72 维（与 amp_share 一致），直接导出 policy，无需 ObsAdapter
        policy_to_export = RawPolicyWrapper(runner.alg.policy.actor)
        policy_to_export.eval()
        test_input = torch.zeros(1, actual_obs_dim, device=agent_cfg.device)
        with torch.inference_mode():
            _ = policy_to_export(test_input)
        export_policy_as_jit(policy_to_export, None, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_to_export, normalizer=None, path=export_model_dir, filename="policy.onnx")
        print(f"[INFO] Exported 1-frame policy: input {actual_obs_dim}. Deploy with frameStack=1 (align with amp_share)")
        policy = runner.get_inference_policy(device=env.device)
    else:
        # 训练为多帧，导出单帧 + ObsAdapter（部署 frameStack=1）
        class ObsAdapterActor(torch.nn.Module):
            def __init__(self, original_actor, obs_normalizer=None, single_frame_dim=None, history_length=None):
                super().__init__()
                self.original_actor = original_actor
                self.obs_normalizer = obs_normalizer
                self.single_frame_dim = single_frame_dim
                self.history_length = history_length
                self._dummy_layer = torch.nn.Linear(single_frame_dim, single_frame_dim)

            def __getitem__(self, idx):
                if idx == 0:
                    return self._dummy_layer
                raise IndexError(f"Index {idx} out of range")

            def forward(self, obs_single_frame):
                batch_size = obs_single_frame.shape[0]
                obs_history = obs_single_frame.unsqueeze(1).repeat(1, self.history_length, 1)
                obs_history = obs_history.reshape(batch_size, -1)
                if self.obs_normalizer is not None:
                    obs_history = self.obs_normalizer(obs_history)
                return self.original_actor(obs_history)

        class ObsAdapter(torch.nn.Module):
            is_recurrent = False
            def __init__(self, original_policy, obs_normalizer=None, single_frame_dim=None, history_length=None):
                super().__init__()
                self.actor = ObsAdapterActor(original_policy, obs_normalizer, single_frame_dim, history_length)
            def forward(self, obs_single_frame):
                return self.actor(obs_single_frame)

        adapted_policy = ObsAdapter(
            runner.alg.policy.actor,
            runner.obs_normalizer if runner.cfg.get("empirical_normalization", False) else None,
            single_frame_dim=single_frame_dim,
            history_length=history_length,
        )
        adapted_policy.eval()
        test_obs_single = torch.zeros(1, single_frame_dim, device=agent_cfg.device)
        with torch.inference_mode():
            _ = adapted_policy(test_obs_single)
        export_policy_as_jit(adapted_policy, None, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(adapted_policy, normalizer=None, path=export_model_dir, filename="policy.onnx")
        print(f"[INFO] Exported single-frame policy: input {single_frame_dim}. Deploy with frameStack=1")
        policy = runner.get_inference_policy(device=env.device)

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard

        keyboard = Keyboard(env)  # noqa:F841

    # 预热观测历史 buffer：训练时 policy 输入为 actor_obs_history_length 帧拼接，
    # 首步前只调一次 get_observations() 时 buffer 仅 1 帧、其余为 0，会导致策略乱动/秒倒
    history_len = getattr(env.cfg.robot, "actor_obs_history_length", 10)
    for _ in range(history_len - 1):
        env.get_observations()
    obs, _ = env.get_observations()

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
    finally:
        # 在关闭 Omniverse 前显式释放环境和 runner，避免 SimStage/plugin 未释放导致的退出警告
        if "runner" in dir() and runner is not None:
            del runner
        if "env" in dir() and env is not None:
            if hasattr(env, "clear"):
                env.clear()
            del env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        simulation_app.close()


if __name__ == "__main__":
    play()
