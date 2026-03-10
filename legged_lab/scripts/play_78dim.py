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
# play_78dim.py：用于「78 维观测（72 + 6 步态）」训练的 checkpoint 的播放与导出。
# 强制 env 使用 include_gait_in_obs=True，单帧 78 维；历史长度由 task 配置决定（1 帧→78 维，10 帧→780 维）。

import argparse
import os

import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Play/export checkpoints trained with 78-dim observation (72 + 6 gait). Forces include_gait_in_obs=True; history from task config (1 frame=78, 10 frames=780)."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--export_full_obs",
    action="store_true",
    help="Export ONNX with full observation (e.g. 10 frames, 780 dim). Deploy with frameStack=10, numSingleObs=78.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Start camera rendering
if args_cli.task and "sensor" in args_cli.task:
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
    print(f"[INFO] play_78dim: Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # Play 环境设置
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.ranges.lin_vel_x = (0.6, -0.6)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.6, 0.6)
    if hasattr(env_cfg, "velocity_curriculum") and env_cfg.velocity_curriculum is not None:
        env_cfg.velocity_curriculum.enable = False
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

    # 强制 78 维单帧（72 + 6 步态），与先前「带步态」训练一致
    env_cfg.robot.include_gait_in_obs = True
    print("[INFO] play_78dim: Forced include_gait_in_obs=True (78 dim per frame: 72 + 6 gait)")

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    runner_class: OnPolicyRunner | AmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if isinstance(runner, AmpOnPolicyRunner):
        runner.load(resume_path, load_optimizer=False, skip_discriminator_on_mismatch=True)
    else:
        runner.load(resume_path, load_optimizer=False)

    # 导出模型（逻辑与 play.py 一致：1 帧直接导出，多帧可选 full_obs 或 ObsAdapter）
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    test_obs, _ = env.get_observations()
    actual_obs_dim = test_obs.shape[1]
    history_length = env.cfg.robot.actor_obs_history_length
    single_frame_dim = actual_obs_dim // history_length

    print(f"[INFO] play_78dim observation: total={actual_obs_dim}, history={history_length}, single_frame={single_frame_dim} (78 with gait)")
    print(f"  - export_full_obs: {getattr(args_cli, 'export_full_obs', False)}")

    export_full_obs = getattr(args_cli, "export_full_obs", False)

    if export_full_obs:
        policy_to_export = runner.alg.policy.actor
        policy_to_export.eval()
        test_input = torch.zeros(1, actual_obs_dim, device=agent_cfg.device)
        with torch.inference_mode():
            _ = policy_to_export(test_input)
        export_policy_as_jit(policy_to_export, None, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_to_export, normalizer=None, path=export_model_dir, filename="policy.onnx")
        print(f"[INFO] play_78dim exported full-obs: input {actual_obs_dim}. Deploy with frameStack={history_length}, numSingleObs={single_frame_dim}")
        policy = runner.get_inference_policy(device=env.device)
    elif history_length == 1:
        policy_to_export = runner.alg.policy.actor
        policy_to_export.eval()
        test_input = torch.zeros(1, actual_obs_dim, device=agent_cfg.device)
        with torch.inference_mode():
            _ = policy_to_export(test_input)
        export_policy_as_jit(policy_to_export, None, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_to_export, normalizer=None, path=export_model_dir, filename="policy.onnx")
        print(f"[INFO] play_78dim exported 1-frame: input {actual_obs_dim} (78 with gait). Deploy with frameStack=1, numSingleObs=78")
        policy = runner.get_inference_policy(device=env.device)
    else:
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
        print(f"[INFO] play_78dim exported single-frame: input {single_frame_dim} (78 with gait). Deploy with frameStack=1")
        policy = runner.get_inference_policy(device=env.device)

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard

        keyboard = Keyboard(env)  # noqa:F841

    history_len = getattr(env.cfg.robot, "actor_obs_history_length", 1)
    for _ in range(history_len - 1):
        env.get_observations()
    obs, _ = env.get_observations()

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
    finally:
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
