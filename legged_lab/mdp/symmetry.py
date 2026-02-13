"""
Symmetry data augmentation functions for Roban S14 bipedal robot.

Implements left-right mirror flipping for observations, actions, and AMP observations.
Used by the PPO/AMP training algorithm to enforce symmetric gaits.

Reference: Mittal et al., 2024 - "Symmetry Considerations for Learning Task Symmetric Robot Policies"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from legged_lab.envs.roban.roban_envs import RobanEnv

# ──────────────────────────────────────────────────────────────────────────────
# Sign maps for joint flipping (determined by default joint positions)
#
# Rule: if default_left ≈ -default_right  → sign = -1  (e.g., hip_roll L2=-0.0437, R2=+0.0437)
#       if default_left ≈  default_right  → sign = +1  (e.g., knee    L4=+0.5,    R4=+0.5)
#       ankle_roll always -1 (left/right roll directions are mirrored)
#
# Roban S14 leg joints (L1..L6): hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
LEG_FLIP_SIGNS = torch.tensor([-1.0, -1.0, -1.0, +1.0, +1.0, -1.0])

# Roban S14 arm joints (A1..A4): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
ARM_FLIP_SIGNS = torch.tensor([+1.0, -1.0, -1.0, +1.0])
# ──────────────────────────────────────────────────────────────────────────────


def _flip_joint_data(data: torch.Tensor, env: RobanEnv) -> torch.Tensor:
    """Flip 21-dim joint data (pos, vel, or action) by swapping left↔right with sign correction.

    Uses env.left_leg_ids, env.right_leg_ids, etc. so it's independent of USD joint ordering.

    Args:
        data: [batch, 21] joint data
        env: RobanEnv instance (provides joint index mappings)

    Returns:
        Flipped [batch, 21] tensor
    """
    flipped = torch.zeros_like(data)
    leg_sign = LEG_FLIP_SIGNS.to(data.device)
    arm_sign = ARM_FLIP_SIGNS.to(data.device)

    # Waist yaw: negate (left turn ↔ right turn)
    flipped[:, env.waist_ids] = -data[:, env.waist_ids]

    # Left leg ← Right leg (with sign)
    for i in range(6):
        flipped[:, env.left_leg_ids[i]] = leg_sign[i] * data[:, env.right_leg_ids[i]]
        flipped[:, env.right_leg_ids[i]] = leg_sign[i] * data[:, env.left_leg_ids[i]]

    # Left arm ← Right arm (with sign)
    for i in range(4):
        flipped[:, env.left_arm_ids[i]] = arm_sign[i] * data[:, env.right_arm_ids[i]]
        flipped[:, env.right_arm_ids[i]] = arm_sign[i] * data[:, env.left_arm_ids[i]]

    return flipped


def _flip_single_actor_obs(obs_frame: torch.Tensor, env: RobanEnv) -> torch.Tensor:
    """Flip a single frame of actor observation.

    Actor obs layout (per frame):
        [0:3]   ang_vel             → [-ωx, ωy, -ωz]
        [3:6]   projected_gravity   → [gx, -gy, gz]
        [6:9]   command             → [vx, -vy, -ωz]
        [9:30]  joint_pos (21)      → flip_joint_data
        [30:51] joint_vel (21)      → flip_joint_data
        [51:72] action (21)         → flip_joint_data
        [72:74] gait_sin (2)        → swap [L, R]
        [74:76] gait_cos (2)        → swap [L, R]
        [76:78] phase_ratio (2)     → swap [L, R]

    Args:
        obs_frame: [batch, obs_dim] single frame observation (72 or 78 dims)
        env: RobanEnv instance

    Returns:
        Flipped observation tensor
    """
    flipped = torch.zeros_like(obs_frame)
    obs_dim = obs_frame.shape[-1]

    # ang_vel: [-ωx, ωy, -ωz]  (roll and yaw are antisymmetric about sagittal plane)
    flipped[:, 0] = -obs_frame[:, 0]
    flipped[:, 1] = obs_frame[:, 1]
    flipped[:, 2] = -obs_frame[:, 2]

    # projected_gravity: [gx, -gy, gz]  (y is lateral direction)
    flipped[:, 3] = obs_frame[:, 3]
    flipped[:, 4] = -obs_frame[:, 4]
    flipped[:, 5] = obs_frame[:, 5]

    # command: [vx, -vy, -ωz]  (lateral velocity and yaw rate flip)
    flipped[:, 6] = obs_frame[:, 6]
    flipped[:, 7] = -obs_frame[:, 7]
    flipped[:, 8] = -obs_frame[:, 8]

    # joint_pos (21), joint_vel (21), action (21)
    flipped[:, 9:30] = _flip_joint_data(obs_frame[:, 9:30], env)
    flipped[:, 30:51] = _flip_joint_data(obs_frame[:, 30:51], env)
    flipped[:, 51:72] = _flip_joint_data(obs_frame[:, 51:72], env)

    # Gait phase signals (if present): swap left ↔ right
    if obs_dim >= 78:
        flipped[:, 72] = obs_frame[:, 73]   # gait_sin: L ← R
        flipped[:, 73] = obs_frame[:, 72]   # gait_sin: R ← L
        flipped[:, 74] = obs_frame[:, 75]   # gait_cos: L ← R
        flipped[:, 75] = obs_frame[:, 74]   # gait_cos: R ← L
        flipped[:, 76] = obs_frame[:, 77]   # phase_ratio: L ← R
        flipped[:, 77] = obs_frame[:, 76]   # phase_ratio: R ← L

    return flipped


def _flip_single_critic_obs(obs_frame: torch.Tensor, env: RobanEnv) -> torch.Tensor:
    """Flip a single frame of critic observation.

    Critic obs = actor_obs + [root_lin_vel(3), feet_contact(2)]

    Extra fields:
        [N:N+3]  root_lin_vel  → [vx, -vy, vz]
        [N+3:N+5] feet_contact → swap [L, R]

    where N = actor_obs_dim (72 or 78)
    """
    flipped = torch.zeros_like(obs_frame)
    # Determine actor obs size from total critic obs size
    # critic = actor + lin_vel(3) + feet_contact(2)
    actor_dim = obs_frame.shape[-1] - 5

    # Flip actor part
    flipped[:, :actor_dim] = _flip_single_actor_obs(obs_frame[:, :actor_dim], env)

    # root_lin_vel: [vx, -vy, vz]
    flipped[:, actor_dim] = obs_frame[:, actor_dim]
    flipped[:, actor_dim + 1] = -obs_frame[:, actor_dim + 1]
    flipped[:, actor_dim + 2] = obs_frame[:, actor_dim + 2]

    # feet_contact: swap [L, R]
    flipped[:, actor_dim + 3] = obs_frame[:, actor_dim + 4]
    flipped[:, actor_dim + 4] = obs_frame[:, actor_dim + 3]

    return flipped


def flip_actor_obs(obs: torch.Tensor, env: RobanEnv) -> torch.Tensor:
    """Flip actor observation with history stacking.

    Actor obs is [batch, history_len * single_frame_dim].
    We reshape, flip each frame, and reshape back.
    """
    history_len = env.cfg.robot.actor_obs_history_length
    single_dim = obs.shape[-1] // history_len
    batch = obs.shape[0]

    obs_reshaped = obs.view(batch, history_len, single_dim)
    flipped_frames = torch.zeros_like(obs_reshaped)
    for t in range(history_len):
        flipped_frames[:, t] = _flip_single_actor_obs(obs_reshaped[:, t], env)

    return flipped_frames.view(batch, -1)


def flip_critic_obs(obs: torch.Tensor, env: RobanEnv) -> torch.Tensor:
    """Flip critic observation with history stacking."""
    history_len = env.cfg.robot.critic_obs_history_length
    single_dim = obs.shape[-1] // history_len
    batch = obs.shape[0]

    obs_reshaped = obs.view(batch, history_len, single_dim)
    flipped_frames = torch.zeros_like(obs_reshaped)
    for t in range(history_len):
        flipped_frames[:, t] = _flip_single_critic_obs(obs_reshaped[:, t], env)

    return flipped_frames.view(batch, -1)


def flip_action(actions: torch.Tensor, env: RobanEnv) -> torch.Tensor:
    """Flip 21-dim action tensor (same as joint data flip)."""
    return _flip_joint_data(actions, env)


def flip_amp_obs(amp_obs: torch.Tensor, env: RobanEnv) -> torch.Tensor:
    """Flip 67-dim AMP observation.

    AMP obs layout (explicit order from get_amp_obs_for_expert_trans):
        [0:21]  joint_pos: [waist(1), left_leg(6), right_leg(6), left_arm(4), right_arm(4)]
        [21:42] joint_vel: same ordering
        [42:43] root_height: keep
        [43:49] root_rotation (tangent+normal): flip y components
        [49:52] root_lin_vel: [vx, -vy, vz]
        [52:55] root_ang_vel: [-ωx, ωy, -ωz]
        [55:58] left_hand_pos → right_hand_pos, y negate
        [58:61] right_hand_pos → left_hand_pos, y negate
        [61:64] left_foot_pos → right_foot_pos, y negate
        [64:67] right_foot_pos → left_foot_pos, y negate

    NOTE: AMP obs uses explicit [waist, left_leg, right_leg, left_arm, right_arm] order,
    NOT the USD joint ordering. So we use hardcoded indices here.
    """
    flipped = torch.zeros_like(amp_obs)
    leg_sign = LEG_FLIP_SIGNS.to(amp_obs.device)
    arm_sign = ARM_FLIP_SIGNS.to(amp_obs.device)

    # --- joint_pos (0:21) and joint_vel (21:42) ---
    for base in [0, 21]:
        # waist (index 0): negate
        flipped[:, base + 0] = -amp_obs[:, base + 0]

        # left_leg (1:7) ← right_leg (7:13) with signs
        for i in range(6):
            flipped[:, base + 1 + i] = leg_sign[i] * amp_obs[:, base + 7 + i]
            flipped[:, base + 7 + i] = leg_sign[i] * amp_obs[:, base + 1 + i]

        # left_arm (13:17) ← right_arm (17:21) with signs
        for i in range(4):
            flipped[:, base + 13 + i] = arm_sign[i] * amp_obs[:, base + 17 + i]
            flipped[:, base + 17 + i] = arm_sign[i] * amp_obs[:, base + 13 + i]

    # --- root_height (42): keep ---
    flipped[:, 42] = amp_obs[:, 42]

    # --- root_rotation tangent (43:46) and normal (46:49): y negate ---
    # tangent: [tx, -ty, tz]
    flipped[:, 43] = amp_obs[:, 43]
    flipped[:, 44] = -amp_obs[:, 44]
    flipped[:, 45] = amp_obs[:, 45]
    # normal: [nx, -ny, nz]
    flipped[:, 46] = amp_obs[:, 46]
    flipped[:, 47] = -amp_obs[:, 47]
    flipped[:, 48] = amp_obs[:, 48]

    # --- root_lin_vel (49:52): [vx, -vy, vz] ---
    flipped[:, 49] = amp_obs[:, 49]
    flipped[:, 50] = -amp_obs[:, 50]
    flipped[:, 51] = amp_obs[:, 51]

    # --- root_ang_vel (52:55): [-ωx, ωy, -ωz] ---
    flipped[:, 52] = -amp_obs[:, 52]
    flipped[:, 53] = amp_obs[:, 53]
    flipped[:, 54] = -amp_obs[:, 54]

    # --- end_effector positions: swap L↔R, negate y ---
    # left_hand (55:58) ← right_hand (58:61)
    flipped[:, 55] = amp_obs[:, 58]       # x
    flipped[:, 56] = -amp_obs[:, 59]      # -y
    flipped[:, 57] = amp_obs[:, 60]       # z
    # right_hand (58:61) ← left_hand (55:58)
    flipped[:, 58] = amp_obs[:, 55]
    flipped[:, 59] = -amp_obs[:, 56]
    flipped[:, 60] = amp_obs[:, 57]
    # left_foot (61:64) ← right_foot (64:67)
    flipped[:, 61] = amp_obs[:, 64]
    flipped[:, 62] = -amp_obs[:, 65]
    flipped[:, 63] = amp_obs[:, 66]
    # right_foot (64:67) ← left_foot (61:64)
    flipped[:, 64] = amp_obs[:, 61]
    flipped[:, 65] = -amp_obs[:, 62]
    flipped[:, 66] = amp_obs[:, 63]

    return flipped


# ══════════════════════════════════════════════════════════════════════════════
# Main data augmentation function (called by PPO/AMPPPO)
# ══════════════════════════════════════════════════════════════════════════════

def roban_symmetry_augmentation(
    obs: torch.Tensor | None,
    actions: torch.Tensor | None,
    env: RobanEnv,
    obs_type: str = "policy",
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Symmetry data augmentation function for Roban S14.

    Generates mirrored copies and concatenates with originals:
        output = cat([original, mirrored], dim=0)

    This is called by AMPPPO.update() during training.

    Args:
        obs: [batch, obs_dim] observation tensor, or None
        actions: [batch, action_dim] action tensor, or None
        env: RobanEnv instance
        obs_type: "policy" for actor obs, "critic" for critic obs

    Returns:
        (augmented_obs, augmented_actions) - each is [2*batch, dim] or None
    """
    if obs is not None:
        if obs_type == "policy":
            mirrored_obs = flip_actor_obs(obs, env)
        elif obs_type == "critic":
            mirrored_obs = flip_critic_obs(obs, env)
        else:
            raise ValueError(f"Unknown obs_type: {obs_type}")
        obs = torch.cat([obs, mirrored_obs], dim=0)

    if actions is not None:
        mirrored_actions = flip_action(actions, env)
        actions = torch.cat([actions, mirrored_actions], dim=0)

    return obs, actions
