# Level 0 (Ignition) 配置与奖励说明文档

> 当前代码快照导出，包含：速度课程配置、升级条件、**奖励函数计算公式**与 **Level 0 有效权重**。

---

## 1. 速度课程 — Level 0 配置

### 1.1 级别定义

| 项 | 值 |
|----|-----|
| **级别名称** | Ignition（点火阶段） |
| **设计目标** | 只前进，利用动量建立稳定步态；治蹭地——站直 + 易触达滞空 + 强罚滑动 |

### 1.2 速度范围（指令采样）

| 维度 | 范围 | 单位 |
|------|------|------|
| `lin_vel_x` | **(0.4, 0.5)** | m/s |
| `lin_vel_y` | **(0.0, 0.0)** | m/s |
| `ang_vel_z` | **(-0.8, 0.8)** | rad/s |

### 1.3 升级条件（Level 0 → Level 1）

同时满足：

- 平均 episode 时长 > **0.75 × max_episode_length_s**（例如 20s 时为 > 15s）
- 平均线速度跟踪误差 < **0.3**（`lin_vel_err_limit`）
- 角速度误差不检查

### 1.4 统一死区

| 参数 | 值 | 说明 |
|------|-----|------|
| `lin_vel_deadband` | 0.2 m/s | \|\|cmd_xy\|\| > 0.2 才视为有指令 |
| `ang_vel_deadband` | 0.2 rad/s | \|cmd_yaw\| > 0.2 才视为有指令 |

指令在死区内时：`is_command_active = False`，步态相关奖励门控归零，`stand_still_penalty` 生效。

### 1.5 其他课程参数

| 参数 | 值 |
|------|-----|
| promote_threshold | 0.75 |
| demote_threshold | 0.35 |
| buffer_size | 2000 |
| cooldown_steps | 500 |

---

## 2. Level 0 奖励权重乘数（reward_multipliers[0]）

未在下表列出的项，Level 0 乘数均为 **1.0**。

| 奖励项 | Level 0 乘数 | 说明 |
|--------|--------------|------|
| track_lin_vel_xy_exp | 1.5 | 聚焦前进 |
| flat_orientation_exp | 1.0 | 保持身体水平 |
| base_height_exp | **2.0** | 强制站直，抬 CoM |
| feet_air_time_easy | **1.0** | 0.1s 滞空即奖励 |
| feet_air_time | **0.0** | Level 0 关闭硬阈值 |
| feet_slide | **10.0** | 蹭地重罚 |
| feet_height | **1.0** | 鼓励任意抬脚 |
| step_frequency | 3.0 | 促步频 |
| straight_knee_landing | 0.3 | 弱化落地膝盖 |
| soft_landing | 0.3 | 弱化轻柔落地 |
| feet_distance | 1.0 | 不变 |
| feet_contact_time_symmetry | 1.0 | 不变 |

---

## 3. 奖励函数计算方式与参数

以下为各奖励项的**计算公式**、**参数**及 **Level 0 有效权重**（effective = base_weight × Level0_multiplier）。  
公式中 `env` 为环境实例，`asset` 为机器人，`cmd` = `env.command_generator.command`（前 2 维为 xy 线速度指令，第 3 维为 yaw 角速度指令）。

---

### 3.1 速度与姿态（Objectives）

| 奖励项 | 函数 | 计算公式 | 参数 | Base 权重 | L0 乘数 | **有效权重** |
|--------|------|----------|------|-----------|---------|----------------|
| track_lin_vel_xy_exp | track_lin_vel_xy_yaw_frame_exp | \( r = \exp\bigl(-\|\|cmd_{xy} - v_{yaw,xy}\|\|^2 / \sigma^2\bigr) \)；\(v_{yaw}\) 为根速度在 yaw 系下的投影 | std=0.5 | 2.0 | 1.5 | **3.0** |
| track_ang_vel_z_exp | track_ang_vel_z_world_exp | \( r = \exp\bigl(-(cmd_z - \omega_z)^2 / \sigma^2\bigr) \) | std=0.5 | 1.5 | 1.0 | **1.5** |
| flat_orientation_exp | flat_orientation_exp | \( r = \exp\bigl(-\bigl(\sum_{i\in\{x,y\}} g_i^2\bigr)^2 / \sigma^2\bigr) \)；\(g\) 为 projected_gravity_b 前两维 | std=0.25 | 1.0 | 1.0 | **1.0** |
| base_height_exp | base_height_exp | \( r = \exp\bigl(-(z_{root} - z_{target})^2 / \sigma^2\bigr) \) | target_height=0.65, std=0.1 | 1.0 | 2.0 | **2.0** |
| lin_vel_z_exp | lin_vel_z_exp | \( r = \exp(-v_{z}^2 / \sigma) \)；根线速度 z 分量 | std=0.5 | 1.0 | 1.0 | 1.0 |
| ang_vel_xy_exp | ang_vel_xy_exp | \( r = \exp(-\|\|\omega_{xy}\|\|^2 / \sigma) \)；根角速度 xy | std=0.5 | 0.5 | 1.0 | 0.5 |

---

### 3.2 步态相关（滞空 / 抬脚 / 步频）

| 奖励项 | 函数 | 计算公式 / 逻辑 | 参数 | Base 权重 | L0 乘数 | **有效权重** |
|--------|------|-----------------|------|-----------|---------|----------------|
| feet_air_time_easy | feet_air_time_clip_biped | 仅在 **first_contact** 给奖。\( air = \max(0,\, last\_air\_time - t_{min}) \) 再 clip 到 \([0,\, t_{max}-t_{min}]\)，按脚求和；无指令时归零 | threshold_min=**0.1**, threshold_max=0.25 | 20.0 | 1.0 | **20.0** |
| feet_air_time | feet_air_time_clip_biped | 同上，阈值更严 | threshold_min=0.25, threshold_max=0.4 | 20.0 | **0.0** | **0**（关闭） |
| feet_height | feet_height_cycle | 每脚在**本摆动周期**内最大离地高度，仅在 **first_contact** 输出，再 clip 到 max_height_clip，双脚求和 | max_height_clip=0.15 | 12.0 | 1.0 | **12.0** |
| step_frequency | step_frequency_penalty | 有指令且**双脚同时着地**时间 > max_grounded_time 时：\( r = -\max(0,\, t_{both} - t_{max}) \times scale \)；否则 0 | max_grounded_time=0.2, penalty_scale=2.0, velocity_threshold=0.12 | 10.0 | 3.0 | **30.0** |
| straight_knee_landing | contact_ground_straight_knee | 仅在 **first_contact**：\( r = \sum \exp(-q_{knee}^2/\sigma^2) \times first\_contact \)；有指令时生效 | std=0.3 | 15.0 | 0.3 | **4.5** |
| soft_landing | contact_momentum | 仅在 **first_contact**：\( r = \sum \exp(-(v_z/clamp)^2/\sigma^2) \times first\_contact \)，\(v_z\) 脚部垂直速度 clamp 到 [-10,0] | std=0.05 | 1.0 | 0.3 | **0.3** |

---

### 3.3 约束与惩罚（Constraints）

| 奖励项 | 函数 | 计算公式 | 参数 | Base 权重 | L0 乘数 | **有效权重** |
|--------|------|----------|------|-----------|---------|----------------|
| feet_slide | feet_slide | \( r = \sum_{feet} \|\|v_{feet,xy}\|\| \cdot \mathbb{1}_{contact} \)；接触时脚部水平速度（惩罚滑动） | body: leg_l6_link, leg_r6_link | -0.5 | 10.0 | **-5.0** |
| feet_distance | feet_distance_penalty | \( d = \|\|p_{L} - p_{R}\|\| \)；\( r = (min\_dist - d)_+^2 + (d - max\_dist)_+^2 \) | min_dist=0.22, max_dist=0.30 | -30.0 | 1.0 | **-30.0** |
| feet_contact_time_symmetry | feet_contact_time_symmetry_exp | 左右脚最近一次支撑相时长 \(T_L,\,T_R\)；\( r = \exp(-(T_L - T_R)^2 / \sigma^2) \)，需两脚都完成过支撑且指令非零 | sigma=0.25 | 1.6 | 1.0 | **1.6** |
| termination_penalty | is_terminated | \( r = 1 \) 当本步发生非 timeout 终止（如摔倒） | — | -200.0 | 1.0 | **-200.0** |
| undesired_contacts | undesired_contacts | 指定 body 接触力 > threshold 则计 1，按 body 求和 | threshold=1.0, body_names=腿/臂/base | -1.0 | 1.0 | -1.0 |
| feet_force | body_force | 脚部法向力超过 threshold 部分，clip 后按脚取范数 | threshold=500, max_reward=400 | -0.01 | 1.0 | -0.01 |
| stand_still_penalty | stand_still_penalty | 无指令时：\( r = \max(0,\, \|\|joint\_vel\|\| - thresh) \) | joint_vel_threshold=0.1 | -2.0 | 1.0 | -2.0 |
| energy | energy | \( r = \|\|torque \odot joint\_vel\|\| \) | — | -1e-5 | 1.0 | -1e-5 |
| dof_acc_l2 | joint_acc_l2 | 指定关节加速度 L2 | — | -1e-7 | 1.0 | -1e-7 |
| action_rate_l2 | action_rate_l2 | \( r = \|\|a_t - a_{t-1}\|\|^2 \) | — | -0.01 | 1.0 | -0.01 |
| ang_acc_xy_l2 | ang_acc_xy_l2 | 躯干 roll/pitch 角加速度 L2 | — | -0.00005 | 1.0 | -0.00005 |
| action_smoothness_l2 | action_smoothness_l2 | 二阶动作差分 L2 | — | -0.01 | 1.0 | -0.01 |

---

## 4. 符号与约定

- **first_contact**：本步刚着地的脚（由 contact_sensor 的 compute_first_contact 得到）。
- **last_air_time**：该脚上一段完整摆动相（离地）的时长。
- **有指令**：\( \|\|cmd_{xy}\|\| + |cmd_z| > 0.1 \)（或各奖励内 0.1/velocity_threshold）。
- **有效权重**：`effective_weight = base_weight * reward_multipliers[0].get(name, 1.0)`；最终每步该奖励贡献为 `effective_weight * r`，其中 `r` 为上表公式得到的标量（或向量求和后标量）。

---

## 5. 关键参数速查（Level 0）

| 类别 | 参数 | 值 |
|------|------|-----|
| 速度范围 | lin_vel_x | (0.4, 0.5) m/s |
| | lin_vel_y | (0.0, 0.0) |
| | ang_vel_z | (-0.8, 0.8) rad/s |
| 升级 | promote_threshold | 0.75 |
| | L0→L1 lin_vel_err_limit | 0.3 |
| 死区 | lin_vel_deadband | 0.2 m/s |
| | ang_vel_deadband | 0.2 rad/s |
| 基座高度 | base_height target / std | 0.65 m / 0.1 |
| 滞空 easy | feet_air_time_easy threshold_min/max | 0.1 s / 0.25 s |
| 滞空 hard | feet_air_time（L0 关闭） | 0.25 s / 0.4 s |
| 步频 | step_frequency max_grounded_time | 0.2 s |
| | step_frequency velocity_threshold | 0.12 |
| 抬脚高度 | feet_height max_height_clip | 0.15 m |
| 两脚距离 | feet_distance min/max | 0.22 m / 0.30 m |

---

## 6. 代码位置

- 速度课程与 Level 0 定义：`legged_lab/envs/roban/walk_cfg.py` — `VelocityCurriculumCfg`（levels、promote_criteria、reward_multipliers）。
- 奖励项与 base 权重：`legged_lab/envs/roban/walk_cfg.py` — `LiteRewardCfg`、`RobanLiteRewardCfg`。
- 奖励函数实现：`legged_lab/mdp/rewards.py`（各函数名与上表一致）。
- Level 0 有效权重应用：`legged_lab/envs/roban/roban_envs.py` — `_apply_reward_scales(level)`。
