## PB_AMP Roban AMP 速度跟踪对齐 amp_share 变更记录（REBUILD）

本文件记录本次在 PB_AMP 中，为了让 **Roban 速度跟踪 AMP 任务** 的训练行为尽量贴近 `amp_roban_share`（下简称 **amp_share**）而做的代码改动，方便后续复现、回溯和继续调整。

---

## 1. 环境与任务配置（`legged_lab/envs/roban/walk_cfg.py`）

- **速度命令（Command）对齐**
  - 使用 `CommandsCfg`，与 amp_share 的 `base_velocity` 配置一致：
    - `resampling_time_range=(0.5, 5.0)`
    - `rel_standing_envs=0.5`
    - `rel_heading_envs=0.8`
    - `heading_command=True`，`heading_control_stiffness=0.3`
    - `ranges.lin_vel_x=(-0.6, 0.6), lin_vel_y=(-0.6, 0.6), ang_vel_z=(-1.0, 1.0), heading=(-π, π)`

- **速度课程（Velocity Curriculum）处理**
  - 原先：`velocity_curriculum: VelocityCurriculumCfg = VelocityCurriculumCfg(enable=False)`。
  - 现在：`velocity_curriculum: VelocityCurriculumCfg | None = None`。
  - 含义：在 `RobanEnv` 中不再初始化 / 使用速度课程逻辑，只保留 **地形 + 推力课程**，与 amp_share 的 Roban AMP 速度任务一致。

- **地形 + 推力课程（Terrain + Force Curriculum）保持对齐**
  - 配置：`TerrainForceCurriculumCfg`（`terrain_force_curriculum` 字段）：
    - `force_level_low=0.2`，`force_level_step=0.05`
    - `stage_two_force_threshold=0.9`
    - `stage_2_reward_weights` 中包含：
      - `track_lin_vel_xy_exp=4.5`
      - `track_ang_vel_z_exp=2.25`
      - `action_rate_l2=-0.01`
      - `dof_acc_l2=-1e-5`
      - `feet_contact_time_symmetry=1.0`
      - `feet_distance_symmetry_per_cycle=1.0`
  - 在 `RobanEnv._init_terrain_force_curriculum()` / `_update_terrain_force_curriculum()`：
    - 使用 episode 内 `track_lin_vel_xy_exp` / `track_ang_vel_z_exp` 的累积总和来：
      - 决定地形难度上/下降（terrain_levels_vel 同款逻辑）。
      - 决定外力等级上/下降（external_force_torque_levels 同款逻辑），更新 `_external_force_torque_level`。
    - 当 `mean(_external_force_torque_level) >= 0.9` 时，执行一次性的 **stage_two 权重重写**，并将 `_external_force_torque_level` 重置为 `force_level_low`。
  - 整体逻辑与 amp_share 的 `CurriculumCfg.terrain_levels / external_force_torque_levels / stage_two` 等价。

- **观测噪声（Noise）微调**
  - 修改 `NoiseScalesCfg` 中 roban 任务的关节位置噪声：
    - `joint_pos: 0.01 → 0.05`。
  - 目的：与 amp_share Policy 观测里对 `joint_pos` 使用的 Unoise 大致量级对齐，保持训练抖动程度相近。

---

## 2. 域随机化与事件（Domain Randomization & Events）

### 2.1 Startup 域随机化（`domain_rand.events`）

针对 `RobanWalkFlatEnvCfg.domain_rand.events`，做了如下对齐：

- **物理材质随机化（physics_material）**
  - 原先（PB_AMP）：摩擦和恢复系数范围较窄：
    - `static_friction_range=(0.6, 1.0)`
    - `dynamic_friction_range=(0.4, 0.8)`
    - `restitution_range=(0.0, 0.005)`
  - 现在：改为与 amp_share 相同量级的更宽范围：
    - `static_friction_range=(0.3, 1.0)`
    - `dynamic_friction_range=(0.2, 1.0)`
    - `restitution_range=(0.0, 0.5)`

- **质量随机化**
  - 删除：原先仅对 `base_link` 做 `add_base_mass`（-5 ~ 5kg）的事件。
  - 新增：`scale_link_mass` 事件，与 amp_share 一致：
    - `asset_cfg=SceneEntityCfg("robot", body_names=["leg_.*_link", "zarm_.*_link"])`
    - `mass_distribution_params=(0.8, 1.2)`，`operation="scale"`。

- **质心 / 执行器 / 关节参数随机化**
  - 新增以下 startup 事件（与 amp_share 结构一致）：
    - `randomize_rigid_body_com`：
      - `asset_cfg("robot", body_names="base_link")`
      - `com_range` 在 x/y/z 各 ±0.05。
    - `scale_actuator_gains`：
      - `asset_cfg("robot", joint_names=".*_joint")`
      - `stiffness_distribution_params=(0.8, 1.2)`
      - `damping_distribution_params=(0.8, 1.2)`
      - `operation="scale"`。
    - `scale_joint_parameters`：
      - `asset_cfg("robot", joint_names=".*_joint")`
      - `friction_distribution_params=(1.0, 1.0)`
      - `armature_distribution_params=(0.5, 1.5)`
      - `operation="scale"`。

> 效果：roban 的 startup 域随机化现在与 amp_share 在“材质、link 质量、base COM、执行器增益、关节参数”五个维度上结构一致，仅参数范围做了直接对齐或轻微裁剪。

### 2.2 Reset 事件

- **根部状态重置（reset_base）**
  - 旧配置：x,y ∈ [-0.5, 0.5]，yaw ∈ [-π, π]，无 pitch/roll 项；速度范围 ±0.5。
  - 新配置：参考 amp_share 调整：
    - `x, y: (-0.7, 0.7)`
    - `yaw: (-3.14, 3.14)`
    - 新增 `pitch: (-0.1, 0.1)`，`roll: (-0.1, 0.1)`
    - 速度 `x,y,z,roll,pitch,yaw` 均在 `(-0.3, 0.3)` 内。

- **关节重置（reset_robot_joints）**
  - 维持与 amp_share 一致：
    - `position_range=(0.5, 1.5)`
    - `velocity_range=(0.0, 0.0)`。

### 2.3 Interval 推力事件

- **函数选择**
  - 原先 roban 使用 `mdp.push_by_setting_velocity_curriculum`，依赖 `_external_force_torque_level` 做“课程缩放推力”。
  - 现在改为与基础环境一致、且更简洁的：
    - `func=mdp.push_by_setting_velocity`。
  - 说明：
    - 在地形+推力课程中，**外力等级** 已通过 `_external_force_torque_level` 管理；这里采用“速度脉冲”形式，与 PB_AMP 其他任务保持一致。
    - 若后续需要完全复制 amp_share 的 `apply_external_force_torque_stochastic`（施加力/力矩），可单独扩展。

- **触发频率**
  - 原先：`interval_range_s=(10.0, 15.0)`（低频推力）。
  - 现在：`interval_range_s=(0.01, 0.3)`，与 amp_share 的外力扰动时间尺度相近（高频、弱扰动风格）。

- **扰动幅度**
  - 保持 `velocity_range={"x": (-1.0, 1.0), "y": (-1.0, 1.0)}` 不变，形成横向/纵向的小速度脉冲。

---

## 3. AMP 训练 Runner 与日志输出（`rsl_rl/rsl_rl/runners/amp_on_policy_runner.py`）

### 3.1 训练逻辑保持与 amp_share 对齐的要点

此部分之前已在 `docs/PB_AMP_roban_velocity_training_flow.md` 中详述，这里仅列对齐要点：

- **Rollout 结构**：每迭代 `num_steps_per_env=24` 步，每步：
  - `act → env.step(actions)`（内部 decimation=4、4 个仿真步同一 action）。
  - 存储 transition（obs、rewards、dones、AMP obs）。
- **AMP 奖励组合**：
  - rollout 期间只存 task reward 与 AMP 状态；
  - 在 `compute_returns()` 内批量计算 style reward，并用：
    - `rewards = task_reward_weight * task + style_reward_weight * style`
  - 对应 amp_share 的 AMP 奖励组合方式。
- **判别器更新**：
  - 每次 `update()` 使用：
    - 策略侧：当前 rollout (24 × num_envs) 的 AMP 状态对；
    - 专家侧：AMP 数据集（多轨迹 npz），构造 expert batch。

这些逻辑没有在本次 REBUILD 中再改动，只做了补充日志上的精简（见下）。

### 3.2 终端日志输出精简

函数：`AmpOnPolicyRunner.log()`。

- **旧行为**：
  - 打印一大块 ASCII 边框；
  - 输出每个 loss 的详细数值；
  - 分别打印 extrinsic / intrinsic reward；
  - 打印大量时间统计，信息虽然完整但终端噪声很大。

- **新行为（更贴近“常规必要数据 + 课程进度”）**：
  - 每次迭代在终端输出一段简洁文本，例如：

    ```text
    [Iter it/tot] fps=XXXX mean_rew=XX.XX ep_len=YY.YY collect=as learn=bs std=Z.ZZ
      task_rew=... style_rew=... amp_ratio=...%     # 若使用 AMP 奖励组合
      Episode/xxx: ...                              # 来自 env.extras['log'] 的课程/episode 统计
      ...
      total_steps=NNNNNN elapsed=HH:MM:SS ETA=HH:MM:SS
    ```

  - 保留对 **课程相关指标** 的打印：
    - `ep_infos` 中的键，如：
      - `Curriculum/terrain_levels`
      - `Curriculum/external_force_level`
      - `Curriculum/mean_lin_sum`
      - `Curriculum/mean_ang_sum`
      - 以及其它来自环境 `infos["log"]` 的统计。
  - 所有 `writer.add_scalar(...)` 的 TensorBoard / WandB 记录逻辑保持不变，便于事后分析。

> 目标：终端只显示「迭代进度 + 性能概况 + 课程状态」，减少无关细节，便于观察 roban AMP 训练整体趋势。

---

## 4. 文档补充

本次对齐工作还新增/更新了以下文档，记录设计与实现细节：

- `docs/PB_AMP_roban_velocity_training_flow.md`
  - 整理了 PB_AMP 中 roban AMP 速度跟踪训练的完整流程（入口 → runner → rollout → AMP 奖励 → 课程），并与 amp_share 的训练流程做逐项对照；
  - 明确了 decimation、transition 记录方式、AMP 奖励组合、判别器与策略数据来源等。

- `docs/PB_AMP_vs_amp_share_event_command_domain_rand.md`
  - 专门对比 PB_AMP 与 amp_share 在 **Command、Event、Domain Randomization（域随机化）** 三个方面的差异与对齐情况；
  - 说明哪些地方已经完全对齐，哪些地方刻意保留差异（如当前仍未完全复制 amp_share 的完整 RewardsCfg，只对速度跟踪相关权重和课程逻辑做了对齐）。

---

## 5. 尚未完全对齐且“有意保留”的部分（提示）

为避免误解，这里明确列出目前 **尚未完全复制 amp_share**、但我们有意识暂时保留差异的部分：

- **奖励函数整体结构（RewardsCfg vs RobanLiteRewardCfg）**
  - 目前仅对齐了：
    - 速度跟踪核心项：`track_lin_vel_xy_exp` / `track_ang_vel_z_exp` 的权重与 stage_two 权重；
    - 与地形+推力课程配套的少量约束项（如 `action_rate_l2`、`dof_acc_l2` 等）。
  - amp_share 中其他大量 reward（如 `feet_height_cycle`、`turning_step_frequency`、更细的 torque/acc/对称性惩罚等）尚未一一迁移到 PB_AMP。
  - 原因：PB_AMP 当前采用的是一套更简化、偏“Pragmatic Minimalist” 的 gait reward 设计，希望先观察 AMP + 简化 reward 的表现，再视需要逐步引入更复杂的 shaping。

若后续需要 **进一步让 roban 的步态与 amp_share 完全一致**，建议的下一步工作是：

1. 在 `legged_lab/mdp/rewards.py` 中，为 Roban 建一个与 amp_share `RewardsCfg` 结构近似的新 RewardCfg；
2. 在 `RobanWalkFlatEnvCfg` 中使用该 RewardCfg，并按 amp_share 的 `stage_two` 配置逐项对齐权重；
3. 在文档中追加“Rewards 对齐说明”小节。

---

## 6. 总结

本次 REBUILD 的核心目标，是在 **不完全牺牲 PB_AMP 原有设计思想** 的前提下，让 **Roban AMP 速度跟踪任务** 在以下方面尽量与 amp_share 对齐：

- 物理与命令：dt / decimation / 命令范围 / 机体配置；
- AMP 算法：rollout 结构、AMP 奖励组合、判别器与策略更新逻辑；
- 课程：地形 + 推力等级 + stage_two 权重重写；
- 域随机化：材质、质量、COM、执行器、关节参数、推力频率与形式；
- 日志：只在终端展示训练进度及课程状态的“必要信息”。

已对齐部分足以让 **速度跟踪质量与训练行为大体处于同一“分布”**；若未来需要做到“几乎一模一样的步态与细节”，则需要进一步对 RewardsCfg 做更细致的迁移与验证。

## PB_AMP Roban AMP 速度跟踪对齐 amp_share 变更记录（REBUILD）

本文件记录本次在 PB_AMP 中，为了让 **Roban 速度跟踪 AMP 任务** 的训练行为尽量贴近 `amp_roban_share`（下简称 **amp_share**）而做的代码改动，方便后续复现、回溯和继续调整。

---

## 1. 环境与任务配置（`legged_lab/envs/roban/walk_cfg.py`）

- **速度命令（Command）对齐**
  - 使用 `CommandsCfg`，与 amp_share 的 `base_velocity` 配置一致：
    - `resampling_time_range=(0.5, 5.0)`
    - `rel_standing_envs=0.5`
    - `rel_heading_envs=0.8`
    - `heading_command=True`，`heading_control_stiffness=0.3`
    - `ranges.lin_vel_x=(-0.6, 0.6), lin_vel_y=(-0.6, 0.6), ang_vel_z=(-1.0, 1.0), heading=(-π, π)`

- **速度课程（Velocity Curriculum）处理**
  - 原先：`velocity_curriculum: VelocityCurriculumCfg = VelocityCurriculumCfg(enable=False)`。
  - 现在：`velocity_curriculum: VelocityCurriculumCfg | None = None`。
  - 含义：在 `RobanEnv` 中不再初始化 / 使用速度课程逻辑，只保留 **地形 + 推力课程**，与 amp_share 的 Roban AMP 速度任务一致。

- **地形 + 推力课程（Terrain + Force Curriculum）保持对齐**
  - 配置：`TerrainForceCurriculumCfg`（`terrain_force_curriculum` 字段）：
    - `force_level_low=0.2`，`force_level_step=0.05`
    - `stage_two_force_threshold=0.9`
    - `stage_2_reward_weights` 中包含：
      - `track_lin_vel_xy_exp=4.5`
      - `track_ang_vel_z_exp=2.25`
      - `action_rate_l2=-0.01`
      - `dof_acc_l2=-1e-5`
      - `feet_contact_time_symmetry=1.0`
      - `feet_distance_symmetry_per_cycle=1.0`
  - 在 `RobanEnv._init_terrain_force_curriculum()` / `_update_terrain_force_curriculum()`：
    - 使用 episode 内 `track_lin_vel_xy_exp` / `track_ang_vel_z_exp` 的累积总和来：
      - 决定地形难度上/下降（terrain_levels_vel 同款逻辑）。
      - 决定外力等级上/下降（external_force_torque_levels 同款逻辑），更新 `_external_force_torque_level`。
    - 当 `mean(_external_force_torque_level) >= 0.9` 时，执行一次性的 **stage_two 权重重写**，并将 `_external_force_torque_level` 重置为 `force_level_low`。
  - 整体逻辑与 amp_share 的 `CurriculumCfg.terrain_levels / external_force_torque_levels / stage_two` 等价。

- **观测噪声（Noise）微调**
  - 修改 `NoiseScalesCfg` 中 roban 任务的关节位置噪声：
    - `joint_pos: 0.01 → 0.05`。
  - 目的：与 amp_share Policy 观测里对 `joint_pos` 使用的 Unoise 大致量级对齐，保持训练抖动程度相近。

---

## 2. 域随机化与事件（Domain Randomization & Events）

### 2.1 Startup 域随机化（`domain_rand.events`）

针对 `RobanWalkFlatEnvCfg.domain_rand.events`，做了如下对齐：

- **物理材质随机化（physics_material）**
  - 原先（PB_AMP）：摩擦和恢复系数范围较窄：
    - `static_friction_range=(0.6, 1.0)`
    - `dynamic_friction_range=(0.4, 0.8)`
    - `restitution_range=(0.0, 0.005)`
  - 现在：改为与 amp_share 相同量级的更宽范围：
    - `static_friction_range=(0.3, 1.0)`
    - `dynamic_friction_range=(0.2, 1.0)`
    - `restitution_range=(0.0, 0.5)`

- **质量随机化**
  - 删除：原先仅对 `base_link` 做 `add_base_mass`（-5 ~ 5kg）的事件。
  - 新增：`scale_link_mass` 事件，与 amp_share 一致：
    - `asset_cfg=SceneEntityCfg("robot", body_names=["leg_.*_link", "zarm_.*_link"])`
    - `mass_distribution_params=(0.8, 1.2)`，`operation="scale"`。

- **质心 / 执行器 / 关节参数随机化**
  - 新增以下 startup 事件（与 amp_share 结构一致）：
    - `randomize_rigid_body_com`：
      - `asset_cfg("robot", body_names="base_link")`
      - `com_range` 在 x/y/z 各 ±0.05。
    - `scale_actuator_gains`：
      - `asset_cfg("robot", joint_names=".*_joint")`
      - `stiffness_distribution_params=(0.8, 1.2)`
      - `damping_distribution_params=(0.8, 1.2)`
      - `operation="scale"`。
    - `scale_joint_parameters`：
      - `asset_cfg("robot", joint_names=".*_joint")`
      - `friction_distribution_params=(1.0, 1.0)`
      - `armature_distribution_params=(0.5, 1.5)`
      - `operation="scale"`。

> 效果：roban 的 startup 域随机化现在与 amp_share 在“材质、link 质量、base COM、执行器增益、关节参数”五个维度上结构一致，仅参数范围做了直接对齐或轻微裁剪。

### 2.2 Reset 事件

- **根部状态重置（reset_base）**
  - 旧配置：x,y ∈ [-0.5, 0.5]，yaw ∈ [-π, π]，无 pitch/roll 项；速度范围 ±0.5。
  - 新配置：参考 amp_share 调整：
    - `x, y: (-0.7, 0.7)`
    - `yaw: (-3.14, 3.14)`
    - 新增 `pitch: (-0.1, 0.1)`，`roll: (-0.1, 0.1)`
    - 速度 `x,y,z,roll,pitch,yaw` 均在 `(-0.3, 0.3)` 内。

- **关节重置（reset_robot_joints）**
  - 维持与 amp_share 一致：
    - `position_range=(0.5, 1.5)`
    - `velocity_range=(0.0, 0.0)`。

### 2.3 Interval 推力事件

- **函数选择**
  - 原先 roban 使用 `mdp.push_by_setting_velocity_curriculum`，依赖 `_external_force_torque_level` 做“课程缩放推力”。
  - 现在改为与基础环境一致、且更简洁的：
    - `func=mdp.push_by_setting_velocity`。
  - 说明：
    - 在地形+推力课程中，**外力等级** 已通过 `_external_force_torque_level` 管理；这里采用“速度脉冲”形式，与 PB_AMP 其他任务保持一致。
    - 若后续需要完全复制 amp_share 的 `apply_external_force_torque_stochastic`（施加力/力矩），可单独扩展。

- **触发频率**
  - 原先：`interval_range_s=(10.0, 15.0)`（低频推力）。
  - 现在：`interval_range_s=(0.01, 0.3)`，与 amp_share 的外力扰动时间尺度相近（高频、弱扰动风格）。

- **扰动幅度**
  - 保持 `velocity_range={"x": (-1.0, 1.0), "y": (-1.0, 1.0)}` 不变，形成横向/纵向的小速度脉冲。

---

## 3. AMP 训练 Runner 与日志输出（`rsl_rl/rsl_rl/runners/amp_on_policy_runner.py`）

### 3.1 训练逻辑保持与 amp_share 对齐的要点

此部分之前已在 `PB_AMP_roban_velocity_training_flow.md` 中详述，这里仅列对齐要点：

- **Rollout 结构**：每迭代 `num_steps_per_env=24` 步，每步：
  - `act → env.step(actions)`（内部 decimation=4、4 个仿真步同一 action）。
  - 存储 transition（obs、rewards、dones、AMP obs）。
- **AMP 奖励组合**：
  - rollout 期间只存 task reward 与 AMP 状态；
  - 在 `compute_returns()` 内批量计算 style reward，并用：
    - `rewards = task_reward_weight * task + style_reward_weight * style`
  - 对应 amp_share 的 AMP 奖励组合方式。
- **判别器更新**：
  - 每次 `update()` 使用：
    - 策略侧：当前 rollout (24 × num_envs) 的 AMP 状态对；
    - 专家侧：AMP 数据集（多轨迹 npz），构造 expert batch。

这些逻辑没有在本次 REBUILD 中再改动，只做了补充日志上的精简（见下）。

### 3.2 终端日志输出精简

函数：`AmpOnPolicyRunner.log()`。

- **旧行为**：
  - 打印一大块 ASCII 边框；
  - 输出每个 loss 的详细数值；
  - 分别打印 extrinsic / intrinsic reward；
  - 打印大量时间统计，信息虽然完整但终端噪声很大。

- **新行为（更贴近“常规必要数据 + 课程进度”）**：
  - 每次迭代在终端输出一段简洁文本，例如：

    ```text
    [Iter it/tot] fps=XXXX mean_rew=XX.XX ep_len=YY.YY collect=as learn=bs std=Z.ZZ
      task_rew=... style_rew=... amp_ratio=...%     # 若使用 AMP 奖励组合
      Episode/xxx: ...                              # 来自 env.extras['log'] 的课程/episode 统计
      ...
      total_steps=NNNNNN elapsed=HH:MM:SS ETA=HH:MM:SS
    ```

  - 保留对 **课程相关指标** 的打印：
    - `ep_infos` 中的键，如：
      - `Curriculum/terrain_levels`
      - `Curriculum/external_force_level`
      - `Curriculum/mean_lin_sum`
      - `Curriculum/mean_ang_sum`
      - 以及其它来自环境 `infos["log"]` 的统计。
  - 所有 `writer.add_scalar(...)` 的 TensorBoard / WandB 记录逻辑保持不变，便于事后分析。

> 目标：终端只显示「迭代进度 + 性能概况 + 课程状态」，减少无关细节，便于观察 roban AMP 训练整体趋势。

---

## 4. 文档补充

本次对齐工作还新增/更新了以下文档，记录设计与实现细节：

- `docs/PB_AMP_roban_velocity_training_flow.md`
  - 整理了 PB_AMP 中 roban AMP 速度跟踪训练的完整流程（入口 → runner → rollout → AMP 奖励 → 课程），并与 amp_share 的训练流程做逐项对照；
  - 明确了 decimation、transition 记录方式、AMP 奖励组合、判别器与策略数据来源等。

- `docs/PB_AMP_vs_amp_share_event_command_domain_rand.md`
  - 专门对比 PB_AMP 与 amp_share 在 **Command、Event、Domain Randomization（域随机化）** 三个方面的差异与对齐情况；
  - 说明哪些地方已经完全对齐，哪些地方刻意保留差异（如当前仍未完全复制 amp_share 的完整 RewardsCfg，只对速度跟踪相关权重和课程逻辑做了对齐）。

---

## 5. 尚未完全对齐且“有意保留”的部分（提示）

为避免误解，这里明确列出目前 **尚未完全复制 amp_share**、但我们有意识暂时保留差异的部分：

- **奖励函数整体结构（RewardsCfg vs RobanLiteRewardCfg）**
  - 目前仅对齐了：
    - 速度跟踪核心项：`track_lin_vel_xy_exp` / `track_ang_vel_z_exp` 的权重与 stage_two 权重；
    - 与地形+推力课程配套的少量约束项（如 `action_rate_l2`、`dof_acc_l2` 等）。
  - amp_share 中其他大量 reward（如 `feet_height_cycle`、`turning_step_frequency`、更细的 torque/acc/对称性惩罚等）尚未一一迁移到 PB_AMP。
  - 原因：PB_AMP 当前采用的是一套更简化、偏“Pragmatic Minimalist” 的 gait reward 设计，希望先观察 AMP + 简化 reward 的表现，再视需要逐步引入更复杂的 shaping。

若后续需要 **进一步让 roban 的步态与 amp_share 完全一致**，建议的下一步工作是：

1. 在 `legged_lab/mdp/rewards.py` 中，为 Roban 建一个与 amp_share `RewardsCfg` 结构近似的新 RewardCfg；
2. 在 `RobanWalkFlatEnvCfg` 中使用该 RewardCfg，并按 amp_share 的 `stage_two` 配置逐项对齐权重；
3. 在文档中追加“Rewards 对齐说明”小节。

---

## 6. 总结

本次 REBUILD 的核心目标，是在 **不完全牺牲 PB_AMP 原有设计思想** 的前提下，让 **Roban AMP 速度跟踪任务** 在以下方面尽量与 amp_share 对齐：

- 物理与命令：dt / decimation / 命令范围 / 机体配置；
- AMP 算法：rollout 结构、AMP 奖励组合、判别器与策略更新逻辑；
- 课程：地形 + 推力等级 + stage_two 权重重写；
- 域随机化：材质、质量、COM、执行器、关节参数、推力频率与形式；
- 日志：只在终端展示训练进度及课程状态的“必要信息”。

已对齐部分足以让 **速度跟踪质量与训练行为大体处于同一“分布”**；若未来需要做到“几乎一模一样的步态与细节”，则需要进一步对 RewardsCfg 做更细致的迁移与验证。+
