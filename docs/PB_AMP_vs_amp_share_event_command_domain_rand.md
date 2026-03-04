# PB_AMP 与 amp_share：Event、Command、域随机化对比（Roban 速度跟踪）

针对 AMP 速度跟踪任务（roban），对两边的 **Command**、**Event**、**域随机化（Domain Randomization）** 做逐项对比。  
对照代码：PB_AMP `legged_lab/envs/roban/walk_cfg.py`（RobanWalkFlatEnvCfg）与 amp_share `velocity_amp_env_cfg.py` + `roban_env_cfg.py`（RobanS2MixEnvCfg / RobanS2RoughEnvCfg）。

---

## 1. Command（速度命令）

| 项目 | amp_share | PB_AMP roban_walk |
|------|-----------|-------------------|
| 命令类型 | `base_velocity` (UniformVelocityCommand) | `commands` → UniformVelocityCommand |
| resampling_time_range | (0.5, 5.0) | (0.5, 5.0) |
| rel_standing_envs | 0.5 | 0.5 |
| rel_heading_envs | 0.8 | 0.8 |
| heading_command | True | True |
| heading_control_stiffness | 0.3 | 0.3 |
| **ranges** | lin_vel_x=(-0.6, 0.6), lin_vel_y=(-0.6, 0.6), ang_vel_z=(-1.0, 1.0), heading=(-π, π) | **相同** |
| 速度课程 | 无（Mix 配置下直接全范围） | 可选 **VelocityCurriculumCfg**：reset 时按级别改 ranges + 奖励权重，可用 `--no-velocity-curriculum` 关闭 |

**结论**：基础 command 范围与重采样等**一致**；PB_AMP 多了一个可选的**速度课程**（分阶段放宽 lin_vel/ang_vel 与奖励权重）。

---

## 2. Event 与域随机化总览

- **amp_share**：`EventCfg` 下多种 event（startup / reset / interval），与 `CurriculumCfg`（terrain_levels、external_force_torque_levels、stage_two）配合。
- **PB_AMP**：`domain_rand.events`（EventCfg） + `terrain_force_curriculum`（地形+推力课程，stage_two 权重重写）；interval 推力使用 **push_by_setting_velocity_curriculum**（按课程 level 缩放）。

---

## 3. Startup 域随机化（每环境启动时一次）

| 项目 | amp_share | PB_AMP roban_walk |
|------|-----------|-------------------|
| **物理材质** | physics_material：static_friction (0.3, 1.0)，dynamic_friction (0.2, 1.0)，restitution (0, 0.5)，make_consistent=True | physics_material：(0.6, 1.0)，(0.4, 0.8)，(0, 0.005)；无 make_consistent |
| **质量** | scale_link_mass：腿/臂 link 质量 **scale** (0.8, 1.2) | add_base_mass：仅 **base_link** 质量 **加** (-5, 5) kg |
| **质心** | randomize_rigid_body_com：base_link 质心 (x,y,z) ±0.05 | **无** |
| **执行器增益** | scale_actuator_gains：刚度/阻尼 scale (0.8, 1.2) | **无** |
| **关节参数** | scale_joint_parameters：摩擦 1.0，armature scale (0.5, 1.5) | **无** |

**结论**：  
- **PB_AMP**：只做**物理材质**（摩擦/恢复系数范围更窄）+ **base 质量加减**，不做 link 质量缩放、质心、执行器、关节参数随机化。  
- **amp_share**：startup 随机化更丰富（材质范围更大、link 质量、质心、执行器、关节均有），仿真形态差异更大。

---

## 4. Reset 时 Event（每 episode 重置时）

| 项目 | amp_share | PB_AMP roban_walk |
|------|-----------|-------------------|
| **reset_base** | pose: x,y ±0.7，yaw ±π，**pitch/roll ±0.1**；velocity ±0.3 | pose: x,y ±0.5，yaw ±π（**无 pitch/roll**）；velocity ±0.5 |
| reset_robot_joints | position_range (0.5, 1.5)，velocity (0, 0) | 同左 |

**结论**：  
- **PB_AMP**：初始位置范围略小（±0.5）、初速范围略大（±0.5）、**不随机 pitch/roll**。  
- **amp_share**：位置范围略大（±0.7）、初速 ±0.3、**有 pitch/roll ±0.1**。

---

## 5. Interval 时 Event（训练中周期性触发）

| 项目 | amp_share | PB_AMP roban_walk |
|------|-----------|-------------------|
| **实现** | apply_external_force_torque_stochastic：对 base 施加**力/力矩** | push_by_setting_velocity_curriculum：直接改 base **线速度**（再按课程 level 缩放） |
| **触发间隔** | interval_range_s = (0.01, 0.3) | interval_range_s = (10.0, 15.0) |
| **强度/范围** | force x,y,z ∈ [-100, 100]，torque ∈ [-50, 50]；probability=0.01 | velocity_range x,y ∈ (-1.0, 1.0) m/s，缩放系数 = 当前 env 的 _external_force_torque_level |
| **与课程关系** | 与 external_force_torque_levels 课程独立（随机力/力矩） | **与地形+推力课程绑定**：同一课程更新 _external_force_torque_level，推力大小 = level × 速度扰动 |

**结论**：  
- **amp_share**：**高频率**（约 0.01–0.3 s 间隔）、**随机力/力矩**、概率 0.01，扰动形式为力。  
- **PB_AMP**：**低频率**（10–15 s 间隔）、**速度脉冲**、强度由课程 level 线性缩放，与 terrain_force_curriculum 一致（“amp_share 推力课程同款”逻辑在 PB_AMP 的 push_by_setting_velocity_curriculum 中）。

---

## 6. 课程（Curriculum）与 Stage Two

| 项目 | amp_share | PB_AMP roban_walk |
|------|-----------|-------------------|
| **地形** | curriculum.terrain_levels（terrain_levels_vel） | terrain_force_curriculum：地形等级 + 推力等级统一管理 |
| **外力/推力等级** | curriculum.external_force_torque_levels | 同上，存在 env._external_force_torque_level，用于 interval 推力缩放 |
| **Stage two** | curriculum.stage_two：mean(外力水平) ≥ 阈值时一次性重写 reward 权重（如 track_lin_vel_xy_exp=4.5, track_ang_vel_z_exp=2.25 等） | TerrainForceCurriculumCfg.stage_two_force_threshold（默认 0.9）：mean(force_level) ≥ 时应用 stage_2_reward_weights（与 amp_share stage_2 权重对齐） |

**结论**：  
- 两边都是「地形 + 外力水平」课程，并在外力达到阈值时进入 **stage two** 重写部分奖励权重。  
- PB_AMP 将地形与推力课程合在一个 **TerrainForceCurriculumCfg** 里，且 interval 推力显式用 **push_by_setting_velocity_curriculum** 按 level 缩放，与文档中“地形+推力课程与 stage two 完全参考 amp_share”的意图一致，但 **interval 的触发方式与物理形式不同**（见上节）。

---

## 7. 观测噪声（Noise）

| 项目 | amp_share | PB_AMP roban_walk |
|------|-----------|-------------------|
| 方式 | Policy 观测各 term 上 Unoise(n_min, n_max)（如 ang_vel ±0.2, gravity ±0.05, joint_pos ±0.05, joint_vel ±1.5） | noise_scales：lin_vel 0.2, ang_vel 0.2, projected_gravity 0.05, joint_pos 0.01, joint_vel 1.5, height_scan 0.1 |
| Critic | enable_corruption=False（无噪声） | 无单独 noise 配置，一般 critic 用真实/privileged 观测 |

**结论**：两边都在 policy 观测上加噪声，量级大致可对齐（PB_AMP 的 joint_pos 更小 0.01 vs 0.05）；amp_share 显式对每个 obs term 设 Unoise，PB_AMP 用统一 noise_scales。

---

## 8. 小结：差异汇总

- **Command**：基础范围与重采样**一致**；PB_AMP 多**可选速度课程**。  
- **Startup 域随机化**：**amp_share 更丰富**（材质范围更大 + link 质量/质心/执行器/关节）；**PB_AMP 更精简**（仅材质 + base 质量加减）。  
- **Reset**：PB_AMP 无 pitch/roll 随机、位置略小、初速略大；amp_share 有 pitch/roll ±0.1。  
- **Interval 推力**：**amp_share** 为高频率随机**力/力矩**；**PB_AMP** 为低频率**速度脉冲**且按课程 level 缩放（“推力课程同款”）。  
- **课程与 stage two**：逻辑对齐（地形+外力等级，阈值后重写权重），PB_AMP 用统一 TerrainForceCurriculumCfg + push_by_setting_velocity_curriculum 实现。

若希望**训练效果更接近 amp_share**，可在 PB_AMP 中考虑：  
- 增加 startup 的 link 质量/质心/执行器/关节随机化；  
- 放宽 physics_material 范围或对齐 amp_share；  
- reset 增加 pitch/roll 小范围随机；  
- 视需求调整 interval 推力频率与形式（或保留当前“低频+课程缩放”以保持与现有课程一致）。
