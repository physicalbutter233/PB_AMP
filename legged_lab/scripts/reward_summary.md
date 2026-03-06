# Roban Walk 奖励函数配置总结

> 配置文件：`legged_lab/envs/roban/walk_cfg.py`
> 实际使用类：`RobanLiteRewardCfg`（继承自 `LiteRewardCfg`，覆盖了部分 body 名称）
> 生成时间：2026-02-12

---

## 1. Task 奖励总览

### 1.1 启用的奖励项

| # | 奖励名称 | 函数 | 权重 | 类别 | 来源 | 说明 |
|---|---------|------|------|------|------|------|
| 1 | `track_lin_vel_xy_exp` | `mdp.track_lin_vel_xy_yaw_frame_exp` | **+2.5** | 速度跟踪 | LiteRewardCfg | yaw系下线速度跟踪，std=0.5 |
| 2 | `track_ang_vel_z_exp` | `mdp.track_ang_vel_z_world_exp` | **+2.0** | 速度跟踪 | LiteRewardCfg | 角速度跟踪，std=0.5 |
| 3 | `lin_vel_z_l2` | `mdp.lin_vel_z_l2` | **-0.5** | 速度惩罚 | LiteRewardCfg | 垂直线速度惩罚 |
| 4 | `ang_vel_xy_l2` | `mdp.ang_vel_xy_l2` | **-0.5** | 速度惩罚 | LiteRewardCfg | 水平角速度惩罚 |
| 5 | `flat_orientation_l2` | `mdp.flat_orientation_l2` | **-1.0** | 姿态控制 | LiteRewardCfg | 保持身体水平 |
| 6 | `energy` | `mdp.energy` | **-1e-5** | 能量消耗 | LiteRewardCfg | 关节力矩×速度 |
| 7 | `dof_acc_l2` | `mdp.joint_acc_l2` | **-1e-7** | 能量消耗 | LiteRewardCfg | 关节加速度惩罚 |
| 8 | `action_rate_l2` | `mdp.action_rate_l2` | **-0.002** | 动作平滑 | LiteRewardCfg | 动作变化率惩罚 |
| 9 | `undesired_contacts` | `mdp.undesired_contacts` | **-1.0** | 碰撞惩罚 | Roban覆盖 | 膝/臂/base碰撞 |
| 10 | `feet_slide` | `mdp.feet_slide` | **-0.05** | 脚部质量 | Roban覆盖 | 脚部滑动惩罚 |
| 11 | `feet_force` | `mdp.body_force` | **-0.01** | 脚部质量 | Roban覆盖 | 脚部接触力惩罚，threshold=500N |
| 12 | `feet_air_time` | `mdp.feet_air_time_clip_biped` | **+20.0** | 步态 | Roban新增 | 滞空时间 [0.25s, 0.4s] 裁剪 |
| 13 | `step_frequency` | `mdp.step_frequency_penalty` | **+10.0** | 步态 | Roban新增 | 双脚同时着地>0.2s惩罚 |
| 14 | `feet_height` | `mdp.feet_height_cycle` | **+12.0** | 步态 | Roban新增 | 抬脚高度，clip=15cm |
| 15 | `straight_knee_landing` | `mdp.contact_ground_straight_knee` | **+15.0** | 落地质量 | Roban新增 | 着地时膝盖伸直，std=0.3 |
| 16 | `soft_landing` | `mdp.contact_momentum` | **+1.0** | 落地质量 | Roban新增 | 轻柔落地，std=0.05 |
| 17 | `feet_distance` | `mdp.feet_distance_penalty` | **-30.0** | 防诡异行走 | Roban新增 | 两脚距离 [22cm, 30cm] 外惩罚 |
| 18 | `feet_contact_time_symmetry` | `mdp.feet_contact_time_symmetry_exp` | **+1.6** | 对称性 | Roban新增 | 左右脚触地时间对称，sigma=0.25 |

### 1.2 注释掉的奖励项（未启用）

| 奖励名称 | 函数 | 权重 | 说明 |
|---------|------|------|------|
| `hip_roll_penalty` | `mdp.hip_roll_conditional_penalty` | -5.0 | 髋关节Roll防劈叉 |
| `hip_yaw_penalty` | `mdp.hip_yaw_conditional_penalty` | -5.0 | 髋关节Yaw防内八 |
| `knee_pitch_penalty` | `mdp.deadzone_penalty` | -5.0 | 膝盖限位惩罚 |
| `joint_pos_symmetry` | `mdp.joint_pos_symmetry_l2` | -0.5 | 关节位置对称性 |
| `joint_vel_symmetry` | `mdp.joint_vel_symmetry_l2` | -0.001 | 关节速度对称性 |

---

## 2. AMP 风格奖励配置

| 参数 | 值 | 说明 |
|------|---|------|
| `task_reward_weight` | 1.0 | Task 奖励乘数 |
| `style_reward_weight` | 0.02 | AMP 风格奖励乘数 |
| `discriminator_reward_scale` | 5.0 | 判别器输出缩放 |
| `amp_motion_files` | `walk_pb_add_root.txt` | 参考动作数据 |
| `amp_discr_hidden_dims` | [1024, 512, 256] | 判别器网络结构 |

**最终奖励公式：**
```
combined_reward = 1.0 × task_reward + 0.02 × (5.0 × style_reward)
                = task_reward + 0.1 × style_reward
```
AMP 最大贡献 = 0.1（当 style_reward = 1.0 时）

---

## 3. 权重分布分析

### 3.1 按绝对值排序（正奖励 + |负惩罚|）

| 排名 | 奖励名称 | |权重| | 类别 |
|------|---------|-------|------|
| 1 | `feet_distance` | 30.0 | 防诡异行走 |
| 2 | `feet_air_time` | 20.0 | 步态 |
| 3 | `straight_knee_landing` | 15.0 | 落地质量 |
| 4 | `feet_height` | 12.0 | 步态 |
| 5 | `step_frequency` | 10.0 | 步态 |
| 6 | **`track_lin_vel_xy_exp`** | **2.5** | **速度跟踪** |
| 7 | **`track_ang_vel_z_exp`** | **2.0** | **速度跟踪** |
| 8 | `feet_contact_time_symmetry` | 1.6 | 对称性 |
| 9 | `flat_orientation_l2` | 1.0 | 姿态控制 |
| 10 | `undesired_contacts` | 1.0 | 碰撞惩罚 |
| 11 | `soft_landing` | 1.0 | 落地质量 |
| 12-18 | 其余 | <0.5 | 各类小惩罚 |

### 3.2 按类别汇总

| 类别 | 正奖励权重和 | 负惩罚权重和 | 净权重 |
|------|------------|------------|-------|
| **速度跟踪** | **4.5** | 0 | **4.5** |
| 速度惩罚 | 0 | -1.0 | -1.0 |
| 步态相关 | 42.0 | 0 | 42.0 |
| 落地质量 | 16.0 | 0 | 16.0 |
| 防诡异行走 | 0 | -30.0 | -30.0 |
| 姿态控制 | 0 | -1.0 | -1.0 |
| 对称性 | 1.6 | 0 | 1.6 |
| 碰撞惩罚 | 0 | -1.0 | -1.0 |
| 脚部质量 | 0 | -0.06 | -0.06 |
| 能量/平滑 | 0 | ~-0.002 | ~0 |
| **AMP 风格** | **0.1** (max) | 0 | **0.1** |

---

## 4. 潜在问题

**速度跟踪权重占比过低：**
- 速度跟踪正奖励总计 = 4.5
- 步态+落地正奖励总计 = 58.0
- 速度跟踪仅占正奖励的 **7.2%**

这导致策略优先优化步态质量而非速度跟踪，velocity curriculum 升级条件（线速度误差 < 0.2 m/s）难以满足。

**建议方案（配置注释中已标注）：**
```python
track_lin_vel_xy_exp: weight 2.5 → 10.0
track_ang_vel_z_exp: weight 2.0 → 8.0
```
调整后速度跟踪占比提升至 **23.6%**，更有利于课程学习升级。

---

## 5. 速度课程学习配置

| 参数 | 值 | 说明 |
|------|---|------|
| `enable` | True | 是否启用 |
| `promote_threshold` | 0.75 | 升级：ep长度 > 75% × 20s = 15s |
| `demote_threshold` | 0.35 | 降级：ep长度 < 35% × 20s = 7s |
| `promote_lin_vel_err_limit` | 0.2 m/s | 升级：线速度误差上限 |
| `promote_ang_vel_err_limit` | 0.2 rad/s | 升级：角速度误差上限 |
| `buffer_size` | 2000 | 统计缓冲区 |
| `cooldown_steps` | 500 | 升/降级后冷却步数 |

| Level | lin_vel_x | lin_vel_y | ang_vel_z |
|-------|-----------|-----------|-----------|
| 0 | (0.0, 0.3) | (-0.1, 0.1) | (-0.3, 0.3) |
| 1 | (-0.2, 0.5) | (-0.2, 0.2) | (-0.8, 0.8) |
| 2 | (-0.4, 0.8) | (-0.3, 0.3) | (-1.2, 1.2) |
| 3 | (-0.6, 1.0) | (-0.5, 0.5) | (-1.57, 1.57) |
