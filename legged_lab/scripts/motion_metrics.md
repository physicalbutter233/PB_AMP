# 运动评估指标系统文档

> 文件位置: `legged_lab/scripts/play.py` — `MotionMetrics` 类  
> 适用环境: `RobanEnv` (Isaac Lab + RSL-RL)

---

## 概览

`MotionMetrics` 在 `play.py` 的推理循环中逐步收集机器人运动数据，并定期输出结构化的评估报告。报告分为四大模块：

| 模块 | 指标数量 | 作用 |
|------|---------|------|
| 速度跟踪统计 | 3 | 衡量策略对速度命令的跟踪精度 |
| 运动质量 | 3 | 衡量躯干姿态稳定性 |
| 效率指标 | 3 | 衡量能量消耗和动作平滑性 |
| 数据统计 | 2 | 记录采集规模 |

---

## 1. 速度跟踪统计

### 1.1 线速度平均误差 (m/s)

**含义**: 命令线速度 (x, y) 与实际线速度在 yaw 坐标系下的欧氏距离，所有环境取均值后再对所有时间步取均值。

**计算公式**:

$$
e_{lin} = \frac{1}{T} \sum_{t=1}^{T} \frac{1}{N} \sum_{i=1}^{N} \| \mathbf{v}_{cmd,xy}^{(i)} - \mathbf{v}_{actual,xy}^{(i)} \|_2
$$

**数据来源**:
- 命令速度: `env.command_generator.command[:, :2]`
- 实际速度: `robot.data.root_lin_vel_w` → 通过 `quat_rotate_inverse(yaw_quat(...))` 转到 yaw 坐标系

**对应 reward 函数**: `track_lin_vel_xy_yaw_frame_exp` (rewards.py)

---

### 1.2 角速度平均误差 (rad/s)

**含义**: 命令 yaw 角速度与实际 yaw 角速度的绝对差，所有环境取均值。

**计算公式**:

$$
e_{ang} = \frac{1}{T} \sum_{t=1}^{T} \frac{1}{N} \sum_{i=1}^{N} | \omega_{cmd,z}^{(i)} - \omega_{actual,z}^{(i)} |
$$

**数据来源**:
- 命令角速度: `env.command_generator.command[:, 2]`
- 实际角速度: `robot.data.root_ang_vel_w[:, 2]`

**对应 reward 函数**: `track_ang_vel_z_world_exp` (rewards.py)

---

### 1.3 综合评分 (%)

**含义**: 基于线速度和角速度误差的指数衰减加权评分，范围 0~100%。

**计算公式**:

$$
S = \left( 0.7 \cdot e^{-e_{lin}/0.5} + 0.3 \cdot e^{-e_{ang}/0.5} \right) \times 100\%
$$

- 权重: 线速度 70%，角速度 30%
- 标准差 σ = 0.5 (误差为 0.5 时得分约 37%)

---

## 2. 运动质量

### 2.1 Roll 稳定性 (rad / °)

**含义**: 机器人绕前进轴 (X) 的倾斜程度，越小越稳定。

**计算公式**:

$$
\text{roll} = \text{atan2}(g_y^{body}, g_z^{body})
$$

$$
\bar{R}_{roll} = \frac{1}{T} \sum_{t=1}^{T} \frac{1}{N} \sum_{i=1}^{N} |\text{roll}^{(i)}_t|
$$

**数据来源**: `robot.data.projected_gravity_b` — 重力向量在 body 坐标系下的投影

**对应 reward 函数**: `flat_orientation_l2` (rewards.py)

---

### 2.2 Pitch 稳定性 (rad / °)

**含义**: 机器人绕侧向轴 (Y) 的俯仰程度。

**计算公式**:

$$
\text{pitch} = \text{atan2}\left(-g_x^{body},\ \sqrt{(g_y^{body})^2 + (g_z^{body})^2}\right)
$$

**数据来源**: 同 Roll，来自 `robot.data.projected_gravity_b`

---

### 2.3 Yaw 稳定性 (rad / °)

**含义**: 机器人绕竖直轴 (Z) 的角速度波动程度，越小说明行走方向越稳。

**计算公式**:

$$
\bar{R}_{yaw} = \frac{1}{T} \sum_{t=1}^{T} \frac{1}{N} \sum_{i=1}^{N} |\omega_{z}^{(i)}_t|
$$

**数据来源**: `robot.data.root_ang_vel_w[:, 2]`

---

## 3. 效率指标

### 3.1 平均功耗

**含义**: 所有关节的瞬时机械功率绝对值之和，在所有环境和时间步上取均值。

**计算公式**:

$$
P = \frac{1}{T} \sum_{t=1}^{T} \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{J} |\tau_j^{(i)} \cdot \dot{q}_j^{(i)}|
$$

**数据来源**:
- 关节力矩: `robot.data.applied_torque`
- 关节速度: `robot.data.joint_vel`

**对应 reward 函数**: `energy` (rewards.py)

---

### 3.2 能效比 (Cost of Transport 的倒数形式)

**含义**: 单位功耗产生的移动速度，越高越省能。

**计算公式**:

$$
\eta = \frac{\bar{v}_{xy}}{\bar{P} + \epsilon}
$$

其中 ε = 1e-6 防止除零。

**数据来源**: 速度来自 yaw 坐标系下的 `root_lin_vel`，功耗同 3.1

---

### 3.3 关节平滑度

**含义**: 连续两帧动作变化量的 L2 范数的倒数映射，范围 (0, 1]，越接近 1 动作越平滑。

**计算公式**:

$$
\Delta a = \frac{1}{T-1} \sum_{t=2}^{T} \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{J} (a_{j,t}^{(i)} - a_{j,t-1}^{(i)})^2
$$

$$
\text{smoothness} = \frac{1}{1 + \Delta a}
$$

**数据来源**: `actions` (策略输出)

**对应 reward 函数**: `action_rate_l2` (rewards.py)

---

## 4. 数据统计

| 指标 | 含义 |
|------|------|
| **传感器数据点** | `update()` 被调用的总次数，即仿真步数 |
| **历史记录数** | 速度误差列表长度（与传感器数据点相同） |

---

## 使用方法

在 `play.py` 的推理循环中已自动集成：

```python
# 初始化（print_interval 控制打印频率）
metrics = MotionMetrics(device=env.device, print_interval=200)

# 每步更新
while simulation_app.is_running():
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        metrics.update(env, actions)      # 收集数据
        metrics.print_stats()             # 按间隔打印

# 退出时打印最终报告
metrics.print_stats(force=True)
```

### 参数调节

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `print_interval` | 200 | 每 N 步打印一次报告 |
| 综合评分 σ | 0.5 | 指数衰减标准差，越小对误差越敏感 |
| 综合评分权重 | 0.7 / 0.3 | 线速度 vs 角速度的重要性比例 |

---

## 输出示例

```
============================================================
  运动评估报告 (步数: 200)
============================================================

速度跟踪统计:
  - 综合评分: 28.6%
  - 线速度平均误差: 1.000 m/s
  - 角速度平均误差: 0.000 rad/s

运动质量:
  - Roll稳定性:  0.0312 rad (1.79°)
  - Pitch稳定性: 0.0156 rad (0.89°)
  - Yaw稳定性:   0.0089 rad (0.51°)

效率指标:
  - 平均功耗: 45.32
  - 能效比: 0.033
  - 关节平滑度: 0.872

数据统计:
  - 传感器数据点: 200
  - 历史记录数: 200
============================================================
```

---

## 指标与 Reward 函数对照表

| 评估指标 | 对应 Reward 函数 | 文件 |
|---------|-----------------|------|
| 线速度误差 | `track_lin_vel_xy_yaw_frame_exp` | `legged_lab/mdp/rewards.py` |
| 角速度误差 | `track_ang_vel_z_world_exp` | `legged_lab/mdp/rewards.py` |
| Roll/Pitch 稳定性 | `flat_orientation_l2` | `legged_lab/mdp/rewards.py` |
| 平均功耗 | `energy` | `legged_lab/mdp/rewards.py` |
| 关节平滑度 | `action_rate_l2` | `legged_lab/mdp/rewards.py` |
