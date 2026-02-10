# Reward 奖励函数公式手册

> 源文件: `legged_lab/mdp/rewards.py`  
> 配置文件: `legged_lab/envs/roban/walk_cfg.py` — `RobanLiteRewardCfg`  
> 适用机器人: Roban S14 (21-DoF 人形)

---

## 总览

最终每步奖励的计算方式：

$$
r_{total} = \underbrace{r_{task}}_{Task\ Reward} \quad \text{其中} \quad r_{task} = \sum_{k} w_k \cdot r_k(s, a)
$$

当启用 AMP 时，最终组合奖励为：

$$
r_{combined} = w_{task} \cdot r_{task} + w_{style} \cdot (s_{disc} \cdot r_{style})
$$

当前配置: $w_{task} = 1.0$, $w_{style} = 0.02$, $s_{disc} = 2.0$

---

## 当前启用的 Reward 项（RobanLiteRewardCfg）

| # | 名称 | 权重 | 类型 | 目标 |
|---|------|------|------|------|
| 1 | `track_lin_vel_xy_exp` | +5.0 | 正向奖励 | 线速度跟踪 |
| 2 | `track_ang_vel_z_exp` | +4.0 | 正向奖励 | 角速度跟踪 |
| 3 | `lin_vel_z_l2` | -0.5 | 惩罚 | 抑制垂直速度 |
| 4 | `ang_vel_xy_l2` | -0.5 | 惩罚 | 抑制 roll/pitch 角速度 |
| 5 | `flat_orientation_l2` | -1.0 | 惩罚 | 保持躯干水平 |
| 6 | `energy` | -1e-5 | 惩罚 | 节省能量 |
| 7 | `dof_acc_l2` | -1e-7 | 惩罚 | 减小关节加速度 |
| 8 | `action_rate_l2` | -0.002 | 惩罚 | 动作平滑 |
| 9 | `undesired_contacts` | -1.0 | 惩罚 | 避免非期望碰撞 |
| 10 | `feet_slide` | -0.05 | 惩罚 | 抑制脚滑 |
| 11 | `feet_force` | -0.01 | 惩罚 | 限制脚部冲击力 |
| 12 | `joint_pos_tracking_exp_all` | +0.3 | 正向奖励 | 关节接近默认位 |
| 13 | `joint_vel_tracking_all` | -0.05 | 惩罚 | 关节速度平滑 |

---

## 一、速度跟踪奖励

### 1. `track_lin_vel_xy_yaw_frame_exp`

**作用**: 奖励机器人 xy 平面线速度与命令速度的匹配程度。

**公式**:

$$
r = \exp\!\left( -\frac{\| \mathbf{v}_{cmd,xy} - \mathbf{v}_{yaw,xy} \|^2}{\sigma^2} \right)
$$

其中:
- $\mathbf{v}_{yaw}$ = 将世界坐标系线速度通过 `quat_rotate_inverse(yaw_quat(q))` 旋转到 yaw 坐标系
- $\sigma = 0.5$ (std 参数)

**值域**: $(0, 1]$，完美跟踪时为 1

**当前配置**: `weight = 5.0`, `std = 0.5`

**代码**:
```python
vel_yaw = quat_rotate_inverse(yaw_quat(root_quat_w), root_lin_vel_w[:, :3])
lin_vel_error = sum((cmd[:, :2] - vel_yaw[:, :2])^2)
reward = exp(-lin_vel_error / std^2)
```

---

### 2. `track_ang_vel_z_world_exp`

**作用**: 奖励机器人 yaw 角速度与命令角速度的匹配。

**公式**:

$$
r = \exp\!\left( -\frac{(\omega_{cmd,z} - \omega_{actual,z})^2}{\sigma^2} \right)
$$

**值域**: $(0, 1]$

**当前配置**: `weight = 4.0`, `std = 0.5`

**代码**:
```python
ang_vel_error = (cmd[:, 2] - root_ang_vel_w[:, 2])^2
reward = exp(-ang_vel_error / std^2)
```

---

## 二、速度惩罚

### 3. `lin_vel_z_l2`

**作用**: 惩罚 body 坐标系下的垂直线速度，鼓励平稳移动。

**公式**:

$$
r = v_{z,body}^2
$$

**当前配置**: `weight = -0.5`

**代码**:
```python
reward = (root_lin_vel_b[:, 2])^2
```

---

### 4. `ang_vel_xy_l2`

**作用**: 惩罚 body 坐标系下的 roll/pitch 角速度。

**公式**:

$$
r = \omega_{x,body}^2 + \omega_{y,body}^2
$$

**当前配置**: `weight = -0.5`

**代码**:
```python
reward = sum((root_ang_vel_b[:, :2])^2)
```

---

## 三、姿态控制

### 5. `flat_orientation_l2`

**作用**: 惩罚躯干偏离水平姿态（通过重力投影衡量）。

**公式**:

$$
r = g_{x,body}^2 + g_{y,body}^2
$$

其中 $\mathbf{g}_{body} = (g_x, g_y, g_z)$ 为重力向量在 body 坐标系的投影。  
理想水平时 $g_x = g_y = 0$, $g_z = -1$。

**当前配置**: `weight = -1.0`

**代码**:
```python
reward = sum((projected_gravity_b[:, :2])^2)
```

---

## 四、能量与平滑性

### 6. `energy`

**作用**: 惩罚关节的机械功消耗。

**公式**:

$$
r = \left\| |\boldsymbol{\tau}| \circ |\dot{\mathbf{q}}| \right\|_2 = \sqrt{\sum_j |\tau_j \cdot \dot{q}_j|^2}
$$

其中 $\boldsymbol{\tau}$ 为关节力矩, $\dot{\mathbf{q}}$ 为关节速度, $\circ$ 为逐元素乘法。

**当前配置**: `weight = -1e-5`

**代码**:
```python
reward = norm(abs(applied_torque * joint_vel), dim=-1)
```

---

### 7. `joint_acc_l2` (dof_acc_l2)

**作用**: 惩罚关节加速度，鼓励平滑运动。

**公式**:

$$
r = \sum_j \ddot{q}_j^2
$$

**当前配置**: `weight = -1e-7`

**代码**:
```python
reward = sum(joint_acc[:, joint_ids]^2)
```

---

### 8. `action_rate_l2`

**作用**: 惩罚连续帧之间的动作变化率，鼓励动作平滑。

**公式**:

$$
r = \sum_j (a_{j,t} - a_{j,t-1})^2
$$

其中 $a_{j,t}$ 和 $a_{j,t-1}$ 从 action buffer 的最近两帧获取。

**当前配置**: `weight = -0.002`

**代码**:
```python
reward = sum((action_buffer[:, -1, :] - action_buffer[:, -2, :])^2)
```

---

## 五、碰撞与接触

### 9. `undesired_contacts`

**作用**: 惩罚非足部 body 与地面的接触。

**公式**:

$$
r = \sum_b \mathbb{1}\!\left[ \max_{history} \| \mathbf{F}_{b} \| > \text{threshold} \right]
$$

统计接触力超过阈值的 body 数量。

**当前配置**: `weight = -1.0`, `threshold = 1.0`  
**监控 body**: `leg_l4_link`, `leg_r4_link`, `zarm_l2_link`, `zarm_r2_link`, `zarm_l4_link`, `zarm_r4_link`, `base_link`

**代码**:
```python
is_contact = max(norm(net_forces_w_history[:, :, body_ids]), dim=1) > threshold
reward = sum(is_contact)
```

---

### 10. `feet_slide`

**作用**: 惩罚脚部触地时的滑动（触地脚应该静止）。

**公式**:

$$
r = \sum_{f \in \text{feet}} \| \mathbf{v}_{f,xy} \| \cdot \mathbb{1}[\text{contact}_f]
$$

其中 $\text{contact}_f$ 表示脚 $f$ 触地（接触力 > 1.0 N）。

**当前配置**: `weight = -0.05`  
**监控 body**: `leg_l6_link`, `leg_r6_link`

**代码**:
```python
contacts = net_forces_w_history.norm().max(dim=1) > 1.0
body_vel = body_lin_vel_w[:, body_ids, :2]
reward = sum(body_vel.norm() * contacts)
```

---

### 11. `body_force` (feet_force)

**作用**: 惩罚脚部过大的 z 方向接触力（超过阈值部分）。

**公式**:

$$
r = \text{clamp}\!\left(\| F_{z,feet} \| - \text{threshold},\ 0,\ \text{max\_reward}\right)
$$

**当前配置**: `weight = -0.01`, `threshold = 500 N`, `max_reward = 400`  
**监控 body**: `leg_l6_link`, `leg_r6_link`

**代码**:
```python
reward = net_forces_w[:, body_ids, 2].norm()
reward[reward < 500] = 0
reward[reward > 500] -= 500
reward = clamp(reward, 0, 400)
```

---

## 六、关节跟踪

### 12. `joint_pos_tracking_exp` (joint_pos_tracking_exp_all)

**作用**: 指数奖励关节接近默认位置，防止诡异行走姿态。

**公式**:

$$
r = \exp\!\left( -\frac{\sum_j (q_j - q_{j,default})^2}{\sigma^2} \right)
$$

**值域**: $(0, 1]$，关节完美对齐默认位时为 1

**当前配置**: `weight = 0.3`, `std = 0.5`, 跟踪**所有关节**

**代码**:
```python
joint_pos_diff = joint_pos[:, joint_ids] - default_joint_pos[:, joint_ids]
error = sum(joint_pos_diff^2)
reward = exp(-error / std^2)
```

---

### 13. `joint_vel_tracking_l2` (joint_vel_tracking_all)

**作用**: 惩罚关节速度偏离默认值（通常为 0），鼓励平滑运动。

**公式**:

$$
r = \sum_j (\dot{q}_j - \dot{q}_{j,default})^2
$$

**当前配置**: `weight = -0.05`, 跟踪**所有关节**

**代码**:
```python
joint_vel_diff = joint_vel[:, joint_ids] - default_joint_vel[:, joint_ids]
reward = sum(joint_vel_diff^2)
```

---

## 七、其他可用但当前未启用的 Reward 函数

以下函数在 `rewards.py` 中定义，但未在 `RobanLiteRewardCfg` 中启用。

### `is_terminated`

**作用**: 惩罚非超时的异常终止。

$$
r = \text{reset\_buf} \cdot \neg \text{time\_out\_buf}
$$

---

### `feet_air_time_positive_biped`

**作用**: 奖励双足交替步行（单脚支撑）。

$$
r = \text{clamp}\!\left(\min_f t_{mode,f} \cdot \mathbb{1}[\text{single\_stance}],\ \text{threshold}\right) \cdot \mathbb{1}[\|\mathbf{v}_{cmd}\| > 0.1]
$$

其中 $t_{mode,f}$ 为当前脚处于着地或腾空的持续时间，`single_stance` 表示恰好一只脚着地。

**参数**: `threshold` (最大奖励时长)

---

### `joint_deviation_l1`

**作用**: 零速度命令时惩罚关节偏离默认位。

$$
r = \sum_j |q_j - q_{j,default}| \cdot \mathbb{1}[\|\mathbf{v}_{cmd}\| < 0.1]
$$

---

### `joint_pos_tracking_l2`

**作用**: L2 形式的关节位置跟踪惩罚。

$$
r = \sum_j (q_j - q_{j,default})^2
$$

---

### `body_orientation_l2`

**作用**: 惩罚指定 body 的姿态偏差（通过重力投影）。

$$
r = g_{x,body}^2 + g_{y,body}^2
$$

与 `flat_orientation_l2` 类似，但可指定特定 body。

---

### `feet_stumble`

**作用**: 惩罚脚部水平力过大（相对于垂直力）。

$$
r = \mathbb{1}\!\left[\exists f:\ \|F_{f,xy}\| > 5 \cdot |F_{f,z}|\right]
$$

---

### `feet_too_near_humanoid`

**作用**: 惩罚双脚过于接近。

$$
r = \text{clamp}(\text{threshold} - \|\mathbf{p}_{left} - \mathbf{p}_{right}\|,\ 0)
$$

**参数**: `threshold = 0.2 m`

---

### `ankle_torque`

**作用**: 惩罚脚踝关节力矩过大。

$$
r = \sum_{j \in \text{ankle}} \tau_j^2
$$

---

### `ankle_action`

**作用**: 惩罚脚踝关节动作幅度。

$$
r = \sum_{j \in \text{ankle}} |a_j|
$$

---

### `hip_roll_action`

**作用**: 惩罚 hip roll 关节动作幅度。

$$
r = |a_{hip\_roll\_L}| + |a_{hip\_roll\_R}|
$$

---

### `hip_yaw_action`

**作用**: 惩罚 hip yaw 关节动作幅度。

$$
r = |a_{hip\_yaw\_L}| + |a_{hip\_yaw\_R}|
$$

---

### `feet_y_distance`

**作用**: 低侧向速度时惩罚双脚 y 方向间距偏离目标值。

$$
r = |y_{left,body} - y_{right,body} - 0.299| \cdot \mathbb{1}[|v_{cmd,y}| < 0.1]
$$

---

## 八、步态周期奖励（Gait Clock）

这些奖励依赖 `gait_clock` 函数生成的相位信号。

### `gait_clock` 信号

将步态周期 $\phi \in [0, 1]$ 分为**摆动期** (swing) 和**支撑期** (stance)：

$$
I_{swing}(\phi) = \begin{cases}
0.5 + \frac{\phi}{2\Delta t} & \phi < \Delta t \quad \text{(过渡)} \\
1.0 & \Delta t \le \phi \le r_{air} - \Delta t \quad \text{(摆动)} \\
-\frac{\phi - r_{air} - \Delta t}{2\Delta t} & r_{air} - \Delta t < \phi < r_{air} + \Delta t \quad \text{(过渡)} \\
0.0 & r_{air} + \Delta t \le \phi \le 1 - \Delta t \quad \text{(支撑)} \\
\frac{\phi - 1 + \Delta t}{2\Delta t} & \phi > 1 - \Delta t \quad \text{(过渡)}
\end{cases}
$$

$$
I_{stance} = 1 - I_{swing}
$$

其中 $r_{air}$ 为摆动期占比，$\Delta t$ 为过渡宽度。

---

### `gait_feet_frc_perio`

**作用**: 摆动期脚不应有触地力。

$$
r = \sum_{f \in \{L,R\}} I_{swing,f} \cdot \exp\!(-200 \cdot F_{avg,f}^2)
$$

**参数**: `delta_t = 0.02`

---

### `gait_feet_spd_perio`

**作用**: 支撑期脚不应有速度。

$$
r = \sum_{f \in \{L,R\}} I_{stance,f} \cdot \exp\!(-100 \cdot v_{avg,f}^2)
$$

**参数**: `delta_t = 0.02`

---

### `gait_feet_frc_support_perio`

**作用**: 支撑期脚应有足够的支撑力。

$$
r = \sum_{f \in \{L,R\}} I_{stance,f} \cdot \left(1 - \exp\!(-10 \cdot F_{avg,f}^2)\right)
$$

**参数**: `delta_t = 0.02`

---

## 符号说明

| 符号 | 含义 |
|------|------|
| $\mathbf{v}_{cmd}$ | 命令速度向量 `command_generator.command` |
| $\mathbf{v}_{yaw}$ | yaw 坐标系下的实际速度 |
| $\omega$ | 角速度 |
| $\mathbf{g}_{body}$ | body 坐标系下的重力投影 `projected_gravity_b` |
| $\boldsymbol{\tau}$ | 关节力矩 `applied_torque` |
| $\dot{\mathbf{q}}$ | 关节速度 `joint_vel` |
| $\ddot{\mathbf{q}}$ | 关节加速度 `joint_acc` |
| $\mathbf{q}$ | 关节位置 `joint_pos` |
| $a$ | 动作 (policy 输出) |
| $\mathbf{F}$ | 接触力 `net_forces_w` |
| $\phi$ | 步态相位 $\in [0, 1]$ |
| $r_{air}$ | 摆动期占比 (air ratio) |
| $\mathbb{1}[\cdot]$ | 指示函数 |
