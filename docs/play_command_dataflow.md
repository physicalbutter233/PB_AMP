# Play 脚本中指令作用于机器人运动的数据流详解

## 概述

本文档详细说明 `play.py` 脚本中指令（命令）如何生成、传递并最终作用于机器人运动的完整数据流。

## 数据流架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        play.py 主循环                            │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  1. 初始化阶段 (play函数开始)          │
        └───────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
┌───────────────┐                    ┌──────────────────┐
│ 环境配置      │                    │ 策略模型加载      │
│ env_cfg       │                    │ runner.load()    │
│ - commands    │                    │ policy           │
│ - ranges      │                    └──────────────────┘
└───────────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│  2. 环境创建 (RobanEnv)                       │
│  - command_generator: UniformVelocityCommand │
│  - 自动生成速度指令                            │
└───────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  3. 主循环 (while simulation_app...)   │
        └───────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
┌──────────────────┐                  ┌──────────────────┐
│ 获取观察值        │                  │ 策略推理          │
│ env.get_obs()    │                  │ policy(obs)       │
└──────────────────┘                  └──────────────────┘
        │                                       │
        │                                       ▼
        │                              ┌──────────────────┐
        │                              │ 动作输出          │
        │                              │ actions [21]      │
        │                              └──────────────────┘
        │                                       │
        └───────────────────┬───────────────────┘
                            ▼
                ┌───────────────────────┐
                │ 环境步进              │
                │ env.step(actions)     │
                └───────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
┌──────────────────┐                  ┌──────────────────┐
│ 动作处理          │                  │ 命令更新          │
│ - 延迟缓冲        │                  │ command_generator│
│ - 缩放            │                  │ .compute()       │
│ - 应用            │                  └──────────────────┘
└──────────────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│  4. 物理仿真执行                                │
│  - robot.set_joint_position_target()          │
│  - sim.step() (多次子步)                       │
│  - 更新机器人状态                               │
└───────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  5. 观察值计算 (下一帧)                │
        │  - 包含更新的命令信息                   │
        └───────────────────────────────────────┘
                            │
                            └─── 循环继续
```

## 详细数据流说明

### 阶段 1: 初始化配置

**位置**: `play.py` 第 73-95 行

```python
# 命令配置
env_cfg.commands.rel_standing_envs = 0.0
env_cfg.commands.ranges.lin_vel_x = (1.0, 1.0)  # 固定前向速度 1.0 m/s
env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)  # 无侧向速度
```

**数据**:
- `lin_vel_x`: 前向速度范围 (1.0, 1.0) → 固定为 1.0 m/s
- `lin_vel_y`: 侧向速度范围 (0.0, 0.0) → 固定为 0.0 m/s
- `ang_vel_z`: 角速度范围（从 walk_cfg.py 继承，默认 -1.57 到 1.57 rad/s）

### 阶段 2: 命令生成器初始化

**位置**: `roban_envs.py` 第 82-92 行

```python
command_cfg = UniformVelocityCommandCfg(
    asset_name="robot",
    resampling_time_range=self.cfg.commands.resampling_time_range,  # (10.0, 10.0)
    rel_standing_envs=self.cfg.commands.rel_standing_envs,        # 0.0
    ranges=self.cfg.commands.ranges,                                # 速度范围
)
self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)
```

**功能**:
- `UniformVelocityCommand` 自动在指定时间间隔内重新采样速度命令
- 根据配置的范围随机生成 `[lin_vel_x, lin_vel_y, ang_vel_z]` 命令向量
- 在 play 模式下，由于范围固定，命令基本保持恒定

### 阶段 3: 观察值计算（包含命令）

**位置**: `roban_envs.py` 第 384-418 行

```python
def compute_current_observations(self):
    # ... 其他观察值 ...
    command = self.command_generator.command  # [num_envs, 3]
    
    base_obs = [
        ang_vel * self.obs_scales.ang_vel,                    # 3维: 角速度
        projected_gravity * self.obs_scales.projected_gravity, # 3维: 投影重力
        command * self.obs_scales.commands,                    # 3维: 速度命令 ⭐
        joint_pos * self.obs_scales.joint_pos,                # 21维: 关节位置
        joint_vel * self.obs_scales.joint_vel,                 # 21维: 关节速度
        action * self.obs_scales.actions,                      # 21维: 上一帧动作
    ]
    # 可选: 步态信息 (6维)
    # 总计: 72维（无步态）或 78维（有步态）
```

**命令在观察值中的位置**:
- **索引**: 观察值向量的第 6-8 维（在角速度和重力之后）
- **内容**: `[lin_vel_x, lin_vel_y, ang_vel_z]`
- **缩放**: 通过 `obs_scales.commands` 缩放（默认 1.0）

**观察值历史缓冲**:
- 单帧观察值 → `CircularBuffer` (10帧历史)
- 最终输入策略: `[num_envs, 780]` (78维 × 10帧)

### 阶段 4: 策略推理

**位置**: `play.py` 第 175-176 行

```python
with torch.inference_mode():
    actions = policy(obs)  # obs: [num_envs, 780], actions: [num_envs, 21]
```

**数据转换**:
1. **输入**: 观察值 `obs` 包含命令信息（在历史帧中）
2. **处理**: 策略网络（Actor）处理观察值
3. **输出**: 动作向量 `actions`，形状 `[num_envs, 21]`
   - 对应 21 个关节的目标位置偏移量

### 阶段 5: 动作处理与应用

**位置**: `roban_envs.py` 第 472-489 行

```python
def step(self, actions: torch.Tensor):
    # 5.1 动作延迟处理
    delayed_actions = self.action_buffer.compute(actions)
    
    # 5.2 动作裁剪
    self.action = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions)
    
    # 5.3 动作缩放和偏移
    processed_actions = self.action * self.action_scale + self.robot.data.default_joint_pos
    # action_scale = 0.25 (从配置读取)
    # default_joint_pos: 21维，各关节的默认位置
    
    # 5.4 应用动作到仿真
    for _ in range(self.cfg.sim.decimation):  # decimation = 4
        self.robot.set_joint_position_target(processed_actions)
        self.sim.step(render=False)  # 物理步进
```

**动作处理流程**:
1. **延迟缓冲**: `DelayBuffer` 可添加动作延迟（play 模式下通常禁用）
2. **裁剪**: 限制在 `[-clip_actions, clip_actions]` 范围内（默认 ±100.0）
3. **缩放**: 乘以 `action_scale` (0.25)，将归一化动作转换为实际关节角度偏移
4. **偏移**: 加上默认关节位置，得到目标关节位置
5. **应用**: 通过 PD 控制器驱动关节到达目标位置

### 阶段 6: 命令更新

**位置**: `roban_envs.py` 第 505 行

```python
self.command_generator.compute(self.step_dt)
```

**功能**:
- 每个环境步进后，命令生成器检查是否需要重新采样命令
- 如果达到重采样时间（`resampling_time_range`），生成新的速度命令
- 在 play 模式下，由于范围固定，命令变化很小

### 阶段 7: 下一帧观察值（包含新命令）

**位置**: `roban_envs.py` 第 514 行

```python
actor_obs, critic_obs = self.compute_observations()
```

**循环**:
- 新的观察值包含更新后的命令
- 命令影响策略的下一次决策
- 形成闭环控制

## 关键数据结构

### 命令向量 (Command)

```python
command = self.command_generator.command
# 形状: [num_envs, 3]
# 内容: [lin_vel_x, lin_vel_y, ang_vel_z]
# 单位: m/s, m/s, rad/s
```

### 观察值向量 (Observation)

```python
# 单帧观察值（无步态）
obs_single = [
    ang_vel[3],           # 角速度
    projected_gravity[3],  # 投影重力
    command[3],           # ⭐ 速度命令
    joint_pos[21],        # 关节位置
    joint_vel[21],        # 关节速度
    action[21],           # 上一帧动作
]
# 总计: 72维

# 历史观察值（输入策略）
obs_history = [obs_t-9, obs_t-8, ..., obs_t]
# 形状: [num_envs, 780] (78维 × 10帧，如果包含步态)
```

### 动作向量 (Action)

```python
actions = policy(obs)
# 形状: [num_envs, 21]
# 内容: 归一化的关节位置偏移量 (-1 到 1)
# 对应关节: waist(1) + left_leg(6) + right_leg(6) + left_arm(4) + right_arm(4)

processed_actions = actions * 0.25 + default_joint_pos
# 转换为实际关节目标位置（弧度）
```

## 命令如何影响运动

### 1. 直接影响：观察值中的命令

- 策略网络看到当前期望的速度命令
- 网络学习如何根据命令生成相应的动作
- 命令变化 → 观察值变化 → 策略输出变化 → 机器人运动变化

### 2. 间接影响：奖励函数（训练时）

虽然 play 模式下不计算奖励，但策略是在奖励指导下训练的：

```python
# walk_cfg.py 中的奖励配置
track_lin_vel_xy_exp = RewTerm(weight=1.2)  # 跟踪线速度
track_ang_vel_z_exp = RewTerm(weight=1.1)   # 跟踪角速度
```

训练时，策略学习：
- 当 `command = [1.0, 0.0, 0.0]` 时 → 生成前向行走动作
- 当 `command = [0.0, 0.0, 0.5]` 时 → 生成转向动作
- 当 `command = [0.5, 0.0, 0.0]` 时 → 生成慢速行走动作

### 3. 命令更新机制

```python
# 命令重采样时间范围
resampling_time_range = (10.0, 10.0)  # 每 10 秒重新采样一次

# 在 step() 中
self.command_generator.compute(self.step_dt)
# 内部逻辑：
# - 检查是否达到重采样时间
# - 如果在范围内，随机生成新命令
# - 在 play 模式下，由于范围固定，命令基本不变
```

## 键盘输入（当前实现）

**位置**: `legged_lab/utils/keyboard.py`

```python
class Keyboard:
    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.input.name == "R":
            # 按 R 键重置环境
            self.env.episode_length_buf = torch.ones_like(...) * 1e6
```

**注意**: 
- 当前键盘输入**不直接修改命令**
- 只用于重置环境（触发 episode 结束）
- 命令由 `UniformVelocityCommand` 自动生成

## 完整数据流时序图

```
时间步 t:
┌─────────────────────────────────────────────────────────┐
│ 1. 获取观察值                                            │
│    obs_t = [ang_vel, gravity, command_t, joints, ...] │
│    command_t = [1.0, 0.0, 0.0]  ← 当前命令              │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ 2. 策略推理                                              │
│    actions_t = policy(obs_t)                            │
│    actions_t = [a1, a2, ..., a21]  ← 21个关节动作      │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ 3. 动作处理                                              │
│    processed = actions_t * 0.25 + default_pos          │
│    → 转换为实际关节目标位置                               │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ 4. 物理仿真 (4个子步，decimation=4)                      │
│    for i in range(4):                                   │
│        robot.set_joint_position_target(processed)       │
│        sim.step()  ← 物理引擎更新                        │
│    → 机器人状态更新                                       │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ 5. 命令更新                                              │
│    command_generator.compute(step_dt)                   │
│    → 检查是否需要重采样命令                              │
│    → 生成 command_{t+1}                                 │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ 6. 计算下一帧观察值                                       │
│    obs_{t+1} = [..., command_{t+1}, ...]                │
│    → 循环继续                                            │
└─────────────────────────────────────────────────────────┘
```

## 总结

1. **命令生成**: 由 `UniformVelocityCommand` 自动生成，在 play 模式下基本固定
2. **命令传递**: 通过观察值向量传递给策略网络（第 6-8 维）
3. **策略响应**: 策略网络根据命令生成相应的关节动作
4. **动作执行**: 动作经过缩放和偏移后应用到机器人关节
5. **闭环反馈**: 机器人状态更新后，新的观察值（包含命令）用于下一帧决策

命令是机器人运动的"目标"，策略网络学习如何根据命令生成相应的动作序列，使机器人达到期望的运动状态。
