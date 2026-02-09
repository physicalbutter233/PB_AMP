# 解决机器人不愿意移动的问题

当机器人不愿意移动时，通常是 AMP 奖励和 Task 奖励不平衡导致的。本文档提供多种解决方案。

## 一、问题诊断

### 1.1 症状

- ✅ Mean reward 很高（45.99）
- ✅ AMP loss 正常
- ❌ 机器人不移动或移动很少
- ❌ 速度跟踪奖励很低（track_lin_vel_xy_exp: 0.4794）

### 1.2 根本原因

**AMP 奖励占主导，Task 奖励太小**：
- AMP 奖励：~0.187/步（占主导）
- Task 奖励：~0.0014/步（太小）
- 最终奖励 = 0.3 × 0.187 + 0.7 × 0.0014 ≈ 0.057

**问题**：
- 策略只关注模仿专家动作（AMP奖励）
- 不关注任务目标（移动、速度跟踪）
- 如果专家数据中有静止动作，策略会学习静止

---

## 二、解决方案

### 方案1：增加移动相关奖励权重（推荐）

**修改文件**: `legged_lab/envs/roban/walk_cfg.py`

**当前配置**:
```python
track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.2, params={"std": 0.5})
track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.1, params={"std": 0.5})
```

**建议修改**:
```python
# 增加移动相关奖励权重
track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.0, params={"std": 0.5})  # 1.2 → 2.0
track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.8, params={"std": 0.5})      # 1.1 → 1.8
```

**效果**:
- Task 奖励增加约 0.7/步（1.2→2.0 增加 0.8，1.1→1.8 增加 0.7）
- 最终奖励中 Task 奖励占比增加
- 更强烈地激励移动

---

### 方案2：调整 AMP 和 Task 奖励平衡

**修改文件**: `legged_lab/envs/roban/walk_cfg.py`

**当前配置**:
```python
amp_reward_coef = 0.3
amp_task_reward_lerp = 0.7  # 70% Task, 30% AMP
```

**选项A：增加 Task 奖励权重**
```python
amp_task_reward_lerp = 0.9  # 90% Task, 10% AMP
```

**选项B：降低 AMP 奖励系数**
```python
amp_reward_coef = 0.1  # 降低 AMP 奖励最大值
amp_task_reward_lerp = 0.8  # 80% Task, 20% AMP
```

**选项C：完全禁用 AMP 奖励插值（仅使用 Task 奖励）**
```python
amp_task_reward_lerp = 1.0  # 100% Task, 0% AMP
```

**效果对比**:

| 配置 | Task权重 | AMP权重 | Task奖励/步 | AMP奖励/步 | 最终奖励/步 |
|------|----------|---------|-------------|------------|-------------|
| 当前 | 0.7 | 0.3 | 0.0014 | 0.187 | 0.057 |
| 选项A | 0.9 | 0.1 | 0.0014 | 0.187 | 0.169 |
| 选项B | 0.8 | 0.2 | 0.0014 | 0.062 | 0.014 |
| 选项C | 1.0 | 0.0 | 0.0014 | - | 0.0014 |

**推荐**: 选项A（`amp_task_reward_lerp = 0.9`），既保留 AMP 奖励，又增加 Task 奖励权重。

---

### 方案3：组合方案（最推荐）

**同时调整奖励权重和 AMP 平衡**:

```python
# 1. 增加移动相关奖励权重
track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.5, params={"std": 0.5})
track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5})

# 2. 调整 AMP 和 Task 奖励平衡
amp_reward_coef = 0.2  # 降低 AMP 奖励
amp_task_reward_lerp = 0.85  # 85% Task, 15% AMP
```

**效果**:
- Task 奖励增加：约 1.3/步（权重增加 1.3）
- AMP 奖励降低：0.187 → 0.125（系数降低）
- 最终奖励：0.85 × 1.3 + 0.15 × 0.125 ≈ 1.12/步
- Task 奖励占比：约 99%

---

### 方案4：添加移动奖励项

**如果上述方案不够，可以添加额外的移动奖励**:

```python
# 在 LiteRewardCfg 中添加
forward_velocity_bonus = RewTerm(
    func=mdp.lin_vel_x_l1,  # 需要实现这个函数
    weight=0.5,
    params={"threshold": 0.1}  # 速度超过 0.1 m/s 时给奖励
)
```

**或者使用现有的奖励函数**:
```python
# 添加前进速度奖励（简化版）
forward_velocity = RewTerm(
    func=lambda env: torch.clamp(env.robot.data.root_lin_vel_b[:, 0], min=0.0),
    weight=0.3
)
```

---

## 三、实施步骤

### 步骤1：备份当前配置

```bash
cp legged_lab/envs/roban/walk_cfg.py legged_lab/envs/roban/walk_cfg.py.backup
```

### 步骤2：修改配置

**推荐修改**（方案3）:

```python
# legged_lab/envs/roban/walk_cfg.py

@configclass
class LiteRewardCfg:
    # 增加移动相关奖励权重
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.5, params={"std": 0.5})  # 1.2 → 2.5
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5})      # 1.1 → 2.0
    
    # ... 其他奖励项保持不变 ...

@configclass
class RobanWalkAgentCfg(RslRlOnPolicyRunnerCfg):
    # ... 其他配置 ...
    
    # 调整 AMP 参数
    amp_reward_coef = 0.2        # 0.3 → 0.2（降低 AMP 奖励）
    amp_task_reward_lerp = 0.85   # 0.7 → 0.85（增加 Task 奖励权重）
```

### 步骤3：重新训练

```bash
# 如果是从检查点继续训练
python legged_lab/scripts/train.py --task=walk --resume

# 如果是重新开始训练
python legged_lab/scripts/train.py --task=walk
```

### 步骤4：监控训练

**关注指标**:
- `Episode_Reward/track_lin_vel_xy_exp`: 应该增加（目标 > 1.0）
- `Mean reward`: 可能会降低（正常，因为 AMP 奖励降低）
- 机器人实际移动速度：应该增加

---

## 四、参数调整指南

### 4.1 奖励权重调整

| 参数 | 当前值 | 建议范围 | 说明 |
|------|--------|----------|------|
| `track_lin_vel_xy_exp.weight` | 1.2 | 2.0-3.0 | 移动速度跟踪奖励 |
| `track_ang_vel_z_exp.weight` | 1.1 | 1.5-2.5 | 角速度跟踪奖励 |

**调整原则**:
- ✅ 如果机器人移动太少：增加权重（2.0-3.0）
- ✅ 如果机器人移动但不稳定：保持或略微降低（1.5-2.0）
- ✅ 如果机器人移动正常：保持当前值

### 4.2 AMP 参数调整

| 参数 | 当前值 | 建议范围 | 说明 |
|------|--------|----------|------|
| `amp_reward_coef` | 0.3 | 0.1-0.3 | AMP 奖励最大值 |
| `amp_task_reward_lerp` | 0.7 | 0.8-0.95 | Task 奖励权重 |

**调整原则**:
- ✅ 如果机器人不移动：增加 `amp_task_reward_lerp`（0.85-0.95）
- ✅ 如果动作不自然：降低 `amp_task_reward_lerp`（0.7-0.8）
- ✅ 如果 AMP 奖励太高：降低 `amp_reward_coef`（0.1-0.2）

---

## 五、其他检查项

### 5.1 检查专家数据

**问题**: 专家数据可能包含静止动作

**检查方法**:
```bash
# 可视化专家动作
python legged_lab/scripts/play_amp_animation.py --task=walk --num_envs=1
```

**如果专家数据有问题**:
- 重新生成专家数据，确保包含移动动作
- 或者使用不同的专家数据文件

### 5.2 检查命令生成

**问题**: 命令速度可能为 0

**检查配置**:
```python
# walk_cfg.py
env_cfg.commands.ranges.lin_vel_x = (0.0, 1.0)  # 确保最小值 > 0
env_cfg.commands.ranges.lin_vel_y = (-0.5, 0.5)
env_cfg.commands.ranges.ang_vel_z = (-1.57, 1.57)
```

### 5.3 检查动作空间

**问题**: 动作缩放可能太小

**检查配置**:
```python
# walk_cfg.py
robot: RobotCfg = RobotCfg(
    action_scale=0.25,  # 如果太小，可以增加到 0.3-0.4
    ...
)
```

---

## 六、调试技巧

### 6.1 监控关键指标

**训练时关注**:
```python
# 在 TensorBoard 或日志中查看
Episode_Reward/track_lin_vel_xy_exp  # 应该 > 1.0
Episode_Reward/track_ang_vel_z_exp   # 应该 > 0.5
Mean reward                           # 可能会降低（正常）
Mean episode length                   # 应该保持或增加
```

### 6.2 测试策略

**训练一段时间后测试**:
```bash
python legged_lab/scripts/play.py --task=walk --num_envs=1
```

**观察**:
- ✅ 机器人是否移动
- ✅ 移动速度是否合理
- ✅ 动作是否自然

### 6.3 渐进式调整

**不要一次性大幅修改**:
1. 先增加移动奖励权重（1.2 → 2.0）
2. 训练一段时间，观察效果
3. 如果还不够，再调整 AMP 参数
4. 逐步微调，找到最佳平衡点

---

## 七、常见问题

### Q1: 修改后 Mean reward 降低了，正常吗？

**A**: 正常。因为降低了 AMP 奖励，总奖励会降低。关键是：
- ✅ 机器人是否开始移动
- ✅ 速度跟踪奖励是否增加
- ✅ 动作是否自然

### Q2: 调整后训练不稳定怎么办？

**A**: 
1. 降低学习率：`learning_rate=5e-4`（从 1e-3 降低）
2. 增加训练步数：`num_learning_epochs=10`（从 5 增加）
3. 逐步调整参数，不要一次性大幅修改

### Q3: 如何平衡 AMP 和 Task 奖励？

**A**: 
- 早期训练：`amp_task_reward_lerp = 0.7-0.8`（更多 AMP，学习动作风格）
- 中期训练：`amp_task_reward_lerp = 0.85-0.9`（更多 Task，学习任务目标）
- 后期训练：`amp_task_reward_lerp = 0.9-0.95`（主要 Task，微调性能）

---

## 八、总结

### 推荐配置（解决不移动问题）

```python
# 1. 增加移动奖励权重
track_lin_vel_xy_exp = RewTerm(..., weight=2.5)  # 1.2 → 2.5
track_ang_vel_z_exp = RewTerm(..., weight=2.0)    # 1.1 → 2.0

# 2. 调整 AMP 平衡
amp_reward_coef = 0.2        # 0.3 → 0.2
amp_task_reward_lerp = 0.85  # 0.7 → 0.85
```

### 预期效果

- ✅ Task 奖励增加：约 1.3/步
- ✅ 最终奖励中 Task 占比：约 99%
- ✅ 机器人开始移动
- ✅ 速度跟踪奖励增加

### 下一步

1. 应用推荐配置
2. 重新训练
3. 监控训练指标
4. 根据效果微调参数
