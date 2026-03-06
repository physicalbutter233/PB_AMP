# 配置 AMP 奖励占比为 4%

本文档说明如何配置使 AMP 奖励占比只有 4%，Task 奖励占比 96%。

## 一、目标

- **AMP 奖励占比**: 4%
- **Task 奖励占比**: 96%

## 二、当前情况分析

**基于训练日志（iteration 1590）**:
- 最终奖励/步: 0.0378
- Task奖励/步: 0.00192
- AMP奖励/步: 约 0.12-0.24（取决于配置）

**当前占比**: AMP 约 95-96%，Task 约 4-5%

## 三、配置计算

### 3.1 目标奖励分配

**如果最终奖励保持 0.0378/步**:
- AMP贡献 = 0.0378 × 0.04 = 0.001512
- Task贡献 = 0.0378 × 0.96 = 0.036288

### 3.2 需要的配置

**公式**:
```
最终奖励 = (1 - lerp) × AMP奖励 + lerp × Task奖励
0.0378 = (1 - lerp) × AMP奖励 + lerp × 0.00192

AMP贡献 = (1 - lerp) × AMP奖励 = 0.001512
Task贡献 = lerp × 0.00192 = 0.036288
```

**计算 lerp**:
```
lerp × 0.00192 = 0.036288
lerp = 0.036288 / 0.00192 = 18.9  ❌ 不可能（lerp ≤ 1.0）
```

**问题**: Task 奖励太小（0.00192），即使权重 100%，贡献也只有 0.00192，无法达到 0.036288。

### 3.3 解决方案

**需要同时增加 Task 奖励本身的值**，通过增加移动奖励权重。

**假设增加移动奖励权重后，Task 奖励增加到 0.003/步**:

**计算 lerp**:
```
lerp × 0.003 = 0.036288
lerp = 0.036288 / 0.003 = 12.096  ❌ 仍然不可能
```

**假设 Task 奖励增加到 0.01/步**:
```
lerp × 0.01 = 0.036288
lerp = 0.036288 / 0.01 = 3.6288  ❌ 仍然不可能
```

**假设 Task 奖励增加到 0.0378/步（等于最终奖励）**:
```
lerp × 0.0378 = 0.036288
lerp = 0.036288 / 0.0378 = 0.96
```

**同时计算 AMP 奖励**:
```
(1 - 0.96) × AMP奖励 = 0.001512
0.04 × AMP奖励 = 0.001512
AMP奖励 = 0.0378
```

**需要的配置**:
- `amp_task_reward_lerp = 0.96` (96% Task, 4% AMP)
- Task 奖励需要增加到约 0.0378/步（当前是 0.00192，需要增加约 19.7 倍）
- `amp_reward_coef` 需要调整，使 AMP 奖励约 0.0378

### 3.4 实际配置方案

**方案1: 大幅增加移动奖励权重**

```python
# 增加移动奖励权重，使 Task 奖励增加
track_lin_vel_xy_exp.weight = 10.0  # 2.5 → 10.0（增加 4 倍）
track_ang_vel_z_exp.weight = 8.0   # 2.0 → 8.0（增加 4 倍）

# 调整 AMP 参数
amp_task_reward_lerp = 0.96  # 96% Task, 4% AMP
amp_reward_coef = 0.1        # 降低 AMP 奖励最大值
```

**预期效果**:
- Task 奖励: 0.00192 × 4 ≈ 0.00768/步
- 如果最终奖励保持 0.0378:
  - Task贡献 = 0.96 × 0.00768 = 0.00737（19.5%）
  - AMP贡献 = 0.04 × AMP奖励 = 0.03043
  - AMP奖励 = 0.7608（太大）

**问题**: 如果最终奖励保持 0.0378，AMP 奖励需要约 0.76，但 `amp_reward_coef=0.1` 时最大只有 0.1。

### 3.5 重新设计：降低最终奖励

**如果降低最终奖励到合理范围**:

**假设最终奖励 = 0.01/步**:
- AMP贡献 = 0.01 × 0.04 = 0.0004
- Task贡献 = 0.01 × 0.96 = 0.0096

**如果 Task 奖励 = 0.01/步**:
```
lerp × 0.01 = 0.0096
lerp = 0.96
```

**如果 AMP 奖励 = 0.01/步**:
```
(1 - 0.96) × 0.01 = 0.0004  ✅ 匹配
```

**需要的配置**:
- `amp_task_reward_lerp = 0.96`
- Task 奖励需要增加到 0.01/步（当前 0.00192，需要增加约 5.2 倍）
- `amp_reward_coef` 需要使 AMP 奖励约 0.01

## 四、推荐配置

### 4.1 配置参数

```python
# 1. 大幅增加移动奖励权重（使 Task 奖励增加）
track_lin_vel_xy_exp = RewTerm(
    func=mdp.track_lin_vel_xy_yaw_frame_exp, 
    weight=10.0,  # 2.5 → 10.0（增加 4 倍）
    params={"std": 0.5}
)
track_ang_vel_z_exp = RewTerm(
    func=mdp.track_ang_vel_z_world_exp, 
    weight=8.0,   # 2.0 → 8.0（增加 4 倍）
    params={"std": 0.5}
)

# 2. 调整 AMP 参数（使 AMP 占比 4%）
amp_reward_coef = 0.01        # 0.04 → 0.01（降低 AMP 奖励）
amp_task_reward_lerp = 0.96   # 1.0 → 0.96（96% Task, 4% AMP）
```

### 4.2 预期效果

**假设 Task 奖励增加到 0.01/步**（通过增加权重）:
- Task贡献 = 0.96 × 0.01 = 0.0096
- AMP贡献 = 0.04 × 0.01 = 0.0004
- 最终奖励 = 0.01
- **Task占比 = 96%** ✅
- **AMP占比 = 4%** ✅

## 五、实施步骤

### 步骤1: 修改配置文件

```python
# legged_lab/envs/roban/walk_cfg.py

@configclass
class LiteRewardCfg:
    # 大幅增加移动奖励权重
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp, 
        weight=10.0,  # 从 2.5 增加到 10.0
        params={"std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=8.0,   # 从 2.0 增加到 8.0
        params={"std": 0.5}
    )
    # ... 其他奖励项保持不变 ...

@configclass
class RobanWalkAgentCfg(RslRlOnPolicyRunnerCfg):
    # ... 其他配置 ...
    
    # 调整 AMP 参数，使 AMP 占比 4%
    amp_reward_coef = 0.01        # 降低 AMP 奖励最大值
    amp_task_reward_lerp = 0.96   # 96% Task, 4% AMP
```

### 步骤2: 重新训练

```bash
python legged_lab/scripts/train.py --task=walk --resume
```

### 步骤3: 监控训练

**关注指标**:
- `Episode_Reward/track_lin_vel_xy_exp`: 应该大幅增加（目标 > 3.0）
- `Mean reward`: 可能会降低（正常，因为 AMP 奖励降低）
- 计算 AMP 占比: `(1 - amp_task_reward_lerp) × AMP奖励 / 最终奖励`

## 六、验证计算

### 6.1 训练后验证

**从训练日志计算**:
```
每步平均奖励 = Mean reward / Mean episode length
每步Task奖励 = Episode_Reward总和 / Mean episode length
每步AMP奖励 = (每步平均奖励 - amp_task_reward_lerp × 每步Task奖励) / (1 - amp_task_reward_lerp)

AMP占比 = (1 - amp_task_reward_lerp) × 每步AMP奖励 / 每步平均奖励
Task占比 = amp_task_reward_lerp × 每步Task奖励 / 每步平均奖励
```

### 6.2 目标值

- **AMP占比**: 约 4%
- **Task占比**: 约 96%
- **移动奖励**: `track_lin_vel_xy_exp` > 3.0

## 七、注意事项

1. **大幅增加奖励权重可能导致训练不稳定**
   - 建议逐步增加（先到 5.0，再到 10.0）
   - 监控训练稳定性

2. **降低 AMP 奖励可能影响动作自然性**
   - 如果动作变得不自然，可以略微增加 `amp_reward_coef`（如 0.02）

3. **最终奖励可能会变化**
   - 如果最终奖励降低，是正常的
   - 关键是 AMP 占比达到 4%

4. **可能需要调整学习率**
   - 如果训练不稳定，可以降低学习率
   - `learning_rate = 5e-4`（从 1e-3 降低）

## 八、总结

**推荐配置**:
- `track_lin_vel_xy_exp.weight = 10.0`
- `track_ang_vel_z_exp.weight = 8.0`
- `amp_reward_coef = 0.01`
- `amp_task_reward_lerp = 0.96`

**预期效果**:
- AMP 占比: 4%
- Task 占比: 96%
- 机器人应该更积极地移动
