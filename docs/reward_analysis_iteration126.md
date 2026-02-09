# 训练日志分析（Iteration 126）- 新配置验证

基于新训练日志（iteration 126）的奖励分析，验证新配置是否生效。

## 一、训练数据

**日志数据**:
- Mean reward: 5.56
- Mean episode length: 49.59 步
- Episode_Reward/track_lin_vel_xy_exp: 0.1537
- Episode_Reward/track_ang_vel_z_exp: 0.1468
- 其他奖励项（惩罚项）总和: -0.0520

**预期配置**:
- `amp_reward_coef = 0.01`
- `amp_task_reward_lerp = 0.96` (96% Task, 4% AMP)

---

## 二、奖励计算

### 2.1 每步平均奖励

```
每步平均奖励 = 5.56 / 49.59 ≈ 0.1121
```

### 2.2 每步 Task 奖励

**Task 奖励项（累积值）**:
```
track_lin_vel_xy_exp:  0.1537
track_ang_vel_z_exp:   0.1468
─────────────────────────────
正奖励总和:             0.3005

lin_vel_z_l2:         -0.0034
ang_vel_xy_l2:        -0.0262
flat_orientation_l2:  -0.0029
energy:               -0.0000
dof_acc_l2:           -0.0102
action_rate_l2:       -0.0037
undesired_contacts:   -0.0011
feet_slide:           -0.0013
feet_force:           -0.0032
─────────────────────────────
惩罚项总和:            -0.0520

Task奖励总和（累积）:  0.3005 - 0.0520 = 0.2485
```

**每步 Task 奖励**:
```
每步Task奖励 = 0.2485 / 49.59 ≈ 0.00501
```

### 2.3 反推 AMP 奖励和配置

**假设使用新配置**: `amp_task_reward_lerp = 0.96`, `amp_reward_coef = 0.01`

**计算**:
```
最终奖励 = 0.04 × AMP奖励 + 0.96 × Task奖励
0.1121 = 0.04 × AMP奖励 + 0.96 × 0.00501
0.1121 = 0.04 × AMP奖励 + 0.00481
AMP奖励 = (0.1121 - 0.00481) / 0.04 ≈ 2.68  ❌ 不合理！
```

**问题**: AMP 奖励 2.68 远大于 `amp_reward_coef = 0.01` 的最大值（0.01），说明**训练使用的不是新配置**。

**假设使用旧配置**: `amp_task_reward_lerp = 0.7`, `amp_reward_coef = 0.3`

**计算**:
```
0.1121 = 0.3 × AMP奖励 + 0.7 × 0.00501
0.1121 = 0.3 × AMP奖励 + 0.00351
AMP奖励 = (0.1121 - 0.00351) / 0.3 ≈ 0.362
```

**验证**:
```
AMP贡献 = 0.3 × 0.362 = 0.1086
Task贡献 = 0.7 × 0.00501 = 0.00351
最终奖励 = 0.1086 + 0.00351 = 0.1121  ✅ 匹配！

AMP占比 = 0.1086 / 0.1121 ≈ 96.9%
Task占比 = 0.00351 / 0.1121 ≈ 3.1%
```

---

## 三、结论

### 3.1 配置状态

**训练使用的配置**:
- `amp_reward_coef = 0.3`（旧配置）
- `amp_task_reward_lerp = 0.7`（旧配置）

**奖励占比**:
- **AMP 占比: 96.9%**
- **Task 占比: 3.1%**

### 3.2 对比分析

| 项目 | Iteration 70 | Iteration 126 | 变化 |
|------|--------------|--------------|------|
| Mean reward | 3.43 | 5.56 | ↑ 62% |
| Episode length | 47.43 | 49.59 | ↑ 4.5% |
| Task奖励/步 | 0.00357 | 0.00501 | ↑ 40% |
| AMP占比 | ~96.5% | ~96.9% | 基本不变 |

**关键发现**:
- ❌ 新配置**仍然没有生效**（仍然使用旧配置）
- ✅ Task 奖励增加了 40%（从 0.00357 到 0.00501）
- ✅ Mean reward 增加了 62%（从 3.43 到 5.56）
- ⚠️ 但 AMP 占比仍然很高（96.9%）

---

## 四、为什么新配置没有生效？

### 4.1 可能的原因

1. **训练从检查点恢复**
   - 虽然配置参数会使用新值，但训练可能使用了 `--resume` 或 `resume=True`
   - 检查点可能保存了旧的 Discriminator 实例

2. **Python 模块缓存**
   - Python 缓存了旧的配置类
   - 需要清除 `__pycache__` 目录

3. **训练脚本在配置修改前启动**
   - 如果训练脚本在配置修改前启动，会使用旧配置

4. **Discriminator 实例已创建**
   - 如果从检查点恢复，Discriminator 可能已经用旧配置创建
   - `load_state_dict()` 只加载权重，不改变配置参数

### 4.2 验证方法

**检查训练启动日志**，应该看到：
```
================================================================================
[AmpOnPolicyRunner] AMP 配置:
  amp_reward_coef = 0.01
  amp_task_reward_lerp = 0.96
================================================================================
[Discriminator] 初始化参数:
  amp_reward_coef = 0.01
  task_reward_lerp = 0.96
================================================================================
```

**如果没有看到这些输出**，说明：
- 训练脚本使用了旧代码（没有调试输出）
- 或者配置没有正确加载

---

## 五、强制使用新配置的方法

### 方法1: 停止训练，清除缓存，重新开始

```bash
# 1. 停止当前训练

# 2. 清除 Python 缓存
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
find . -name "*.pyc" -delete

# 3. 确保配置文件正确
python3 legged_lab/scripts/check_amp_config.py

# 4. 重新开始训练（不使用检查点）
# 确保 walk_cfg.py 中 resume = False
python legged_lab/scripts/train.py --task=walk
```

### 方法2: 在代码中强制设置（临时调试）

**修改 `amp_on_policy_runner.py`**，在 Discriminator 创建后强制设置：

```python
# 在第124行之后添加
discriminator = Discriminator(...).to(self.device)

# 强制覆盖配置（仅用于调试）
discriminator.amp_reward_coef = 0.01
discriminator.task_reward_lerp = 0.96
print(f"[FORCED] amp_reward_coef = {discriminator.amp_reward_coef}")
print(f"[FORCED] task_reward_lerp = {discriminator.task_reward_lerp}")
```

### 方法3: 检查检查点文件

**检查检查点文件中是否保存了配置**:

```python
import torch
checkpoint = torch.load("logs/walk/.../model_*.pt", weights_only=False)
print("Checkpoint keys:", checkpoint.keys())
# 检查是否有 'amp_reward_coef' 或 'amp_task_reward_lerp'
```

---

## 六、如果新配置生效，预期结果

**如果使用新配置**（`coef=0.01, lerp=0.96`）:

**假设 Task 奖励 = 0.00501/步**:
```
AMP奖励最大 = 0.01
最终奖励 = 0.04 × 0.01 + 0.96 × 0.00501
        = 0.0004 + 0.00481
        = 0.00521

AMP占比 = 0.0004 / 0.00521 ≈ 7.7%
Task占比 = 0.00481 / 0.00521 ≈ 92.3%
```

**但实际 Mean reward = 5.56，每步 = 0.1121**，说明：
- 配置没有生效（使用旧配置）
- 或者有其他因素影响

---

## 七、总结

### 7.1 当前状态

- **AMP 占比: 约 96.9%**（仍然很高）
- **训练使用的可能是旧配置**（`coef=0.3, lerp=0.7`）
- **新配置可能没有生效**

### 7.2 改进

- ✅ Task 奖励增加了 40%
- ✅ Mean reward 增加了 62%
- ❌ 但 AMP 占比仍然很高

### 7.3 下一步行动

1. **检查训练启动日志**，看是否有配置打印输出
2. **如果没有看到配置输出**:
   - 停止训练
   - 清除缓存
   - 重新启动训练
3. **如果看到配置输出但值不对**:
   - 检查配置文件
   - 使用强制设置方法（方法2）

### 7.4 验证方法

**从训练日志计算**:
```
如果使用新配置（lerp=0.96, coef=0.01）:
- AMP奖励最大 = 0.01
- 如果最终奖励 = 0.1121，AMP占比应该很小（< 20%）

如果使用旧配置（lerp=0.7, coef=0.3）:
- AMP奖励可能很大（0.3-0.4）
- AMP占比会很高（> 90%）

当前情况: AMP占比 96.9%，说明使用的是旧配置
```
