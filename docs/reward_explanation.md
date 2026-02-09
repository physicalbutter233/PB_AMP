# AMP 训练中奖励计算机制详解

本文档解释为什么各项奖励项很小，但 Mean reward 却很高的原因。

## 一、问题现象

从训练日志可以看到：
- **Mean reward**: 45.99（很高）
- **各项 Episode_Reward/xxx**: 都很小（0.4794, 0.6569, -0.0045 等）
- **Mean episode length**: 802.98 步

**疑问**: 各项奖励项加起来远小于 45.99，为什么总奖励这么高？

---

## 二、关键机制：AMP 奖励替换

### 2.1 奖励处理流程

在 AMP 训练中，**环境返回的原始 task reward 会被替换成 AMP 奖励**：

**代码位置**: `rsl_rl/rsl_rl/runners/amp_on_policy_runner.py` 第 271-273 行

```python
# 1. 环境返回原始 task reward
obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))

# 2. ⭐ 关键：task reward 被替换成 AMP 奖励
rewards = self.alg.discriminator.predict_amp_reward(
    amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer
)[0]

# 3. 替换后的 rewards 用于训练
self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)
```

### 2.2 AMP 奖励计算公式

**代码位置**: `rsl_rl/rsl_rl/modules/discriminator.py` 第 102-128 行

```python
def predict_amp_reward(self, state, next_state, task_reward, normalizer=None):
    # 1. 计算判别器输出
    d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
    
    # 2. 计算 AMP 奖励（基于判别器输出）
    disc_reward = self.amp_reward_coef * torch.clamp(
        1 - (1 / 4) * torch.square(d - 1), 
        min=0
    )
    
    # 3. ⭐ 与 task reward 线性插值
    if self.task_reward_lerp > 0:
        reward = self._lerp_reward(disc_reward, task_reward.unsqueeze(-1))
    
    return reward.squeeze(), d

def _lerp_reward(self, disc_r, task_r):
    # 线性插值公式
    r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
    return r
```

### 2.3 配置参数

**代码位置**: `legged_lab/envs/roban/walk_cfg.py` 第 322-325 行

```python
amp_reward_coef = 0.3        # AMP 奖励系数
amp_task_reward_lerp = 0.7    # Task reward 插值权重
```

**含义**:
- `amp_reward_coef = 0.3`: AMP 奖励的最大值约为 0.3（当 d=1 时）
- `amp_task_reward_lerp = 0.7`: 最终奖励 = 30% AMP奖励 + 70% Task奖励

---

## 三、奖励计算详解

### 3.1 每步奖励计算

**公式**:
```
最终奖励 = 0.3 * AMP奖励 + 0.7 * Task奖励
```

其中：
- **Task奖励** = 各项奖励项的总和（track_lin_vel, track_ang_vel, energy 等）
- **AMP奖励** = `0.3 * clamp(1 - 0.25 * (d-1)², min=0)`
  - `d` 是判别器输出（logits）
  - 当 `d=1` 时，AMP奖励最大 = 0.3
  - 当 `d` 远离 1 时，AMP奖励减小

### 3.2 从日志数据估算

**各项 Task 奖励（每步）**:
```
track_lin_vel_xy_exp:  0.4794 / 803 ≈ 0.0006  (每步)
track_ang_vel_z_exp:   0.6569 / 803 ≈ 0.0008  (每步)
lin_vel_z_l2:         -0.0045 / 803 ≈ -0.000006
ang_vel_xy_l2:        -0.0839 / 803 ≈ -0.0001
flat_orientation_l2:  -0.0090 / 803 ≈ -0.00001
energy:               -0.0003 / 803 ≈ -0.0000004
dof_acc_l2:           -0.0198 / 803 ≈ -0.00002
action_rate_l2:       -0.0294 / 803 ≈ -0.00004
undesired_contacts:   -0.0005 / 803 ≈ -0.0000006
feet_slide:           -0.0071 / 803 ≈ -0.000009
feet_force:           -0.0191 / 803 ≈ -0.00002
─────────────────────────────────────────────
Task奖励总和（每步）: ≈ 0.0014
```

**Mean reward（每步）**:
```
45.99 / 803 ≈ 0.057  (每步)
```

**计算验证**:
```
最终奖励 = 0.3 * AMP奖励 + 0.7 * 0.0014
0.057 = 0.3 * AMP奖励 + 0.00098
AMP奖励 ≈ (0.057 - 0.00098) / 0.3 ≈ 0.187
```

**结论**: AMP 奖励约为 0.187（每步），远大于 Task 奖励（0.0014），所以最终奖励主要由 AMP 奖励贡献。

---

## 四、为什么 AMP 奖励这么大？

### 4.1 AMP 奖励的范围

**理论最大值**:
- 当判别器输出 `d = 1` 时（完美匹配专家）
- `AMP奖励 = 0.3 * clamp(1 - 0.25 * (1-1)², min=0) = 0.3`

**实际值**:
- 从估算看，AMP奖励约为 0.187
- 说明判别器输出 `d` 接近 1（策略动作与专家动作相似）

### 4.2 AMP 奖励的意义

**AMP 奖励的作用**:
- ✅ **鼓励模仿专家动作**: 当策略动作与专家动作相似时，获得高奖励
- ✅ **学习自然运动模式**: 不依赖具体的任务奖励，而是学习整体的运动风格
- ✅ **提供密集奖励信号**: 每步都有奖励，不像任务奖励可能稀疏

**为什么比 Task 奖励大**:
- Task 奖励各项都有权重限制（如 track_lin_vel 权重 1.2）
- AMP 奖励是判别器直接输出的，没有这些限制
- AMP 奖励范围是 [0, 0.3]，而 Task 奖励各项都很小

---

## 五、日志中的显示差异

### 5.1 Episode_Reward/xxx

**含义**: 各项 Task 奖励在 episode 结束时的**累积值**

**计算方式**:
- 每步计算各项奖励项
- 在 episode 结束时累加
- 显示的是整个 episode 的累积值

**示例**:
```
Episode_Reward/track_lin_vel_xy_exp: 0.4794
```
表示整个 episode（803 步）中，`track_lin_vel_xy_exp` 奖励项的累积值。

### 5.2 Mean reward

**含义**: 实际用于训练的奖励（经过 AMP 处理）在 episode 结束时的**累积值**

**计算方式**:
- 每步：`最终奖励 = 0.3 * AMP奖励 + 0.7 * Task奖励`
- 在 episode 结束时累加所有步的最终奖励
- 显示的是整个 episode 的累积值

**示例**:
```
Mean reward: 45.99
```
表示整个 episode（803 步）中，所有步的最终奖励的累积值。

### 5.3 为什么差异这么大？

**原因**:
1. **AMP 奖励占主导**: 最终奖励 = 30% AMP + 70% Task，但 AMP 奖励（0.187）远大于 Task 奖励（0.0014）
2. **累积效应**: 803 步累积，即使每步奖励不大，累积起来也会很大
3. **日志显示的是累积值**: 不是平均值，而是整个 episode 的总和

---

## 六、验证计算

### 6.1 从日志数据验证

**已知数据**:
- Mean reward: 45.99
- Mean episode length: 802.98 步
- 各项 Episode_Reward 的累积值

**计算每步平均奖励**:
```
每步平均奖励 = 45.99 / 802.98 ≈ 0.057
```

**计算每步 Task 奖励**:
```
每步 Task 奖励 ≈ (0.4794 + 0.6569 - 0.0045 - ...) / 802.98
                ≈ 0.97 / 802.98
                ≈ 0.0012
```

**计算每步 AMP 奖励**:
```
0.057 = 0.3 * AMP奖励 + 0.7 * 0.0012
AMP奖励 = (0.057 - 0.00084) / 0.3 ≈ 0.187
```

**验证**:
- ✅ AMP 奖励（0.187）远大于 Task 奖励（0.0012）
- ✅ 最终奖励主要由 AMP 奖励贡献
- ✅ 累积 803 步后，总奖励 ≈ 0.057 * 803 ≈ 45.77，接近 45.99

---

## 七、总结

### 7.1 关键点

1. **AMP 训练中，Task 奖励被替换成 AMP 奖励**
   - 替换公式：`最终奖励 = 0.3 * AMP奖励 + 0.7 * Task奖励`

2. **AMP 奖励远大于 Task 奖励**
   - AMP 奖励范围：[0, 0.3]
   - Task 奖励总和：约 0.0014（每步）
   - 实际 AMP 奖励：约 0.187（每步）

3. **日志显示的是累积值**
   - `Episode_Reward/xxx`: Task 奖励的累积值
   - `Mean reward`: 最终奖励的累积值
   - 不是平均值，而是整个 episode 的总和

4. **Mean reward 高的原因**
   - AMP 奖励本身较大（0.187/步）
   - Episode 较长（803 步）
   - 累积效应：0.057 * 803 ≈ 45.77

### 7.2 配置参数影响

| 参数 | 值 | 影响 |
|------|-----|------|
| `amp_reward_coef` | 0.3 | AMP 奖励的最大值 |
| `amp_task_reward_lerp` | 0.7 | Task 奖励在最终奖励中的权重 |
| Episode 长度 | 803 步 | 累积奖励的倍数 |

### 7.3 训练建议

- ✅ **正常现象**: Mean reward 高是正常的，说明 AMP 奖励工作正常
- ✅ **关注 AMP loss**: 如果 `Mean amp loss` 很小，说明判别器训练良好
- ✅ **关注各项 Task 奖励**: 虽然小，但它们是任务相关的，仍然重要
- ✅ **平衡 AMP 和 Task**: 通过 `amp_task_reward_lerp` 调整两者的权重

---

## 八、相关代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| AMP 奖励替换 | `rsl_rl/rsl_rl/runners/amp_on_policy_runner.py` | 271-273 |
| AMP 奖励计算 | `rsl_rl/rsl_rl/modules/discriminator.py` | 102-141 |
| AMP 配置参数 | `legged_lab/envs/roban/walk_cfg.py` | 322-325 |
| Task 奖励配置 | `legged_lab/envs/roban/walk_cfg.py` | 59-110 |
| 奖励累积记录 | `rsl_rl/rsl_rl/runners/amp_on_policy_runner.py` | 292-301 |
