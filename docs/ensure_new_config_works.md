# 确保新配置生效的完整指南

本文档提供系统性的方法来确保新配置（`amp_reward_coef=0.01, amp_task_reward_lerp=0.96`）正确生效。

## 一、配置加载流程分析

### 1.1 配置加载路径

```
walk_cfg.py (配置文件)
    ↓
task_registry.get_cfgs() (读取配置)
    ↓
agent_cfg.to_dict() (转换为字典)
    ↓
AmpOnPolicyRunner.__init__(train_cfg) (传入 runner)
    ↓
Discriminator.__init__(amp_reward_coef, task_reward_lerp) (创建判别器)
```

### 1.2 关键代码位置

**配置读取**:
- `legged_lab/scripts/train.py` 第67行: `env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)`
- `legged_lab/scripts/train.py` 第96行: `runner = runner_class(env, agent_cfg.to_dict(), ...)`

**配置使用**:
- `rsl_rl/rsl_rl/runners/amp_on_policy_runner.py` 第120行: `train_cfg["amp_reward_coef"]`
- `rsl_rl/rsl_rl/runners/amp_on_policy_runner.py` 第123行: `train_cfg["amp_task_reward_lerp"]`

**Discriminator 初始化**:
- `rsl_rl/rsl_rl/modules/discriminator.py` 第42行: `__init__(..., amp_reward_coef, ..., task_reward_lerp)`
- 第48行: `self.amp_reward_coef = amp_reward_coef`
- 第61行: `self.task_reward_lerp = task_reward_lerp`

### 1.3 检查点恢复的影响

**重要发现**: `load_state_dict()` **不会覆盖配置参数**！

- `rsl_rl/rsl_rl/runners/amp_on_policy_runner.py` 第483行: `self.alg.discriminator.load_state_dict(...)`
- `load_state_dict()` 只加载模型权重（`trunk`, `amp_linear`），**不会改变** `amp_reward_coef` 和 `task_reward_lerp`
- 这两个参数在 `__init__` 时设置，之后不会改变

**结论**: 即使从检查点恢复，Discriminator 也会使用**新配置**中的 `amp_reward_coef` 和 `amp_task_reward_lerp`！

---

## 二、为什么新配置可能没有生效？

### 2.1 可能的原因

1. **配置文件没有保存**
   - 修改后没有保存文件
   - 文件被其他进程覆盖

2. **Python 模块缓存**
   - Python 缓存了旧的配置类
   - 需要重启 Python 进程

3. **训练脚本使用了不同的配置**
   - 使用了不同的配置文件
   - 使用了命令行参数覆盖

4. **检查点中的配置被误读**
   - 虽然 `load_state_dict()` 不会覆盖配置，但可能有其他逻辑

---

## 三、验证配置是否生效

### 3.1 方法1: 使用验证脚本

**运行验证脚本**:
```bash
python legged_lab/scripts/verify_amp_config.py
```

**预期输出**:
```
AMP 配置验证
================================================================================

1. 从配置文件读取的值:
   amp_reward_coef = 0.01
   amp_task_reward_lerp = 0.96
   track_lin_vel_xy_exp.weight = 10.0
   track_ang_vel_z_exp.weight = 8.0

2. 转换为字典后的值:
   agent_dict['amp_reward_coef'] = 0.01
   agent_dict['amp_task_reward_lerp'] = 0.96

3. 配置验证:
   ✅ amp_reward_coef = 0.01 (正确)
   ✅ amp_task_reward_lerp = 0.96 (正确)
```

### 3.2 方法2: 在训练脚本中添加调试输出

**修改 `train.py`**:
```python
# 在 train.py 第96行之后添加
runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

# 添加调试输出
print("=" * 80)
print("AMP 配置验证（训练时）")
print("=" * 80)
print(f"amp_reward_coef = {agent_cfg.amp_reward_coef}")
print(f"amp_task_reward_lerp = {agent_cfg.amp_task_reward_lerp}")
print(f"Discriminator.amp_reward_coef = {runner.alg.discriminator.amp_reward_coef}")
print(f"Discriminator.task_reward_lerp = {runner.alg.discriminator.task_reward_lerp}")
print("=" * 80)
```

### 3.3 方法3: 在 Discriminator 中添加调试输出

**修改 `discriminator.py`**:
```python
# 在 discriminator.py 第42行 __init__ 方法中添加
def __init__(self, input_dim, amp_reward_coef, hidden_layer_sizes, device, task_reward_lerp=0.0):
    super().__init__()
    
    # 添加调试输出
    print(f"[Discriminator] 初始化参数:")
    print(f"  amp_reward_coef = {amp_reward_coef}")
    print(f"  task_reward_lerp = {task_reward_lerp}")
    
    self.device = device
    self.input_dim = input_dim
    self.amp_reward_coef = amp_reward_coef
    # ... 其余代码
```

---

## 四、确保新配置生效的步骤

### 步骤1: 确认配置文件已保存

```bash
# 检查配置文件
grep -A 2 "amp_reward_coef" legged_lab/envs/roban/walk_cfg.py
grep -A 2 "amp_task_reward_lerp" legged_lab/envs/roban/walk_cfg.py
```

**预期输出**:
```
    amp_reward_coef = 0.01        # 大幅降低 AMP 奖励最大值
    amp_motion_files = ["legged_lab/envs/roban/datasets/motion_amp_expert/walk_pb_easy.txt"]
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.96   # 96% Task, 4% AMP
```

### 步骤2: 运行验证脚本

```bash
python legged_lab/scripts/verify_amp_config.py
```

**如果验证失败**:
- 检查配置文件是否正确
- 检查 Python 模块缓存（删除 `__pycache__` 目录）

### 步骤3: 清除 Python 缓存

```bash
# 清除所有 Python 缓存
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
find . -name "*.pyc" -delete
```

### 步骤4: 重新开始训练（不使用检查点）

**重要**: 如果从检查点恢复，虽然配置参数会使用新值，但**模型权重**是旧的，可能需要重新训练。

**方法1: 不使用 --resume**
```bash
# 确保配置文件中的 resume = False
# 然后运行
python legged_lab/scripts/train.py --task=walk
```

**方法2: 使用新的 run_name**
```python
# 在 walk_cfg.py 中
run_name = "amp_4percent"  # 使用新的 run_name，避免加载旧检查点
```

### 步骤5: 在训练时验证配置

**添加调试输出到训练脚本**（见方法2）

**或者修改 `amp_on_policy_runner.py`**:
```python
# 在第124行之后添加
discriminator = Discriminator(...).to(self.device)

# 添加验证输出
print(f"[AmpOnPolicyRunner] Discriminator 配置:")
print(f"  amp_reward_coef = {discriminator.amp_reward_coef}")
print(f"  task_reward_lerp = {discriminator.task_reward_lerp}")
```

---

## 五、从训练日志验证配置

### 5.1 计算 AMP 占比

**从训练日志计算**:
```
每步平均奖励 = Mean reward / Mean episode length
每步Task奖励 = Episode_Reward总和 / Mean episode length
每步AMP奖励 = (每步平均奖励 - lerp × 每步Task奖励) / (1 - lerp)

AMP占比 = (1 - lerp) × 每步AMP奖励 / 每步平均奖励
```

**如果使用新配置**（`lerp=0.96, coef=0.01`）:
- AMP 奖励最大 = 0.01
- 如果最终奖励 = 0.01，AMP 占比应该 ≈ 4%

### 5.2 预期结果

**使用新配置后，应该看到**:
- AMP 占比 ≈ 4%
- Task 占比 ≈ 96%
- Mean reward 可能会降低（因为 AMP 奖励降低）

---

## 六、常见问题排查

### Q1: 配置文件已修改，但训练仍使用旧配置

**排查步骤**:
1. ✅ 确认文件已保存
2. ✅ 清除 Python 缓存
3. ✅ 重启训练脚本
4. ✅ 运行验证脚本

### Q2: 从检查点恢复后，配置没有生效

**原因**: `load_state_dict()` 不会覆盖配置参数，所以配置应该会生效。

**验证**:
- 添加调试输出，检查 `discriminator.amp_reward_coef` 的值
- 从训练日志计算 AMP 占比

### Q3: 训练日志显示 AMP 占比仍然很高

**可能原因**:
1. 配置没有正确加载（检查步骤1-3）
2. Task 奖励太小，即使权重96%，贡献仍然很小
3. 需要进一步增加移动奖励权重

**解决方案**:
- 验证配置是否正确加载
- 进一步增加移动奖励权重（如 15.0 和 12.0）
- 进一步降低 `amp_reward_coef`（如 0.005）

---

## 七、强制使用新配置的方法

### 方法1: 修改 Discriminator 初始化（临时调试）

**在 `amp_on_policy_runner.py` 中强制设置**:
```python
# 第118-124行
discriminator = Discriminator(
    amp_data.observation_dim * 2,
    train_cfg["amp_reward_coef"],
    train_cfg["amp_discr_hidden_dims"],
    device,
    train_cfg["amp_task_reward_lerp"],
).to(self.device)

# 强制覆盖（仅用于调试）
discriminator.amp_reward_coef = 0.01
discriminator.task_reward_lerp = 0.96
print(f"[FORCED] amp_reward_coef = {discriminator.amp_reward_coef}")
print(f"[FORCED] task_reward_lerp = {discriminator.task_reward_lerp}")
```

### 方法2: 在 predict_amp_reward 中添加日志

**修改 `discriminator.py`**:
```python
def predict_amp_reward(self, state, next_state, task_reward, normalizer=None):
    # 添加日志（每1000步打印一次）
    if not hasattr(self, '_log_counter'):
        self._log_counter = 0
    self._log_counter += 1
    if self._log_counter % 1000 == 0:
        print(f"[Discriminator] amp_reward_coef={self.amp_reward_coef}, task_reward_lerp={self.task_reward_lerp}")
    
    # ... 原有代码
```

---

## 八、总结

### 8.1 配置加载机制

1. ✅ **配置从文件读取**: `task_registry.get_cfgs()`
2. ✅ **转换为字典**: `agent_cfg.to_dict()`
3. ✅ **传递给 Runner**: `AmpOnPolicyRunner(env, train_cfg, ...)`
4. ✅ **传递给 Discriminator**: `Discriminator(..., amp_reward_coef, ..., task_reward_lerp)`
5. ✅ **保存为实例变量**: `self.amp_reward_coef`, `self.task_reward_lerp`
6. ✅ **检查点恢复不影响**: `load_state_dict()` 只加载权重，不改变配置参数

### 8.2 确保新配置生效的检查清单

- [ ] 配置文件已保存（`amp_reward_coef=0.01, lerp=0.96`）
- [ ] 运行验证脚本通过
- [ ] 清除 Python 缓存
- [ ] 重启训练脚本
- [ ] 在训练时添加调试输出验证
- [ ] 从训练日志计算 AMP 占比验证

### 8.3 如果仍然不生效

1. **检查是否有其他配置文件**
2. **检查命令行参数是否覆盖了配置**
3. **添加强制设置代码（方法1）**
4. **检查训练日志，计算实际 AMP 占比**

---

## 九、下一步

1. **运行验证脚本**: `python legged_lab/scripts/verify_amp_config.py`
2. **如果验证通过，重新训练**
3. **监控训练日志，计算 AMP 占比**
4. **如果占比仍然高，进一步调整参数**
