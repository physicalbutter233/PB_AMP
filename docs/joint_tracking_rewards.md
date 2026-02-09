# 关节位置和角度跟踪奖励函数使用指南

## 概述

本文档介绍如何使用关节位置和角度跟踪奖励函数来控制机器人的全身关节运动。这些奖励函数可以帮助机器人：
- 保持关节在合理的范围内
- 跟踪参考动作（如默认姿态）
- 实现平滑的关节运动

## 可用的奖励函数

### 1. `joint_pos_tracking_l2`
**功能**: 使用 L2 范数跟踪关节位置相对于默认位置的偏差

**特点**:
- 惩罚关节偏离默认位置
- 使用负权重（值越小越好）
- 适用于需要严格跟踪的场景

**参数**:
- `asset_cfg`: 指定要跟踪的关节（可以是所有关节或特定关节组）

**使用示例**:
```python
# 跟踪所有关节
joint_pos_tracking_all = RewTerm(
    func=mdp.joint_pos_tracking_l2,
    weight=-0.5,  # 负权重，值越小越好
    params={
        "asset_cfg": SceneEntityCfg("robot"),
    },
)

# 跟踪特定关节组（例如腿部）
joint_pos_tracking_legs = RewTerm(
    func=mdp.joint_pos_tracking_l2,
    weight=-0.2,
    params={
        "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=["leg_l1_joint", "leg_l2_joint", ...],
        ),
    },
)
```

### 2. `joint_pos_tracking_exp`
**功能**: 使用指数奖励跟踪关节位置

**特点**:
- 当关节接近默认位置时给出高奖励
- 使用正权重（值越大越好）
- 通过 `std` 参数控制敏感度

**参数**:
- `std`: 标准差，控制奖励衰减速度（越小越严格）
- `asset_cfg`: 指定要跟踪的关节

**使用示例**:
```python
joint_pos_tracking_exp_all = RewTerm(
    func=mdp.joint_pos_tracking_exp,
    weight=0.3,  # 正权重，值越大越好
    params={
        "std": 0.5,  # 标准差，控制敏感度
        "asset_cfg": SceneEntityCfg("robot"),
    },
)
```

**权重和 std 的选择**:
- `std` 较小（0.1-0.3）: 更严格的跟踪，只奖励非常接近默认位置的关节
- `std` 较大（0.5-1.0）: 更宽松的跟踪，允许更大的偏差
- `weight`: 通常设置为 0.1-1.0，根据需要的跟踪强度调整

### 3. `joint_vel_tracking_l2`
**功能**: 跟踪关节速度，鼓励平滑运动

**特点**:
- 惩罚关节速度偏离默认速度（通常为0）
- 使用负权重
- 有助于减少抖动和不必要的运动

**使用示例**:
```python
joint_vel_tracking_all = RewTerm(
    func=mdp.joint_vel_tracking_l2,
    weight=-0.05,  # 负权重，值越小越好
    params={
        "asset_cfg": SceneEntityCfg("robot"),
    },
)
```

## 在 Roban 配置中的使用

在 `RobanLiteRewardCfg` 中，已经添加了以下关节跟踪奖励：

### 1. 全身关节跟踪
```python
# L2 形式的跟踪（惩罚偏差）
joint_pos_tracking_all = RewTerm(
    func=mdp.joint_pos_tracking_l2,
    weight=-0.5,
    params={"asset_cfg": SceneEntityCfg("robot")},
)

# 指数形式的跟踪（奖励接近）
joint_pos_tracking_exp_all = RewTerm(
    func=mdp.joint_pos_tracking_exp,
    weight=0.3,
    params={"std": 0.5, "asset_cfg": SceneEntityCfg("robot")},
)

# 速度跟踪（平滑性）
joint_vel_tracking_all = RewTerm(
    func=mdp.joint_vel_tracking_l2,
    weight=-0.05,
    params={"asset_cfg": SceneEntityCfg("robot")},
)
```

### 2. 分组跟踪（可选）
```python
# 只跟踪腿部关节
joint_pos_tracking_legs = RewTerm(...)

# 只跟踪手臂关节
joint_pos_tracking_arms = RewTerm(...)
```

## 权重调整建议

### 初始权重设置
1. **关节位置跟踪（L2）**: `-0.1` 到 `-1.0`
   - 如果关节偏离太大，增加权重绝对值（如 `-0.5` → `-1.0`）
   - 如果过于严格导致动作僵硬，减小权重绝对值（如 `-0.5` → `-0.2`）

2. **关节位置跟踪（指数）**: `0.1` 到 `1.0`
   - 与 L2 形式配合使用，提供正向激励
   - 通常设置为 L2 权重的 0.5-1.0 倍

3. **关节速度跟踪**: `-0.01` 到 `-0.1`
   - 主要用于平滑性，权重通常较小
   - 如果动作抖动严重，可以增加到 `-0.1`

### 与其他奖励的平衡
- **速度跟踪奖励**: 通常权重最高（10.0-8.0），关节跟踪不应与之冲突
- **能量消耗**: 关节跟踪可能增加能量消耗，需要平衡
- **动作平滑度**: `action_rate_l2` 与 `joint_vel_tracking_l2` 功能类似，可以只使用一个

## 使用场景

### 场景 1: 保持默认姿态
**目标**: 让机器人在静止时保持默认姿态

**推荐配置**:
```python
joint_pos_tracking_exp_all = RewTerm(
    func=mdp.joint_pos_tracking_exp,
    weight=0.5,
    params={"std": 0.3, "asset_cfg": SceneEntityCfg("robot")},
)
```

### 场景 2: 跟踪参考动作
**目标**: 让机器人跟踪 AMP expert motion 的关节角度

**注意**: 这需要修改奖励函数，使其跟踪参考动作而非默认位置。可以考虑：
- 在环境中存储参考关节位置
- 修改奖励函数以使用参考位置而非默认位置

### 场景 3: 减少关节抖动
**目标**: 减少不必要的关节运动

**推荐配置**:
```python
joint_vel_tracking_all = RewTerm(
    func=mdp.joint_vel_tracking_l2,
    weight=-0.1,  # 较大的权重以减少抖动
    params={"asset_cfg": SceneEntityCfg("robot")},
)
```

### 场景 4: 特定关节组跟踪
**目标**: 只跟踪某些关键关节（如腿部或手臂）

**推荐配置**:
```python
# 只跟踪腿部，权重较高
joint_pos_tracking_legs = RewTerm(
    func=mdp.joint_pos_tracking_l2,
    weight=-0.3,
    params={
        "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=["leg_l1_joint", "leg_l2_joint", ...],
        ),
    },
)
```

## 调试建议

1. **监控奖励值**: 在 TensorBoard 中查看关节跟踪奖励的值
   - L2 形式：值应该接近 0（越小越好）
   - 指数形式：值应该在 0-1 之间（越大越好）

2. **观察关节角度**: 可视化关节角度随时间的变化
   - 检查是否在合理范围内
   - 检查是否有异常抖动

3. **调整权重**: 根据训练效果逐步调整
   - 如果关节偏离太大：增加 L2 权重绝对值或减小指数 std
   - 如果动作过于僵硬：减小权重或增加 std

4. **与其他奖励平衡**: 确保关节跟踪不会过度抑制其他重要行为
   - 速度跟踪应该优先
   - 关节跟踪应该辅助而非主导

## 注意事项

1. **默认位置**: 当前实现跟踪的是 `default_joint_pos`，这是机器人的默认/静止姿态
2. **参考动作**: 如果要跟踪 AMP expert motion，需要修改奖励函数以使用参考关节位置
3. **权重选择**: 不同任务可能需要不同的权重，需要根据实际情况调整
4. **计算开销**: 跟踪所有关节会增加计算开销，如果性能有问题，可以考虑只跟踪关键关节

## 示例：完整配置

```python
@configclass
class RobanLiteRewardCfg(LiteRewardCfg):
    # ... 其他奖励 ...
    
    # 关节跟踪奖励
    # 选项1: 使用 L2 形式（推荐用于严格跟踪）
    joint_pos_tracking_all = RewTerm(
        func=mdp.joint_pos_tracking_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # 选项2: 使用指数形式（推荐用于宽松跟踪）
    joint_pos_tracking_exp_all = RewTerm(
        func=mdp.joint_pos_tracking_exp,
        weight=0.3,
        params={"std": 0.5, "asset_cfg": SceneEntityCfg("robot")},
    )
    
    # 选项3: 速度跟踪（平滑性）
    joint_vel_tracking_all = RewTerm(
        func=mdp.joint_vel_tracking_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
```

## 总结

关节跟踪奖励函数提供了灵活的方式来控制机器人的关节运动。根据具体需求选择合适的函数和权重：

- **严格跟踪**: 使用 `joint_pos_tracking_l2`，较大的负权重
- **宽松跟踪**: 使用 `joint_pos_tracking_exp`，较大的 std 值
- **平滑运动**: 使用 `joint_vel_tracking_l2`，适中的负权重
- **分组跟踪**: 针对特定关节组使用不同的权重

记住：奖励函数的设计需要平衡多个目标，关节跟踪应该与其他奖励（如速度跟踪、能量消耗等）协调工作。
