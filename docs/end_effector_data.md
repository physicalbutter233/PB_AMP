# End Effector（末端执行器）数据详解

本文档详细说明 AMP 训练中 end_effector 数据的维度、内容和计算方法。

## 一、数据维度总览

**End Effector 总维度**: **12 维**

| 末端执行器 | 维度 | 说明 |
|-----------|------|------|
| left_hand_pos | 3 | 左手位置 [x, y, z] |
| right_hand_pos | 3 | 右手位置 [x, y, z] |
| left_foot_pos | 3 | 左脚位置 [x, y, z] |
| right_foot_pos | 3 | 右脚位置 [x, y, z] |
| **总计** | **12** | 所有末端执行器位置 |

---

## 二、数据内容详解

### 2.1 手部位置 (Hand Position)

**维度**: 每个手 3 维，共 6 维

**计算方式**:
```python
# 左手位置计算
left_hand_pos = (
    elbow_body_pos_world                    # 肘部在世界坐标系的位置
    - root_pos_world                         # 减去 root 位置
    + quat_rotate(elbow_quat, [0, 0, -0.3]) # 加上肘部坐标系下的局部向量
)
left_hand_pos = quat_apply(root_quat_conj, left_hand_pos)  # 转换到 root 坐标系

# 右手位置计算（类似）
right_hand_pos = (
    elbow_body_pos_world
    - root_pos_world
    + quat_rotate(elbow_quat, [0, 0, -0.3])
)
right_hand_pos = quat_apply(root_quat_conj, right_hand_pos)
```

**关键参数**:
- **肘部 body**: `elbow_body_ids[0]` (左手), `elbow_body_ids[1]` (右手)
- **局部向量**: `[0.0, 0.0, -0.3]` 米（从肘部指向手部的向量）
- **坐标系**: 最终转换到 **root 坐标系**（机器人基座坐标系）

**物理含义**:
- 手部位置 = 肘部位置 + 从肘部到手的偏移向量
- 偏移向量 `[0, 0, -0.3]` 表示在肘部坐标系下，沿 z 轴负方向 0.3 米

### 2.2 脚部位置 (Foot Position)

**维度**: 每个脚 3 维，共 6 维

**计算方式**:
```python
# 左脚位置计算
left_foot_pos = (
    foot_body_pos_world  # 脚部 body 在世界坐标系的位置
    - root_pos_world     # 减去 root 位置
)
left_foot_pos = quat_apply(root_quat_conj, left_foot_pos)  # 转换到 root 坐标系

# 右脚位置计算（类似）
right_foot_pos = (
    foot_body_pos_world
    - root_pos_world
)
right_foot_pos = quat_apply(root_quat_conj, right_foot_pos)
```

**关键参数**:
- **脚部 body**: `feet_body_ids[0]` (左脚), `feet_body_ids[1]` (右脚)
- **坐标系**: 最终转换到 **root 坐标系**

**物理含义**:
- 脚部位置 = 脚部 body 相对于 root 的位置
- 直接使用脚部 body 的位置，无需额外偏移

---

## 三、在 AMP 数据中的位置

### 3.1 motion_amp_expert 格式

**完整数据维度**:
- **Roban S14**: 54 维 = 21(dof_pos) + 21(dof_vel) + 12(end_effector)
- **TienKung**: 52 维 = 20(dof_pos) + 20(dof_vel) + 12(end_effector)

**数据排列顺序**:
```
[dof_pos, dof_vel, left_hand_pos, right_hand_pos, left_foot_pos, right_foot_pos]
```

**索引位置** (Roban S14):
- `[0:21]`: dof_pos (21 维)
- `[21:42]`: dof_vel (21 维)
- `[42:45]`: left_hand_pos (3 维)
- `[45:48]`: right_hand_pos (3 维)
- `[48:51]`: left_foot_pos (3 维)
- `[51:54]`: right_foot_pos (3 维)

### 3.2 代码中的定义

**文件**: `rsl_rl/rsl_rl/utils/motion_loader.py`

```python
class AMPLoader:
    JOINT_POS_SIZE = 21
    JOINT_VEL_SIZE = 21
    END_EFFECTOR_POS_SIZE = 12  # ⭐ 12 维
    
    END_POS_START_IDX = JOINT_VEL_END_IDX  # 42
    END_POS_END_IDX = END_POS_START_IDX + END_EFFECTOR_POS_SIZE  # 54
```

---

## 四、计算代码位置

### 4.1 训练时计算

**文件**: `legged_lab/envs/roban/roban_envs.py`

**函数**: `get_amp_obs_for_expert_trans()` (第 578-631 行)

```python
def get_amp_obs_for_expert_trans(self):
    """AMP obs: joint_pos(21) + joint_vel(21) + end_effector_pos(12) = 54."""
    
    # 计算手部位置
    left_hand_pos = (
        self.robot.data.body_state_w[:, self.elbow_body_ids[0], :3]
        - self.robot.data.root_state_w[:, 0:3]
        + quat_rotate(self.robot.data.body_state_w[:, self.elbow_body_ids[0], 3:7], 
                     self.left_arm_local_vec)
    )
    left_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), 
                               left_hand_pos)
    
    # 计算脚部位置
    left_foot_pos = (
        self.robot.data.body_state_w[:, self.feet_body_ids[0], :3] 
        - self.robot.data.root_state_w[:, 0:3]
    )
    left_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), 
                               left_foot_pos)
    
    # 返回完整 AMP 观察值
    return torch.cat([
        joint_pos_21,      # 21 维
        joint_vel_21,      # 21 维
        left_hand_pos,     # 3 维
        right_hand_pos,    # 3 维
        left_foot_pos,     # 3 维
        right_foot_pos,    # 3 维
    ], dim=-1)  # 总计 54 维
```

### 4.2 可视化时计算

**文件**: `legged_lab/envs/roban/roban_envs.py`

**函数**: `visualize_motion()` (第 240-382 行)

计算方式与训练时相同，用于生成 `motion_amp_expert` 数据。

---

## 五、关键参数说明

### 5.1 手部局部向量

**定义位置**: `roban_envs.py` 第 213-214 行

```python
self.left_arm_local_vec = torch.tensor([0.0, 0.0, -0.3], device=self.device)
self.right_arm_local_vec = torch.tensor([0.0, 0.0, -0.3], device=self.device)
```

**含义**:
- `[0.0, 0.0, -0.3]`: 在肘部坐标系下，从肘部指向手部的向量
- 单位: 米
- 方向: z 轴负方向 0.3 米

**为什么需要这个向量**:
- Roban S14 的手部不是独立的 body，而是手臂末端
- 需要通过肘部位置 + 局部偏移来计算手部位置
- 0.3 米是手臂从肘部到手的长度

### 5.2 Body ID 定义

**肘部 body**:
```python
self.elbow_body_ids, _ = self.robot.find_bodies(
    name_keys=["zarm_l4_link", "zarm_r4_link"],  # 左/右臂第4个link（肘部）
    preserve_order=True,
)
```

**脚部 body**:
```python
self.feet_body_ids, _ = self.robot.find_bodies(
    name_keys=["leg_l6_link", "leg_r6_link"],  # 左/右脚
    preserve_order=True,
)
```

---

## 六、坐标系说明

### 6.1 Root 坐标系

**定义**: 机器人基座（base_link）的坐标系

**特点**:
- 原点: root 位置
- 方向: 与 root 姿态一致
- 用途: 所有 end_effector 位置最终都转换到此坐标系

### 6.2 世界坐标系 → Root 坐标系转换

**转换步骤**:
1. 计算相对位置: `world_pos - root_pos`
2. 应用 root 四元数的共轭: `quat_apply(root_quat_conj, relative_pos)`

**代码**:
```python
# 获取 root 四元数的共轭
root_quat_conj = quat_conjugate(self.robot.data.root_state_w[:, 3:7])

# 转换到 root 坐标系
pos_in_root_frame = quat_apply(root_quat_conj, pos_relative_to_root)
```

**为什么转换到 root 坐标系**:
- AMP 训练需要与机器人姿态无关的特征
- Root 坐标系下的位置更稳定，不受机器人整体旋转影响
- 便于判别器学习运动模式

---

## 七、数据用途

### 7.1 AMP 训练

**作用**: 作为专家参考数据的一部分，用于训练 AMP 判别器

**训练流程**:
1. **专家数据**: 从 `motion_amp_expert/*.txt` 加载，包含 end_effector 位置
2. **策略数据**: 从当前策略执行中计算，使用相同的计算方法
3. **判别器训练**: 对比专家和策略的 end_effector 位置，学习运动模式

### 7.2 运动质量评估

**用途**: 
- 评估手部和脚部的运动轨迹
- 确保策略生成的动作与专家动作相似
- 帮助学习自然的步态和手臂摆动

---

## 八、总结

| 项目 | 值 |
|------|-----|
| **总维度** | 12 维 |
| **左手位置** | 3 维 (x, y, z) |
| **右手位置** | 3 维 (x, y, z) |
| **左脚位置** | 3 维 (x, y, z) |
| **右脚位置** | 3 维 (x, y, z) |
| **坐标系** | Root 坐标系（机器人基座） |
| **单位** | 米 (m) |
| **计算方式** | 基于 body 位置 + 局部偏移（手部）或直接使用（脚部） |

**关键点**:
- ✅ 所有位置都在 root 坐标系下
- ✅ 手部位置通过肘部位置 + 局部向量计算
- ✅ 脚部位置直接使用脚部 body 位置
- ✅ 用于 AMP 训练，帮助学习自然运动模式
