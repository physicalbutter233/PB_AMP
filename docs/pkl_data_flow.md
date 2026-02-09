# PKL 文件在仓库中的处理流程详解

本文档详细说明 pkl 文件进入仓库后经历的完整处理流程，以及每个阶段生成的文件格式和包含的信息。

## 一、完整数据流概览

```
GMR 输出的 pkl 文件
        │
        ▼ 步骤1: gmr_data_conversion.py
        │  文件: legged_lab/scripts/gmr_data_conversion.py
        │  处理: 格式转换、速度计算、四元数转欧拉角
        │
motion_visualization/*.txt (JSON格式)
        │  位置: legged_lab/envs/{robot}/datasets/motion_visualization/
        │  用途: 用于动作可视化播放
        │
        ▼ 步骤2: play_amp_animation.py (带 --save_path)
        │  文件: legged_lab/scripts/play_amp_animation.py
        │  处理: 仿真插帧、计算末端位姿、高度偏移
        │
motion_amp_expert/*.txt (JSON格式)
        │  位置: legged_lab/envs/{robot}/datasets/motion_amp_expert/
        │  用途: AMP 训练时的专家参考数据
        │
        ▼ 步骤3: 训练时加载
        │  文件: rsl_rl/rsl_rl/utils/motion_loader.py
        │  处理: 加载专家数据，用于 AMP 判别器训练
        │
AMP 训练流程 (train.py)
```

---

## 二、阶段1: PKL → motion_visualization

### 2.1 输入文件：PKL 格式

**来源**: GMR (smplx_to_robot) 输出

**文件结构**:
```python
{
    "root_pos": np.array,      # (N, 3) - root 世界坐标 [x, y, z]，单位：米
    "root_rot": np.array,      # (N, 4) - 四元数 xyzw 格式
    "dof_pos": np.array,       # (N, 20 或 23) - 关节角度，单位：弧度
}
```

**处理脚本**: `legged_lab/scripts/gmr_data_conversion.py`

**关键处理步骤**:

| 步骤 | 操作 | 代码位置 | 说明 |
|------|------|----------|------|
| 1 | 加载 pkl | `pickle.load(f)` | 读取 GMR 输出的字典数据 |
| 2 | 四元数转换 | `root_rot[:, [3,0,1,2]]` | xyzw → wxyz（内部计算用） |
| 3 | 移除手腕（可选） | `dof_pos[:, ROBAN_JOINT_INDICES_NO_WRIST]` | Roban S14: 23关节 → 21关节 |
| 4 | 计算线性速度 | `(root_pos[1:] - root_pos[:-1]) / dt` | 相邻帧位置差分 |
| 5 | 计算角速度 | `Rotation.from_quat().as_rotvec() / dt` | 四元数差 → axis-angle |
| 6 | 计算关节速度 | `(dof_pos[1:] - dof_pos[:-1]) / dt` | 相邻帧关节角度差分 |
| 7 | 四元数转欧拉角 | `Rotation.as_euler('XYZ')` | 转换为 XYZ 欧拉角（弧度） |
| 8 | 欧拉角连续性处理 | `np.unwrap()` | 处理跨 ±π 跳变 |
| 9 | 帧数调整 | 丢弃最后一帧 | 速度差分导致少一帧 |

### 2.2 输出文件：motion_visualization/*.txt

**文件格式**: JSON

**文件位置**: 
- `legged_lab/envs/roban/datasets/motion_visualization/*.txt`
- `legged_lab/envs/tienkung/datasets/motion_visualization/*.txt`

**数据结构**:
```json
{
    "LoopMode": "Wrap",
    "FrameDuration": 0.033,  // 1.0 / fps
    "EnableCycleOffsetPosition": true,
    "EnableCycleOffsetRotation": true,
    "MotionWeight": 0.5,
    "Frames": [
        [52维数据],  // 每帧数据
        ...
    ]
}
```

**每帧数据格式（52维）**:

| 列索引 | 维度 | 字段名 | 含义 | 单位/约定 |
|--------|------|--------|------|-----------|
| 0-2 | 3 | root_pos | root 位置 [x, y, z] | 米 |
| 3-5 | 3 | root_rot_euler | 欧拉角 [roll, pitch, yaw] | 弧度，XYZ顺序 |
| 6-25 | 20 | dof_pos | 关节位置 | 弧度，顺序：左腿→右腿→左臂→右臂 |
| 26-28 | 3 | root_lin_vel | root 线性速度 [vx, vy, vz] | m/s |
| 29-31 | 3 | root_ang_vel | root 角速度 [wx, wy, wz] | rad/s |
| 32-51 | 20 | dof_vel | 关节速度 | rad/s |

**配置文件引用**:
- `walk_cfg.py`: `amp_motion_files_display = ["path/to/motion_visualization/*.txt"]`

**使用场景**:
- `play_amp_animation.py` 用于可视化播放动作
- 检查动作质量和正确性

---

## 三、阶段2: motion_visualization → motion_amp_expert

### 3.1 输入文件：motion_visualization/*.txt

**格式**: JSON，每帧 52 维（与阶段1输出一致）

**处理脚本**: `legged_lab/scripts/play_amp_animation.py` (带 `--save_path` 参数)

**关键处理步骤**:

| 步骤 | 操作 | 代码位置 | 说明 |
|------|------|----------|------|
| 1 | 加载 motion_visualization | `AMPLoaderDisplay` | 从 JSON 文件加载数据 |
| 2 | 时间插值 | `get_full_frame_at_time()` | 线性插值生成平滑帧 |
| 3 | 高度偏移 | `root_pos[2] += 0.3` | 补偿坐标系差异，避免穿地 |
| 4 | 仿真步进 | `sim.step()` | 在 Isaac Sim 中执行物理仿真 |
| 5 | 计算末端位姿 | `visualize_motion()` | 计算手部和脚部位置（root系） |
| 6 | 关节顺序重排 | 按训练顺序排列 | right_arm→left_arm→right_leg→left_leg |
| 7 | 保存为 JSON | 写入文件 | 保存为 motion_amp_expert 格式 |

### 3.2 输出文件：motion_amp_expert/*.txt

**文件格式**: JSON

**文件位置**:
- `legged_lab/envs/roban/datasets/motion_amp_expert/*.txt`
- `legged_lab/envs/tienkung/datasets/motion_amp_expert/*.txt`

**数据结构**:
```json
{
    "LoopMode": "Wrap",
    "FrameDuration": 0.033,
    "EnableCycleOffsetPosition": true,
    "EnableCycleOffsetRotation": true,
    "MotionWeight": 0.5,
    "Frames": [
        [52维数据],  // 每帧数据（Roban S14: 21+21+12=54维）
        ...
    ]
}
```

**每帧数据格式（Roban S14: 54维，TienKung: 52维）**:

**Roban S14 (54维)**:

| 列索引 | 维度 | 字段名 | 含义 | 单位 |
|--------|------|--------|------|------|
| 0-20 | 21 | dof_pos | 关节位置 | 弧度，顺序：waist(1) + left_leg(6) + right_leg(6) + left_arm(4) + right_arm(4) |
| 21-41 | 21 | dof_vel | 关节速度 | rad/s，顺序同上 |
| 42-44 | 3 | left_hand_pos | 左手位置（root系） | 米 |
| 45-47 | 3 | right_hand_pos | 右手位置（root系） | 米 |
| 48-50 | 3 | left_foot_pos | 左脚位置（root系） | 米 |
| 51-53 | 3 | right_foot_pos | 右脚位置（root系） | 米 |

**TienKung (52维)**:

| 列索引 | 维度 | 字段名 | 含义 | 单位 |
|--------|------|--------|------|------|
| 0-19 | 20 | dof_pos | 关节位置 | 弧度，顺序：right_arm(4) + left_arm(4) + right_leg(6) + left_leg(6) |
| 20-39 | 20 | dof_vel | 关节速度 | rad/s，顺序同上 |
| 40-42 | 3 | left_hand_pos | 左手位置（root系） | 米 |
| 43-45 | 3 | right_hand_pos | 右手位置（root系） | 米 |
| 46-48 | 3 | left_foot_pos | 左脚位置（root系） | 米 |
| 49-51 | 3 | right_foot_pos | 右脚位置（root系） | 米 |

**关键差异**:
- ❌ **移除**: root_pos, root_rot_euler, root_lin_vel, root_ang_vel（不再需要）
- ✅ **新增**: 末端执行器位置（left_hand, right_hand, left_foot, right_foot）
- 🔄 **重排**: 关节顺序改为训练时的观测顺序

**配置文件引用**:
- `walk_cfg.py`: `amp_motion_files = ["path/to/motion_amp_expert/*.txt"]`

**使用场景**:
- AMP 训练时的专家参考数据
- 用于训练 AMP 判别器（Discriminator）

---

## 四、阶段3: 训练时加载和使用

### 4.1 数据加载

**加载器**: `rsl_rl/rsl_rl/utils/motion_loader.py` → `AMPLoader`

**加载过程**:

```python
# 1. 初始化 AMPLoader
loader = AMPLoader(
    device=device,
    time_between_frames=0.02,  # 采样间隔
    motion_files=glob.glob("datasets/motion_amp_expert/*")
)

# 2. 加载 JSON 文件
with open(motion_file) as f:
    motion_json = json.load(f)
    motion_data = np.array(motion_json["Frames"])  # (N, 54)
    
# 3. 转换为 PyTorch Tensor
trajectories.append(
    torch.tensor(motion_data[:, :54], dtype=torch.float32, device=device)
)
```

**存储的数据结构**:

| 属性 | 类型 | 说明 |
|------|------|------|
| `trajectories` | `List[torch.Tensor]` | 每个轨迹的帧数据 (N, 54) |
| `trajectory_names` | `List[str]` | 轨迹文件名列表 |
| `trajectory_lens` | `List[float]` | 每个轨迹的长度（秒） |
| `trajectory_weights` | `np.array` | 轨迹采样权重（归一化） |
| `trajectory_frame_durations` | `np.array` | 每帧持续时间 |

### 4.2 训练时的使用

**使用位置**: `rsl_rl/rsl_rl/algorithms/amp_ppo.py`

**AMP 训练流程**:

```
1. 采样专家数据
   ├─ loader.get_full_frame_batch(num_frames)
   └─ 随机采样轨迹和时间点，返回 (batch_size, 54) 的专家状态

2. 采样策略数据
   ├─ env.get_amp_obs_for_expert_trans()
   └─ 从当前策略执行中获取 AMP 观察值 (batch_size, 54)

3. 训练判别器
   ├─ discriminator(policy_state, policy_next_state) → policy_d
   ├─ discriminator(expert_state, expert_next_state) → expert_d
   └─ loss = MSE(expert_d, 1) + MSE(policy_d, -1) + grad_penalty

4. 计算 AMP 奖励
   ├─ rewards = discriminator.predict_amp_reward(amp_obs, next_amp_obs)
   └─ 用于 PPO 策略更新
```

**关键代码位置**:

| 功能 | 文件 | 行号/函数 |
|------|------|-----------|
| 加载专家数据 | `motion_loader.py` | `AMPLoader.__init__()` |
| 采样专家帧 | `motion_loader.py` | `get_full_frame_batch()` |
| 训练判别器 | `amp_ppo.py` | `update()` → 判别器损失 |
| 计算 AMP 奖励 | `amp_ppo.py` | `discriminator.predict_amp_reward()` |

---

## 五、文件位置总结

### 5.1 输入文件（PKL）

**来源**: GMR 输出
- 格式: Python pickle 文件
- 内容: `{root_pos, root_rot, dof_pos}`

### 5.2 中间文件（motion_visualization）

**位置**:
- `legged_lab/envs/roban/datasets/motion_visualization/*.txt`
- `legged_lab/envs/tienkung/datasets/motion_visualization/*.txt`

**格式**: JSON
**内容**: 52维/帧 `[root_pos(3), euler(3), dof_pos(20), root_lin_vel(3), root_ang_vel(3), dof_vel(20)]`

**配置引用**: `walk_cfg.py` → `amp_motion_files_display`

### 5.3 最终文件（motion_amp_expert）

**位置**:
- `legged_lab/envs/roban/datasets/motion_amp_expert/*.txt`
- `legged_lab/envs/tienkung/datasets/motion_amp_expert/*.txt`

**格式**: JSON
**内容**: 54维/帧（Roban）或 52维/帧（TienKung）`[dof_pos, dof_vel, end_effector_pos]`

**配置引用**: `walk_cfg.py` → `amp_motion_files`

---

## 六、关键处理细节

### 6.1 关节数量变化

**Roban S14**:
- GMR 输出: 23 关节（含手腕）
- 移除手腕后: 21 关节（waist(1) + legs(12) + arms(8)）
- 使用 `--remove_roban_wrist` 参数

**TienKung**:
- GMR 输出: 20 关节（无手腕）
- 保持不变: 20 关节

### 6.2 坐标系转换

**四元数格式**:
- PKL: `xyzw`
- 内部计算: `wxyz` (Isaac Sim 格式)
- 输出: `XYZ` 欧拉角（弧度）

**高度偏移**:
- `root_pos[2] += 0.3` (米)
- 原因: 补偿动捕坐标系与仿真坐标系的差异

### 6.3 数据维度变化

| 阶段 | 维度 | 说明 |
|------|------|------|
| PKL | 3+4+20/23 | root_pos(3) + root_rot(4) + dof_pos(20/23) |
| motion_visualization | 52 | root_pos(3) + euler(3) + dof_pos(20) + velocities(26) |
| motion_amp_expert | 54/52 | dof_pos(21/20) + dof_vel(21/20) + end_effector(12) |

---

## 七、使用命令示例

### 7.1 步骤1: PKL → motion_visualization

```bash
python legged_lab/scripts/gmr_data_conversion.py \
    --input_pkl <path_to_gmr_output.pkl> \
    --output_txt legged_lab/envs/roban/datasets/motion_visualization/walk.txt \
    --fps 30.0 \
    --remove_roban_wrist  # 仅 Roban S14 需要
```

### 7.2 步骤2: motion_visualization → motion_amp_expert

```bash
python legged_lab/scripts/play_amp_animation.py \
    --task=walk \
    --num_envs=1 \
    --save_path legged_lab/envs/roban/datasets/motion_amp_expert/walk.txt \
    --fps 30.0
```

### 7.3 可视化检查

```bash
# 可视化 motion_visualization 数据
python legged_lab/scripts/play_amp_animation.py --task=walk --num_envs=1
```

---

## 八、相关文件清单

### 8.1 处理脚本

| 文件 | 功能 |
|------|------|
| `legged_lab/scripts/gmr_data_conversion.py` | PKL → motion_visualization |
| `legged_lab/scripts/play_amp_animation.py` | motion_visualization → motion_amp_expert |

### 8.2 数据加载器

| 文件 | 功能 |
|------|------|
| `rsl_rl/rsl_rl/utils/motion_loader.py` | 训练时加载 motion_amp_expert |
| `rsl_rl/rsl_rl/utils/motion_loader_for_display.py` | 可视化时加载 motion_visualization |

### 8.3 配置文件

| 文件 | 配置项 |
|------|--------|
| `legged_lab/envs/roban/walk_cfg.py` | `amp_motion_files_display`, `amp_motion_files` |
| `legged_lab/envs/tienkung/walk_cfg.py` | `amp_motion_files_display`, `amp_motion_files` |

### 8.4 环境实现

| 文件 | 功能 |
|------|------|
| `legged_lab/envs/roban/roban_envs.py` | `visualize_motion()`, `get_amp_obs_for_expert_trans()` |
| `legged_lab/envs/tienkung/tienkung_env.py` | `visualize_motion()`, `get_amp_obs_for_expert_trans()` |

---

## 九、总结

PKL 文件进入仓库后经历三个阶段：

1. **格式转换** (gmr_data_conversion.py)
   - PKL → motion_visualization/*.txt
   - 添加速度信息，转换坐标系

2. **仿真处理** (play_amp_animation.py)
   - motion_visualization → motion_amp_expert/*.txt
   - 通过仿真计算末端位姿，移除 root 信息

3. **训练使用** (motion_loader.py)
   - 加载 motion_amp_expert 数据
   - 用于 AMP 判别器训练

每个阶段生成的文件都有特定的格式和用途，确保数据在 AMP 训练流程中正确使用。
