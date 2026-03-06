# 动作数据处理流程说明

本文档说明 TienKung-Lab 中从 GMR 重定向结果到 AMP 训练数据的完整处理链路，以及各步骤的格式约定。

---

## 一、gmr_data_conversion.py 对 pkl 的处理

### 1. 输入 pkl 格式（GMR smplx_to_robot 输出）

| 字段 | 维度 | 含义 |
|------|------|------|
| `root_pos` | (N, 3) | root 世界坐标 [x, y, z]，单位：米 |
| `root_rot` | (N, 4) | 四元数 **xyzw** |
| `dof_pos` | (N, 20) | 20 个关节角度，单位：弧度 |

### 2. 处理步骤

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 四元数转换 | `root_rot[:, [3,0,1,2]]`，xyzw → wxyz（内部计算用） |
| 2 | 线性速度 | `(root_pos[1:] - root_pos[:-1]) / dt` |
| 3 | 角速度 | 相邻帧四元数差 → axis-angle → 除以 dt |
| 4 | 关节速度 | `(dof_pos[1:] - dof_pos[:-1]) / dt` |
| 5 | 欧拉角 | 四元数 → `scipy.Rotation.as_euler('XYZ', degrees=False)` |
| 6 | 欧拉角连续性 | `np.unwrap()` 处理跨 ±π 跳变 |
| 7 | 帧数 | 丢弃最后一帧（速度差分导致少一帧） |

### 3. 输出格式（motion_visualization）

**总列数：52 列**

| 列索引 | 维度 | 含义 | 单位/约定 |
|--------|------|------|-----------|
| 0–2 | 3 | root_pos [x, y, z] | 米 |
| 3–5 | 3 | root_rot 欧拉角 [roll, pitch, yaw] | 弧度，XYZ 顺序 |
| 6–25 | 20 | dof_pos | 弧度，顺序：左腿→右腿→左臂→右臂 |
| 26–28 | 3 | root_lin_vel [vx, vy, vz] | m/s |
| 29–31 | 3 | root_ang_vel [wx, wy, wz] | rad/s |
| 32–51 | 20 | dof_vel | rad/s |

### 4. 四元数与帧率

- **四元数**：pkl 使用 xyzw；输出用欧拉 XYZ（弧度），不再使用四元数。
- **帧率**：`--fps` 默认 30，用于差分和 `FrameDuration`，假定 pkl 帧率与 fps 一致。

---

## 二、play_amp_animation 对输入的处理与输出

### 1. 输入（motion_visualization）

- 格式：JSON，含 `Frames`、`FrameDuration` 等
- 每帧 52 维，与 gmr_data_conversion 输出一致

### 2. 输出（motion_amp_expert，带 `--save_path` 时）

**总列数：52 列**

| 列索引 | 维度 | 含义 |
|--------|------|------|
| 0–19 | 20 | dof_pos（right_arm, left_arm, right_leg, left_leg） |
| 20–39 | 20 | dof_vel |
| 40–42 | 3 | left_hand_pos（root 系） |
| 43–45 | 3 | right_hand_pos |
| 46–48 | 3 | left_foot_pos |
| 49–51 | 3 | right_foot_pos |

---

## 三、play_amp_animation 关键操作说明

### 3.1 插帧（帧间插值）

**位置**：`AMPLoaderDisplay.get_full_frame_at_time()` → `blend_frame_pose()`

**实现**：

```python
# 将连续时间映射到 [0, 1] 的归一化位置
p = float(time) / self.trajectory_lens[traj_idx]
n = self.trajectories_full[traj_idx].shape[0]
idx_low = int(np.floor(p * n))
idx_high = int(np.ceil(p * n))
blend = p * n - idx_low  # 插值权重 [0, 1)

# 线性插值
return (1.0 - blend) * frame_start + blend * frame_end
```

**原因**：

1. **时间离散 vs 连续**：数据是离散帧序列，而采样时间 `time = frame_cnt * (1.0/fps)` 是连续的。例如 `time=0.05s` 时，真实帧可能在 0.033s 和 0.066s 之间，需要插值。
2. **平滑播放**：若直接取最近帧，会出现明显卡顿；插值后动画更平滑。
3. **支持任意 fps**：`--fps` 可与数据 `FrameDuration` 不同，插值可处理任意采样率。
4. **插值方式**：对关节位置、速度、欧拉角等使用**线性插值**。注意：对欧拉角做线性插值在极端姿态下可能不够理想，但对一般步态通常可接受。

---

### 3.2 root_pos[2] += 0.3（根高度偏移）

**位置**：`tienkung_env.py` → `visualize_motion()` 中：

```python
root_pos = visual_motion_frame[:3].clone()
root_pos[2] += 0.3
```

**原因**：

1. **坐标系与地面定义不同**  
   - 动捕 / GMR 的 root 高度通常基于动捕地面或 SMPL 坐标系。  
   - Isaac Sim 中地面在 `z=0`。  
   两者对“地面”和“根高度”的定义可能不一致，需要高度偏移。

2. **避免穿地**  
   若直接用原始 root 高度，机器人可能部分陷入地面，导致穿模和物理异常。加 0.3m 相当于整体上移，确保脚与地面接触关系合理。

3. **与 TienKung 尺寸匹配**  
   TienKung 站立时 pelvis 高度约 0.9–1.0 m。动捕数据中 root z 常在 ~0.83 m 左右。0.3 m 的偏移是为了将这类数据对齐到仿真中合理的站立高度。

4. **经验值**  
   0.3 是经验常数，用于：
   - 补偿动捕坐标系差异
   - 保证在 `terrain_type="plane"` 时机器人正确站立  
   若更换动捕来源或机器人，可能需要重新标定该偏移。

---

## 四、数据流总览

```
GMR pkl (root_pos, root_rot_xyzw, dof_pos)
        │
        ▼  gmr_data_conversion.py
        │  - 四元数 xyzw → 欧拉 XYZ
        │  - 差分计算 root_lin_vel, root_ang_vel, dof_vel
        │  - 帧数 N → N-1
        │
motion_visualization/*.txt (JSON)
        │  [root_pos, euler_XYZ, dof_pos, root_lin_vel, root_ang_vel, dof_vel]
        │
        ▼  play_amp_animation.py（带 --save_path）
        │  - AMPLoaderDisplay 插帧
        │  - root_pos[2] += 0.3
        │  - 仿真步进，计算末端位姿
        │
motion_amp_expert/*.txt (JSON)
        │  [dof_pos, dof_vel, left_hand, right_hand, left_foot, right_foot]
        │
        ▼  train.py / AMP 判别器
```

---

## 五、补充说明

- **关节顺序**：motion_visualization 为 left_leg→right_leg→left_arm→right_arm；motion_amp_expert 为 right_arm→left_arm→right_leg→left_leg，与训练观测顺序一致。
- **末端位姿**：通过 Isaac Sim 的刚体位姿和手臂局部向量计算，再转换到 root 系。
- **欧拉约定**：XYZ 弧度；scipy 用 xyzw；Isaac 用 wxyz。
