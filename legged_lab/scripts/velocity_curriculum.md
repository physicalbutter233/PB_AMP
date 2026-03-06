# 速度课程学习（Velocity Curriculum）说明

本文档说明：**为什么**要开启速度课程学习、**如何**实现，以及相关代码与文件位置，便于复现与消融实验。

---

## 1. 为什么开启课程学习

### 1.1 动机

- **任务**：机器人需要跟踪随机采样的线速度/角速度指令（`lin_vel_x/y`, `ang_vel_z`），即“按指令走路+转弯”。
- **问题**：若从训练一开始就使用**完整**速度范围（例如 `lin_vel_x=(-0.6, 1.0)`, `ang_vel_z=(-1.57, 1.57)`），指令难度跨度大，策略容易：
  - 在“大速度/大角速度”指令下频繁摔倒，导致 early termination；
  - 学到“站桩”或保守策略以延长 episode，而不是真正跟踪指令。
- **思路**：采用**课程学习**——训练初期只给**小范围、易跟踪**的速度指令（先学慢走、小转弯），等策略在该难度上稳定后，再逐步扩大指令范围，最终覆盖完整速度空间。

### 1.2 设计要点

1. **升级条件（防“站桩升级”）**  
   升级不能只看“episode 是否活得久”，否则站着不动也能升级。因此升级需**同时**满足：
   - 平均 episode 长度 > 阈值（例如 75% 最大时长）；
   - 平均**线速度跟踪误差** < 上限；
   - 平均**角速度跟踪误差** < 上限。  
   即：既要存活够久，又要跟得上指令。

2. **降级条件**  
   若近期平均 episode 长度过短（例如 < 35% 最大时长），说明当前速度范围过难，则**降级**，缩小指令范围。

3. **冷却机制**  
   级别变更后，缓冲区里仍有一批“旧级别”下开始的 episode。若立刻用这些数据做下一次升/降级判断，会导致“火箭跳级”或抖动。因此引入 **cooldown**：在级别变更后的一段时间内，**忽略**在旧级别下开始的 episode，只统计在新级别下完整跑完的 episode。

---

## 2. 实现概览与文件位置

| 内容           | 文件路径 |
|----------------|----------|
| 课程配置类     | `legged_lab/envs/roban/walk_cfg.py` |
| 环境内初始化与更新 | `legged_lab/envs/roban/roban_envs.py` |
| 一键关闭（CLI） | `legged_lab/scripts/train.py` |

下面按**配置 → 初始化 → 每步累积 → reset 时更新**的顺序说明，并标注代码位置。

---

## 3. 配置：`VelocityCurriculumCfg`（walk_cfg.py）

**文件**：`legged_lab/envs/roban/walk_cfg.py`

- **类定义**：约第 **58–108** 行（`VelocityCurriculumCfg`）。
- **在场景配置中的使用**：约第 **429** 行，`velocity_curriculum: VelocityCurriculumCfg = VelocityCurriculumCfg()`。

### 3.1 关键字段

```python
# 约 58–108 行
@configclass
class VelocityCurriculumCfg:
    enable: bool = True   # 总开关，False 则不用课程，始终用 commands.ranges 的完整范围

    # 升/降级阈值
    promote_threshold: float = 0.75   # 平均 episode 长度 > 75% max_episode_length → 可升级
    demote_threshold: float = 0.35    # 平均 episode 长度 < 35% → 降级

    # 跟踪误差上限（防站桩升级）
    promote_lin_vel_err_limit: float = 0.2   # m/s
    promote_ang_vel_err_limit: float = 0.2   # rad/s

    # 统计缓冲区大小（最近 N 个 episode）
    buffer_size: int = 2000

    # 级别变更后的冷却步数（env steps）
    cooldown_steps: int = 500

    # 各级别速度范围，levels[0] 最简单，levels[-1] 与 commands.ranges 一致
    levels: list = None   # __post_init__ 中默认 4 级，见 97–108 行
```

默认 4 级（在 `__post_init__` 中，约 97–108 行）：

- **Level 0**：`lin_vel_x=(0, 0.3)`, `lin_vel_y=(-0.1, 0.1)`, `ang_vel_z=(-0.3, 0.3)`（慢走、小转弯）
- **Level 1–2**：逐步扩大
- **Level 3**：`lin_vel_x=(-0.6, 1.0)`, `lin_vel_y=(-0.5, 0.5)`, `ang_vel_z=(-1.57, 1.57)`（与 `commands.ranges` 一致）

### 3.2 与 commands 的关系

- 未启用课程（`enable=False`）时：速度指令的采样范围始终由 `commands.ranges`（约 422–427 行）决定。
- 启用课程后：实际采样范围由**当前级别**的 `velocity_curriculum.levels[level]` 决定，环境会把这些值**写回** `command_generator.cfg.ranges` 的 `lin_vel_x/y`, `ang_vel_z`（见下文的 `_apply_velocity_level`）。

---

## 4. 环境实现（roban_envs.py）

**文件**：`legged_lab/envs/roban/roban_envs.py`

### 4.1 初始化（必须在第一次 reset 之前）

- **位置**：约第 **100–101** 行，在 `init_buffers()` 之后、`reset(env_ids)` 之前调用 `_init_velocity_curriculum()`。
- **原因**：`reset()` 内会调用 `_update_velocity_curriculum()`，会读 `self._vel_curriculum_enabled`，因此必须先初始化。

```python
# 约 98–107 行
self.init_buffers()

# ── 速度课程学习初始化（必须在 reset 之前，reset 会调用 _update_velocity_curriculum）──
self._init_velocity_curriculum()

env_ids = torch.arange(self.num_envs, device=self.device)
# ...
self.reset(env_ids)
```

### 4.2 `_init_velocity_curriculum()`（约 608–638 行）

- 根据 `self.cfg.velocity_curriculum` 是否存在且 `enable=True` 设置 `self._vel_curriculum_enabled`。
- 若未启用，直接 return，不创建任何课程相关状态。
- 若启用：
  - 保存配置 `_vel_cur_cfg`，当前级别 `_vel_cur_level = 0`；
  - 分配统计缓冲区：`_vel_cur_ep_lengths`, `_vel_cur_ep_lin_errs`, `_vel_cur_ep_ang_errs`（deque，长度 `buffer_size`）；
  - 每环境跟踪误差累积：`_vel_cur_lin_err_acc`, `_vel_cur_ang_err_acc`（tensor）；
  - 冷却时间戳：`_vel_cur_level_change_step`；
  - 调用 `_apply_velocity_level(0)` 把 Level 0 的速度范围写入 `command_generator.cfg.ranges`。

### 4.3 `_apply_velocity_level(level)`（约 640–646 行）

- 将 `velocity_curriculum.levels[level]` 的 `lin_vel_x`, `lin_vel_y`, `ang_vel_z` 写入 `self.command_generator.cfg.ranges`，从而之后采样的速度指令落在当前级别范围内。

### 4.4 每步：累积跟踪误差 `_accumulate_tracking_error()`（约 648–666 行）

- **调用位置**：在 `step()` 中，约第 **549–551** 行，仅当 `_vel_curriculum_enabled` 时调用。
- **作用**：每 env step 计算当前指令与机器人实际线速度（yaw 系）、角速度的误差，累加到 `_vel_cur_lin_err_acc`、`_vel_cur_ang_err_acc`（按 env 维度）。  
  Episode 结束后用「累积值 / 步数」近似该 episode 的平均跟踪误差，用于升级判断。

```python
# 约 549–551 行（step 内）
if self._vel_curriculum_enabled:
    self._accumulate_tracking_error()
```

### 4.5 Reset 时：`_update_velocity_curriculum(env_ids)`（约 668–767 行）

- **调用位置**：在 `reset(env_ids)` 中，约第 **479–484** 行；只对**本步被 reset 的 env_ids** 传入。
- **流程概要**：
  1. 若未启用课程，直接返回 `{}`。
  2. 对传入的 `env_ids`，取刚结束的 episode 长度与对应 env 的累积跟踪误差，计算该 episode 的平均线/角速度误差。
  3. **冷却过滤**：若该 episode 的起始步数 < `_vel_cur_level_change_step`，说明是旧级别下开始的，丢弃不进入统计。
  4. 将符合条件的 episode 长度与平均误差加入 deque 缓冲区。
  5. 重置这些 env 的 `_vel_cur_lin_err_acc` / `_vel_cur_ang_err_acc`。
  6. 当缓冲区样本数 ≥ `buffer_size // 4` 时：
     - 计算近期**平均 episode 长度**、**平均线速度误差**、**平均角速度误差**；
     - **升级**：若当前未满级且平均长度 > promote 阈值且线/角误差均低于上限，则 level+1，`_apply_velocity_level`，清空缓冲区并设置 `_vel_cur_level_change_step = 当前步 + cooldown_steps`；
     - **降级**：若 level>0 且平均长度 < demote 阈值，则 level-1，同样更新范围和冷却。
  7. 返回用于 TensorBoard 的 extras（如 `Curriculum/velocity_level`, `Curriculum/mean_episode_length` 等），在 482–484 行并入 `self.extras["log"]`。

```python
# 约 479–484 行（reset 内）
vel_cur_extras = self._update_velocity_curriculum(env_ids)
self.extras["log"] = dict()
if vel_cur_extras:
    self.extras["log"].update(vel_cur_extras)
```

---

## 5. 一键关闭（消融实验）

### 5.1 配置关闭

在 `legged_lab/envs/roban/walk_cfg.py` 中，将默认课程改为关闭：

```python
velocity_curriculum: VelocityCurriculumCfg = VelocityCurriculumCfg(enable=False)
```

则环境始终使用 `commands.ranges` 的完整速度范围，不再做级别切换。

### 5.2 命令行关闭（不改配置文件）

**文件**：`legged_lab/scripts/train.py`

- **参数定义**：约第 **34–38** 行，`--no-velocity-curriculum`。
- **应用**：约第 **78–81** 行，若传入该参数，则在加载 env 配置后设置 `env_cfg.velocity_curriculum.enable = False`。

```bash
python legged_lab/scripts/train.py --task=roban_walk --num_envs=4096 --headless --no-velocity-curriculum
```

---

## 6. 小结

| 项目       | 说明 |
|------------|------|
| **为何开启** | 先易后难，避免一开始就全范围速度指令导致难收敛或站桩；升级时同时要求存活时间与跟踪精度，避免站桩升级。 |
| **配置**   | `walk_cfg.py` 中 `VelocityCurriculumCfg`（enable、阈值、levels、buffer、cooldown）。 |
| **初始化** | `roban_envs.py` 中在第一次 `reset` 前调用 `_init_velocity_curriculum()`，设置 `_vel_curriculum_enabled` 并应用 Level 0。 |
| **每步**   | `step()` 中若启用则调用 `_accumulate_tracking_error()` 累积线/角速度跟踪误差。 |
| **Reset**  | `reset()` 中对结束的 episode 调用 `_update_velocity_curriculum(env_ids)`，做冷却过滤、缓冲区统计与升/降级，并写回 `command_generator.cfg.ranges`。 |
| **关闭**   | 配置中 `VelocityCurriculumCfg(enable=False)` 或训练时加 `--no-velocity-curriculum`。 |

以上即为“为什么开启课程学习”以及“如何实现、代码在哪”的完整说明，便于复现与做消融（关闭课程对比实验）。
