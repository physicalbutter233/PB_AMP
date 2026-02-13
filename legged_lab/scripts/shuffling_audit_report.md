# 四足/双足机器人「蹭地/拖脚」行为审计报告

## 审计目标
诊断导致机器人出现「蹭地」（不抬脚、脚在地面拖行）的配置与奖励因素。  
原理：当**抬腿成本（能量/稳定性）> 拖行成本（摩擦/惩罚）**时，策略会倾向于拖脚。

---

## 审计结果汇总表

| Category | Parameter Name | Current Value | Risk Level | Why it causes Shuffling |
| :--- | :--- | :--- | :--- | :--- |
| **Rewards** | `feet_slide` (base weight) | `-0.5` | **High** | 基础惩罚偏弱；Level 1 乘数仅 1.0，相对 Level 0 的 10.0 大幅减弱，升级后蹭地惩罚骤降，易重新出现拖脚。 |
| **Rewards** | `feet_slide` (Level 0 multiplier) | `10.0` | Low | Level 0 已加强，有效约 -5.0，有利于治蹭地。 |
| **Rewards** | `feet_slide` (Level 1 multiplier) | `1.0` | **High** | 从 10→1 后蹭地惩罚仅为 Level 0 的 1/10，策略容易在 Level 1 又学会拖脚。 |
| **Rewards** | `feet_air_time` (hard) `threshold_min` | `0.25` s | **High** | 滞空 <0.25s 时**无任何**硬阈值奖励；初学者抬腿时间短时得不到正反馈，容易放弃抬腿。 |
| **Rewards** | `feet_air_time` (hard) `threshold_max` | `0.4` s | Low | 上限合理，防大跨步。 |
| **Rewards** | `feet_air_time_easy` `threshold_min` | `0.1` s | Low | Level 0 下 0.1s 即给奖，有助于破稀疏奖励。 |
| **Rewards** | `feet_height` (weight) | `12.0`，clip `0.15` m | Medium | 抬脚高度奖励存在且合理；Level 1 乘数 0.6 会削弱该奖励，与 feet_slide 减弱叠加易导致蹭地回潮。 |
| **Rewards** | `feet_height` (Level 1 multiplier) | `0.6` | Medium | 抬脚奖励在 Level 1 被压低，相对「拖脚惩罚减弱」而言，抬脚动机不足。 |
| **Rewards** | `base_height_exp` (target_height) | `0.65` m, std `0.1` | Low | 站直有明确目标，Level 0 乘数 2.0 有利于抬 CoM、减少屈膝蹭地。 |
| **Rewards** | `feet_force` (body_force) | weight `-0.01`, threshold `500`, max `400` | Low | 仅对**极大**接触力惩罚，力度小（约 -4 封顶），不会明显抑制正常迈步。 |
| **Physics** | Ground (scene) `static_friction` | `1.0` | Low | 地面静摩擦不低，拖行物理成本足够。 |
| **Physics** | Ground `dynamic_friction` | `1.0` | Low | 动摩擦 1.0，拖行代价不低。 |
| **Physics** | Domain rand friction (robot body) | static (0.6, 1.0), dynamic (0.4, 0.8) | Low | 仅作用于**机器人本体**，不改变地面；地面仍为 1.0/1.0。 |
| **Physics** | Leg stiffness (e.g. leg 2/4) | 99.1 (leg 2/4), 40.2 (1/3), 14.3 (5/6) | Low | 髋/膝刚度足够，具备蹬地能力。 |
| **Physics** | Leg damping (e.g. leg 2/4) | 6.31 (leg 2/4), 2.56 (1/3), 0.91 (5/6) | Low | 阻尼在常见范围内。 |
| **Terrain** | `terrain_type` / generator | `terrain_type="generator"`, `GRAVEL_TERRAINS_CFG` | Low | 使用碎石地形（random_rough），非纯平面，对拖脚有一定自然惩罚。 |
| **Terrain** | GRAVEL curriculum | `curriculum=False` | Medium | 无地形课程，所有 env 同难度；若希望更强「抬脚压力」，可考虑启用 curriculum 或更难地形。 |

---

## 结论与优先修改建议

### 导致蹭地的核心逻辑
1. **蹭地惩罚在 Level 1 过弱**：`feet_slide` 乘数从 10.0 降到 1.0，拖脚成本骤降。  
2. **硬滞空门槛过高**：`feet_air_time` 的 `threshold_min=0.25` 使短时抬腿（如 0.1–0.2s）得不到奖励，抬腿正反馈不足。  
3. **Level 1 抬脚相关奖励被压低**：`feet_height` 乘数 0.6，与 `feet_slide` 减弱叠加，抬腿的「收益」相对「拖脚成本」不足。

### 建议优先修改的 3 个变量（按优先级）

| 优先级 | 变量 | 当前值 | 建议调整 | 理由 |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Level 1 `feet_slide` 乘数** | `1.0` | 改为 **3.0～5.0**（或保持 10.0 到 Level 2 再降） | 避免一升级就大幅减轻蹭地惩罚，让「拖脚成本」在 Level 1 仍足够高。 |
| **2** | **`feet_air_time` (hard) `threshold_min`** | `0.25` s | 改为 **0.15～0.2** s | 让 0.15–0.25s 的短滞空也能拿到部分奖励，避免策略因拿不到奖励而放弃抬腿。 |
| **3** | **Level 1 `feet_height` 乘数** | `0.6` | 改为 **1.0**（或 0.8～1.0） | 在 Level 1 保持足够的「抬脚高度」正反馈，与加强的 feet_slide 配合，使抬腿收益 > 拖脚收益。 |

### 可选后续优化
- **`feet_air_time_easy` 在 Level 1 的权重**：若仍偏蹭地，可保持 Level 1 的 easy 乘数在 0.5 或略提高到 0.7，让短滞空奖励更明显。  
- **地形**：若希望进一步压制蹭地，可尝试 `curriculum=True` 或加入更多粗糙/台阶地形，使拖脚在物理上更吃亏。  
- **`feet_slide` 基础 weight**：若全局仍偏弱，可将 base weight 从 `-0.5` 提到 `-0.8` 或 `-1.0`，再配合乘数微调。

---

## 涉及文件与位置速查

| 参数 | 文件 | 位置（约） |
| :--- | :--- | :--- |
| `reward_multipliers` (Level 0/1 feet_slide, feet_height, feet_air_time) | `legged_lab/envs/roban/walk_cfg.py` | 134–166 行 |
| `feet_air_time` / `feet_air_time_easy` params (threshold_min/max) | `legged_lab/envs/roban/walk_cfg.py` | 272–294 行 |
| `feet_slide` / `feet_force` base definition | `legged_lab/envs/roban/walk_cfg.py` | 251–268 行 |
| Ground friction (physics_material) | `legged_lab/envs/roban/roban_envs.py` | 69–73 行 |
| Stiffness / damping | `legged_lab/assets/roban_s14/roban_s14.py` | 101–125 行 |
| Terrain | `legged_lab/envs/roban/walk_cfg.py` | 473–476 行；`legged_lab/terrains/terrain_generator_cfg.py` |

---

*本报告由 Legged Robot Sim-to-Real Auditor 根据当前代码库静态审计生成。*
