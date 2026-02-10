# AMP (Adversarial Motion Priors) 计算流程与数学公式

> 文件位置: `rsl_rl/rsl_rl/algorithms/amp_ppo.py`, `rsl_rl/rsl_rl/modules/discriminator.py`  
> 适用环境: `RobanEnv` (Isaac Lab + RSL-RL)

---

## 目录

1. [概览](#概览)
2. [判别器网络结构](#判别器网络结构)
3. [判别器训练](#判别器训练)
4. [AMP 奖励计算](#amp-奖励计算)
5. [奖励组合机制](#奖励组合机制)
6. [返回计算](#返回计算)
7. [配置参数说明](#配置参数说明)
8. [日志指标解释](#日志指标解释)

---

## 概览

AMP (Adversarial Motion Priors) 是一种模仿学习方法，通过对抗训练学习专家运动风格。核心思想是：

1. **判别器 (Discriminator)**：区分专家运动数据和策略生成的运动数据
2. **策略 (Policy)**：生成与专家风格相似的运动，同时完成目标任务
3. **奖励机制**：策略获得奖励当判别器认为其运动像专家运动

### 训练流程

```
┌─────────────────┐
│  专家数据       │ ──┐
│  (Expert)       │   │
└─────────────────┘   │
                      ├──→ 判别器训练
┌─────────────────┐   │   (MSE Loss)
│  策略数据       │ ──┘
│  (Policy)       │
└─────────────────┘
         │
         ├──→ 计算 AMP 奖励
         │
         └──→ 与任务奖励组合
```

---

## 判别器网络结构

### 网络架构

判别器是一个多层感知机 (MLP)，结构如下：

```
输入: [state, next_state]  (维度: 2 × AMP_obs_dim)
  ↓
MLP Trunk: [1024, 512, 256] (ReLU 激活)
  ↓
Linear Layer: 256 → 1
  ↓
输出: d (标量 logit)
```

### 数学表示

$$
d = \text{Discriminator}(s_t, s_{t+1}) = \text{Linear}(\text{MLP}([s_t; s_{t+1}]))
$$

其中：
- $s_t$：当前 AMP 观测状态（维度：54，包含 21 关节位置 + 21 关节速度 + 12 末端执行器位置）
- $s_{t+1}$：下一时刻 AMP 观测状态
- $[s_t; s_{t+1}]$：拼接后的输入向量（维度：108）
- $d$：判别器输出 logit（标量）

### 代码实现

```python
# rsl_rl/rsl_rl/modules/discriminator.py
d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
```

---

## 判别器训练

### 训练目标

判别器被训练为：
- **专家数据** → 输出接近 **+1**
- **策略数据** → 输出接近 **-1**

### 损失函数

#### 1. MSE 损失

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{2} \left( \mathcal{L}_{\text{expert}} + \mathcal{L}_{\text{policy}} \right)
$$

其中：

$$
\mathcal{L}_{\text{expert}} = \text{MSE}(d_{\text{expert}}, 1) = \frac{1}{N} \sum_{i=1}^{N} (d_{\text{expert}}^{(i)} - 1)^2
$$

$$
\mathcal{L}_{\text{policy}} = \text{MSE}(d_{\text{policy}}, -1) = \frac{1}{N} \sum_{i=1}^{N} (d_{\text{policy}}^{(i)} + 1)^2
$$

#### 2. 梯度惩罚 (Gradient Penalty)

为了稳定训练，对专家数据添加梯度惩罚：

$$
\mathcal{L}_{\text{grad\_pen}} = \lambda \cdot \mathbb{E}_{s \sim \text{expert}} \left[ (\|\nabla_s D(s)\|_2 - 0)^2 \right]
$$

其中 $\lambda = 10$ 是梯度惩罚系数。

#### 3. 总损失

$$
\mathcal{L}_{\text{discriminator}} = \mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{grad\_pen}}
$$

### 代码实现

```python
# rsl_rl/rsl_rl/algorithms/amp_ppo.py (第 458-464 行)
policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))

expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
amp_loss = 0.5 * (expert_loss + policy_loss)

grad_pen_loss = self.discriminator.compute_grad_pen(*sample_amp_expert, lambda_=10)
loss += self.amploss_coef * amp_loss + self.amploss_coef * grad_pen_loss
```

---

## AMP 奖励计算

### 奖励公式

AMP 奖励基于判别器输出 $d$，使用二次函数映射：

$$
r_{\text{amp}} = \text{reward\_scale} \cdot \max\left(0, 1 - \frac{1}{4}(d - 1)^2\right)
$$

### 公式解释

- **当 $d = 1$**（判别器认为像专家）：$r_{\text{amp}} = \text{reward\_scale} \cdot 1 = 2.0$（最大值）
- **当 $d = -1$ 或 $d = 3$**：$r_{\text{amp}} = 0$（最小值）
- **当 $d \in (-1, 3)$**：奖励在 $[0, 2.0]$ 之间平滑变化

### 奖励曲线

```
r_amp
  ↑
2.0|     ●
   |    / \
   |   /   \
   |  /     \
   | /       \
 0.0|─────────●───────●──→ d
   -1        1        3
```

### 代码实现

```python
# rsl_rl/rsl_rl/modules/discriminator.py (第 174 行)
d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
reward = reward_scale * torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)
```

### 参数说明

- `reward_scale` = `discriminator_reward_scale` = **2.0**（默认值）
- 单步 AMP 奖励范围：**[0, 2.0]**
- 单步 AMP 奖励平均值：约 **0.72**（根据训练日志）

---

## 奖励组合机制

### 组合公式

最终奖励是任务奖励和 AMP 风格奖励的加权组合：

$$
r_{\text{total}} = w_{\text{task}} \cdot r_{\text{task}} + w_{\text{style}} \cdot r_{\text{amp}}
$$

### 默认参数

根据 `walk_cfg.py` 配置：

- `task_reward_weight` = **1.0**
- `style_reward_weight` = **0.02**
- `discriminator_reward_scale` = **2.0**

### 实际计算

$$
r_{\text{total}} = 1.0 \cdot r_{\text{task}} + 0.02 \cdot r_{\text{amp}}
$$

### AMP 贡献分析

假设：
- 单步任务奖励平均值：$r_{\text{task}} \approx 0.08$
- 单步 AMP 奖励平均值：$r_{\text{amp}} \approx 0.72$

则：
- 单步总奖励：$r_{\text{total}} = 1.0 \times 0.08 + 0.02 \times 0.72 = 0.0944$
- AMP 贡献比例：$\frac{0.02 \times 0.72}{0.0944} \approx 15.3\%$

### 最大 AMP 贡献

当 $r_{\text{amp}} = 2.0$（最大值）时：
- AMP 贡献：$0.02 \times 2.0 = 0.04$
- 如果任务奖励为 $r_{\text{task}} = 1.0$，则 AMP 贡献比例：$\frac{0.04}{1.04} \approx 3.8\%$

### 代码实现

```python
# rsl_rl/rsl_rl/algorithms/amp_ppo.py (第 240-243 行)
if self.use_amp_reward_combination:
    self.storage.rewards = (
        self.task_reward_weight * self.storage.task_rewards +
        self.style_reward_weight * self.storage.amp_rewards
    )
```

### 奖励计算时机

1. **环境步骤**：分别计算任务奖励和 AMP 奖励，暂不组合
2. **返回计算前**：在 `compute_returns()` 中组合奖励
3. **原因**：确保 bootstrapping 基于任务奖励，然后组合 AMP 奖励

---

## 返回计算

### GAE (Generalized Advantage Estimation)

组合后的奖励用于计算优势函数和返回值：

#### TD 误差

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

其中：
- $r_t$：组合后的总奖励
- $\gamma = 0.99$：折扣因子
- $V(s_t)$：价值函数估计

#### 优势函数

$$
A_t = \delta_t + (\gamma \lambda) A_{t+1}
$$

其中 $\lambda = 0.95$ 是 GAE 参数。

#### 返回值

$$
R_t = A_t + V(s_t)
$$

### 代码实现

```python
# rsl_rl/rsl_rl/storage/rollout_storage.py (第 168-183 行)
def compute_returns(self, last_values, gamma, lam, normalize_advantage=True):
    advantage = 0
    for step in reversed(range(self.num_transitions_per_env)):
        if step == self.num_transitions_per_env - 1:
            next_values = last_values
        else:
            next_values = self.values[step + 1]
        
        next_is_not_terminal = 1.0 - self.dones[step].float()
        delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
        advantage = delta + next_is_not_terminal * gamma * lam * advantage
        self.returns[step] = advantage + self.values[step]
    
    self.advantages = self.returns - self.values
    if normalize_advantage:
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
```

---

## 配置参数说明

### 判别器配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `amp_discr_hidden_dims` | `[1024, 512, 256]` | 判别器隐藏层维度 |
| `amp_num_preload_transitions` | `200000` | 预加载的专家转换数量 |
| `amploss_coef` | `1.0` | 判别器损失系数 |

### 奖励组合配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `task_reward_weight` | `1.0` | 任务奖励权重 |
| `style_reward_weight` | `0.02` | AMP 风格奖励权重 |
| `discriminator_reward_scale` | `2.0` | 判别器奖励缩放因子 |

### 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gamma` | `0.99` | 折扣因子 |
| `lam` | `0.95` | GAE 参数 |
| `grad_pen_lambda` | `10.0` | 梯度惩罚系数 |

---

## 日志指标解释

### 训练日志中的关键指标

#### 1. 判别器损失

```
Mean amp loss: 0.1814
Mean amp_grad_pen loss: 0.2029
```

- `amp_loss`：MSE 损失（专家和策略）
- `amp_grad_pen_loss`：梯度惩罚损失

#### 2. 判别器预测

```
Mean amp_policy_pred loss: -0.6131
Mean amp_expert_pred loss: 0.6099
```

- `amp_policy_pred`：策略数据的判别器输出（目标：-1）
- `amp_expert_pred`：专家数据的判别器输出（目标：+1）

#### 3. 奖励统计

```
Mean reward: 63.38
Mean task reward: 63.3753
Mean AMP reward: 571.8891
AMP contribution: 15.3%
```

**重要说明**：
- `Mean reward`：组合后的总奖励（episode 累计）
- `Mean task reward`：任务奖励（episode 累计）
- `Mean AMP reward`：AMP 奖励（episode 累计，**未加权**）
- `AMP contribution`：加权后的 AMP 贡献比例

**计算示例**：
- Episode 长度：792 步
- 单步任务奖励：$63.38 / 792 \approx 0.08$
- 单步 AMP 奖励：$571.89 / 792 \approx 0.72$
- 单步总奖励：$1.0 \times 0.08 + 0.02 \times 0.72 = 0.0944$
- Episode 总奖励：$0.0944 \times 792 \approx 74.8$
- AMP 贡献：$\frac{0.02 \times 571.89}{74.8} \approx 15.3\%$

---

## 总结

### 关键公式汇总

1. **判别器输出**：
   $$
   d = \text{Discriminator}(s_t, s_{t+1})
   $$

2. **AMP 奖励**：
   $$
   r_{\text{amp}} = 2.0 \cdot \max\left(0, 1 - \frac{1}{4}(d - 1)^2\right)
   $$

3. **总奖励**：
   $$
   r_{\text{total}} = 1.0 \cdot r_{\text{task}} + 0.02 \cdot r_{\text{amp}}
   $$

4. **判别器损失**：
   $$
   \mathcal{L} = \frac{1}{2}\left[\text{MSE}(d_{\text{expert}}, 1) + \text{MSE}(d_{\text{policy}}, -1)\right] + 10 \cdot \mathcal{L}_{\text{grad\_pen}}
   $$

### 设计要点

1. **奖励范围**：单步 AMP 奖励在 $[0, 2.0]$ 之间
2. **贡献控制**：通过 `style_reward_weight = 0.02` 控制 AMP 贡献在约 15% 左右
3. **训练稳定**：使用梯度惩罚和 MSE 损失确保判别器训练稳定
4. **风格学习**：策略学习模仿专家运动风格，同时完成目标任务

---

## 参考资料

- 代码位置：
  - `rsl_rl/rsl_rl/algorithms/amp_ppo.py`
  - `rsl_rl/rsl_rl/modules/discriminator.py`
  - `rsl_rl/rsl_rl/runners/amp_on_policy_runner.py`
  - `legged_lab/envs/roban/walk_cfg.py`

- 相关文档：
  - `legged_lab/scripts/motion_metrics.md`
  - `legged_lab/scripts/reward_formulas.md`
