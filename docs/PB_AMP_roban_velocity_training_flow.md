# PB_AMP 中 Roban 速度跟踪训练流程

本文档以与 amp_share 相同的框架描述 PB_AMP 中 roban 速度跟踪（roban_walk）的训练流程。

---

## 1. 整体流程概览

- **加载参数与配置**：通过 `task_registry` 按任务名（如 `roban_walk`）取环境配置与智能体配置。
- **创建环境**：用环境类 `RobanEnv` 与 `RobanWalkFlatEnvCfg` 实例化 `env`。
- **创建 Runner**：`train.py` 中根据 `agent_cfg.runner_class_name`（如 `"AmpOnPolicyRunner"`）实例化 `AmpOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=..., device=...)`，内部会创建 **agent**（算法对象，含策略、价值网络、判别器、与观测/动作空间及 AMP 数据集的接口）和训练所需存储，Runner 负责环境与 agent 的交互循环。
- **开始训练**：调用 `runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)`，在单智能体（单进程多环境）下按配置的迭代次数与每轮步数执行训练。

---

## 2. 训练循环结构（单智能体）

PB_AMP 的 `AmpOnPolicyRunner.learn()` **没有** `pre_interaction` / `post_interaction` 钩子，而是**内联**实现：每轮迭代先做 rollout，再 `compute_returns`，再 `update`。

### 2.1 初始化

- 若 `init_at_random_ep_len=True`，对 `env.episode_length_buf` 做随机初始化，用于随机化初始 episode 长度。
- 获取初始 `obs`、`privileged_obs`（critic 观测）、`amp_obs`，并切换到训练模式。

### 2.2 主循环（按 `max_iterations` 迭代）

对每一次迭代 `it`：

1. **Rollout 阶段**（`torch.inference_mode()` 下，共 `num_steps_per_env` 步，默认 24）：

   - **根据当前观测输出 action**  
     `actions = self.alg.act(obs, privileged_obs, amp_obs)`，策略输入见下文「策略输入」。
   - **环境步进**  
     `obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))`：
     - 在 `env.step(actions)` 内，根据 **decimation**（默认 4）执行 4 次仿真步（每步 `sim.dt`），**这 4 步共用同一组 actions**。
     - **Transition 只使用两帧**：**step 开始前那一帧**的 state/obs，与 **4 个仿真步结束后**的 state 得到 `next_obs`、`rewards`、`terminated`/`truncated`（在 PB_AMP 里合并为 `dones`）、`infos`。
     - 即：**不会**把中间 3 帧单独当作 RL 的 step，只记录「step 前 → 4 步后」这一条 transition。
   - **观测与奖励处理**  
     - `next_amp_obs = self.env.get_amp_obs_for_expert_trans()`，用于 AMP。
     - `privileged_obs` 来自 `infos["observations"]["critic"]`（若存在 state_space/critic 观测），否则与 actor 观测一致。
     - 若启用 `use_amp_reward_combination`：rollout 中只存 **task_rewards** 和 **amp_states**，不在此处算 style reward；否则会用判别器当场算 AMP 奖励。
   - **记录 transition**  
     `self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term, task_rewards, amp_rewards)`：
     - 将 **task_rewards**、**amp_states**（以及可选的 amp_rewards）写入 agent 的 storage（rollout buffer）与 amp_storage，供后续算 return/advantage 和 AMP style reward。
     - 对 timeout 做 value bootstrap（若 `infos` 中有 `time_outs`）。
     - `obs` / `privileged_obs` / `amp_obs` 更新为下一步的观测，供下一轮 `act` 使用。

2. **Rollout 结束后**  
   - **计算 return / advantage**  
     `self.alg.compute_returns(privileged_obs)`：  
     - 若启用 `use_amp_reward_combination`：用当前 rollout 中存储的 **amp_states** 批量算 **style reward**，再组合  
       `storage.rewards = task_reward_weight * task_rewards + style_reward_weight * style_reward`。  
     - 在此基础上用 GAE 等算 returns/advantages。

3. **更新**  
   `loss_dict = self.alg.update()`：  
   - **判别器**：策略端只用**当前这批 rollout**（最近 `num_steps_per_env * num_envs` 条）的 AMP 转移，打乱后按 mini-batch 采样；专家端从 **AMP 数据集**（`amp_data`）中按 batch 采样；两者一起参与判别器更新（并可选对称性损失）。  
   - **策略与价值网络**：只使用**当前 rollout** 的 buffer（`storage`），通过 `num_learning_epochs * num_mini_batches` 次 mini-batch 更新。

4. **必要时重置环境**  
   在 `env.step(actions)` 内部，根据 `reset_buf` 对相应 `env_ids` 调用 `env.reset(env_ids)`；多环境时，未 reset 的 env 的 `obs` 在 runner 中即为上一步的 `obs`（即 `next_obs`），无需额外赋值。

5. **每个 step 结束后的场景/课程**  
   - 在 **`env.step(actions)` 末尾**：  
     - `command_generator.compute(self.step_dt)` 更新速度命令（为下一个 step 准备）。  
     - `event_manager.apply(mode="interval", dt=self.step_dt)`：**interval** 类事件在本 step 结束后生效，影响**下一个 step** 的仿真/场景。  
   - 在 **`env.reset(env_ids)` 内**：  
     - **velocity_curriculum**（若启用）：根据刚结束的 episode 的存活率与跟踪误差，在 `_update_velocity_curriculum(env_ids)` 中可能升/降级，并调用 `_apply_velocity_level(level)` 更新 command 的 ranges 与奖励权重。  
     - **terrain / 其他课程**：地形等级、外力等可在 reset 或 event 中更新。  
     - `event_manager.apply(mode="reset", ...)`：**reset** 类事件在 reset 时生效。

---

## 3. 数量关系（与 amp_share 对齐的表述）

- 若 **max_iterations = 100**，**num_envs = 4096**，**num_steps_per_env = 24**：
  - 每个 iteration 产生的 **transition 总数** = 4096 × 24。
  - **Rollout 次数（timesteps）** = 100 × 24（即 100 个 iteration，每个 24 个 env-step）。
  - **Agent 更新次数** = 100（每个 iteration 一次 `update()`）。
  - **参数更新次数** = 100 × num_learning_epochs × num_mini_batches（例如 100 × 5 × 4 = 2000）。

---

## 4. 判别器与策略/价值更新（与 amp_share 一致的理解）

- **判别器**  
  - 策略侧：仅使用**当前一次 rollout** 的 `num_steps_per_env * num_envs` 条 AMP 转移，打乱后按 mini-batch 采样。  
  - 专家侧：从 **AMP 专家数据池**（`amp_data`，多轨迹 npz 等）中按 batch 采样；可与既有专家 buffer 混合后随机采样。  
  - 每次 `update()` 内会进行多轮 mini-batch（`num_learning_epochs * num_mini_batches`），判别器与策略/价值一起在这些 batch 上迭代。

- **策略与价值网络**  
  - 仅基于**当前 rollout** 的 `storage`（最近 24 × num_envs 条 transition）做 return/advantage 计算与 PPO 更新，不跨 rollout 使用旧数据。

---

## 5. Decimation 与 env.step(actions)

- **sim.dt**：例如 0.005 s；**decimation = 4**。  
- **一次 `env.step(actions)`**：内部执行 4 次仿真步（4 × sim.dt），**4 步均使用同一组 actions**。  
- **Transition 记录**：只使用「step 前那一帧」和「4 步后那一帧」两帧状态得到 `obs`、`next_obs`、`rewards`、`dones`、`infos`，中间 3 帧不单独作为 RL 的 step。

---

## 6. 策略输入与命令（速度跟踪）

- **env.step(actions) 的 actions**：来自当前步策略输出（以及可选的 action delay 等处理）。  
- **策略输入**：与当前/上一时刻状态及本步命令相关，主要包括：  
  - 角速度、重力投影、**当前步的 command**（`velocity_commands`）；  
  - 关节位置/速度、上一步 action 等；  
  - 若使用历史，则为多帧拼接（如 `actor_obs_history_length=10`）。  
- **Command**：由环境中的 **command_generator**（如 `UniformVelocityCommand`）生成，配置在 `CommandsCfg` 的 `ranges`（如 `lin_vel_x`、`lin_vel_y`、`ang_vel_z`）。  
  - 若启用 **velocity_curriculum**：在 reset 时通过 `_apply_velocity_level(level)` 动态修改 `command_generator.cfg.ranges` 和部分奖励权重，实现「先动量、后精度」的速度课程。  
  - 每个 **env step 结束后**调用 `command_generator.compute(self.step_dt)`，下一个 step 的观测中会带上更新后的 command。

---

## 7. 与 amp_share：相同部分 vs 区别部分

### 7.1 相同部分

- **整体流水**：加载参数/配置 → 创建环境 → 用 Runner 实例化 agent（策略、价值、判别器 + 与 env/AMP 数据集接口）和 trainer 逻辑 → `learn()` 里按 timesteps 做「rollout → 记录 transition → 算 return/advantage → update」。
- **每轮步数**：每轮迭代都是 **24 个 env-step**（num_steps_per_env=24），即 24 条 transition 每环境。
- **Decimation**：一次 `env.step(actions)` 内做 **4 个仿真步**（decimation=4），4 步共用同一组 actions；**transition 只记「step 前那一帧」和「4 步后那一帧」**，中间 3 帧不单独成 RL step。
- **策略输入**：当前/上一时刻状态（角速度、重力、关节位姿/速度、上一步 action）+ **当前步的 velocity command**（由 command_manager/command_generator 给出）。
- **AMP 奖励逻辑**：rollout 时只存 **task reward** 和 **amp_obs/amp_states**；**style reward 在 compute_returns 时批量算**，再按权重组合成最终 reward（task_reward_weight × task + style_reward_weight × style）。
- **判别器数据**：策略侧只用**当前一次** 24×num_envs 条策略 AMP 转移，打乱后 mini-batch 采样；专家侧从专家数据池（可混合后）随机采样。
- **策略/价值更新**：只看**当前 rollout** 的 buffer（24×num_envs 条），不跨 rollout 用旧数据；每次迭代一次 `update()`，内部为 num_learning_epochs × num_mini_batches 次参数更新。
- **数量关系**：总 transition = num_envs × max_iterations × 24；rollout 次数 = max_iterations × 24；agent 更新次数 = max_iterations；参数更新次数 = max_iterations × num_learning_epochs × num_mini_batches。
- **事件时机**：**上一步修改的 events 在本 step 的 4 个仿真步中生效**；**本 step 结束后**设置/应用下一 step 要用的 event（interval 在 step 末，reset 在 reset 时）。

### 7.2 区别部分

| 项目 | amp_share（你描述的流程） | PB_AMP roban_walk |
|------|---------------------------|-------------------|
| **循环结构 / 钩子** | 有 `agent.pre_interaction(timestep, timesteps)` 与 `agent.post_interaction(timestep, timesteps)`；post 里做「rollout+1、每 24 step 调一次 update」等。 | **无** pre/post 钩子；rollout、compute_returns、update **全部内联**在 `learn()` 的一个 for 里：先 for 24 步 step+process_env_step，再 compute_returns，再 update。 |
| **state / critic 观测** | 你写的是 `state = infos["observations"]["critic"] if self.env.state_space else observations`。 | 一致：`privileged_obs` 来自 `infos["observations"]["critic"]`，没有则用 actor 观测。 |
| **速度课程** | 你只写了「根据情况和课程安排对下一个 step 的场景做调整」，未细化速度课程。 | 显式 **VelocityCurriculumCfg**：在 reset 时根据 episode 存活率与跟踪误差升/降级，`_apply_velocity_level(level)` 改 command ranges（lin_vel_x/y, ang_vel_z）和部分奖励权重；可选 `--no-velocity-curriculum` 关掉。 |
| **事件/场景 API** | 用「events」「课程安排」等概括。 | 具体为 **event_manager**：`mode="interval"` 在 step 末、`mode="reset"` 在 reset 时；地形/推力等课程在 reset 或 event 里改。 |

### 7.3 小结表（与原差异小节合并）

| 项目 | amp_share | PB_AMP roban_walk |
|------|-----------|-------------------|
| 钩子 | pre_interaction / post_interaction | 无；内联 rollout → compute_returns → update |
| 每轮步数 | 24 | 24 |
| Transition | step 前一帧 + 4 步后一帧 | 同左（decimation=4） |
| AMP 奖励 | rollout 存 task，compute_returns 时算 style | 同左（use_amp_reward_combination） |
| 速度课程 | 可选（你未细写） | VelocityCurriculumCfg，reset 时改 ranges + 奖励权重 |
| 事件 | 上一步 events 本步生效，本步末设下一步 | 同：interval 在 step 末，reset 在 reset 时 |

以上即为 PB_AMP 中 roban 速度跟踪训练的完整流程描述，以及与 amp_share 的**相同部分**和**区别部分**。

---

## 8. 两者在实现上是否训练效果一致？

**结论：在相同配置与设定下，两者在实现上理应训练效果一致。**

### 8.1 为何等价

- **执行顺序一致**：无论用 pre/post_interaction 钩子还是内联 for 循环，每轮迭代实际执行的都是：  
  「24 次 (act → step → process_env_step) → compute_returns → update」。  
  钩子只是把同一逻辑拆成「每步 pre / 每步 post、满 24 步在 post 里调 update」，与 PB_AMP 的「先 for 24 步再 compute_returns 再 update」在**数据与调用顺序上等价**。
- **数据与算法一致**：transition 写法、AMP 奖励（rollout 存 task+amp_states，compute_returns 时批量算 style 再加权）、判别器/策略用的数据范围、GAE 与 PPO 更新方式在两边一致，因此**同一批配置下理论上应得到相同训练曲线**（在相同随机种子下可期望数值上非常接近）。

### 8.2 要“效果一致”需满足的条件

- **超参与环境一致**：如 num_envs、max_iterations、num_steps_per_env(24)、learning_epochs、mini_batches、reward 权重、网络结构、AMP 数据与权重等一致。
- **课程与事件一致**：若做严格对比，速度课程（是否开启、各级 ranges/奖励权重）、地形/推力等课程、event 触发方式需对齐，否则差异来自**任务设定**而非「循环实现」。
- **随机种子**：同一 seed 时更易复现；不同 seed 时仍应属于同一分布，只是单次曲线会有波动。

### 8.3 可能产生细微差异的地方

- **不同代码库**：浮点顺序、归一化/裁剪的实现细节、日志或调试逻辑若不同，可能带来极小数值差异，一般不影响收敛性与最终性能。
- **未对齐的配置**：若 amp_share 侧有未写明的默认值（如某 reward 权重、课程阶段数），需与 PB_AMP 逐项对齐后再比较。

**总结**：钩子 vs 内联只是代码结构不同，**算法与数据流一致**，因此在实现上两者理应训练效果一致；做严格对比时注意配置与课程、种子对齐即可。
