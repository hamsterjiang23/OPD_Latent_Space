# On-Policy Distillation (OPD) 介绍

> 基于 Thinking Machines Lab 博客 (Kevin Lu, 2025.10) 整理
> 原文链接：https://thinkingmachines.ai/blog/on-policy-distillation/

---

## 一、背景：LLM 训练三阶段

| 阶段 | 目标 | 学到什么 |
|------|------|---------|
| **Pre-training** | 通用能力 | 语言使用、广泛推理、世界知识 |
| **Mid-training** | 领域知识 | 代码、医疗数据、内部文档 |
| **Post-training** | 目标行为 | 指令遵循、数学推理、对话 |

核心理念：**小模型 + 强训练 > 大模型 + 弱训练**。小模型可本地部署（隐私/安全）、可持续训练更新、推理成本低。

---

## 二、Post-training 的两种范式

### On-Policy vs Off-Policy

| | On-Policy 训练 | Off-Policy 训练 |
|---|---|---|
| **数据来源** | 学生模型自己采样生成 | 外部来源的固定目标输出 |
| **代表方法** | RL、OPD | SFT、传统蒸馏 |
| **反馈** | 对学生自己行为的评价 | 模仿预设的正确答案 |

### 问题场景：训练小模型解数学题

**RL（On-Policy + 稀疏奖励）**：
- 学生生成 rollout → 只判断最终答案对不对
- 优点：在自身分布上训练，直接学习避免错误
- 致命缺陷：**只教 O(1) bits/episode**，不告诉学生哪一步错了——是运算顺序错了还是算术错了？

**Off-Policy 蒸馏（Off-Policy + 密集奖励）**：
- 学生学习教师模型生成的完整推理轨迹
- 优点：每步都有 token 级监督信号
- 致命缺陷：学生在教师常去的状态上训练，而不是自己常去的状态 → **复合误差（compounding error）**：学生早期犯一个小错 → 漂移到训练时从未见过的状态 → 越来越离谱

**国际象棋类比**：
- RL = 下棋无教练，只知道最终输赢，不知道哪步是败招
- Off-Policy 蒸馏 = 看大师下棋，但大师的棋局状态新手根本打不出来
- **OPD = 每一步都给你的棋打分（从"大漏招"到"精彩"）**

---

## 三、OPD 核心思路

```
OPD = On-Policy 采样（学生自己生成）+ Dense 监督（教师逐 token 评分）
```

| 方法 | 采样来源 | 奖励密度 |
|------|---------|---------|
| SFT (Off-Policy 蒸馏) | 教师 | 密集 |
| RL | 学生 | 稀疏 |
| **OPD** | **学生** | **密集** |

---

## 四、实现细节

### Loss Function：Reverse KL

OPD 最小化学生分布 π_θ 与教师分布 π_teacher 之间的逐 token reverse KL：

```
KL(π_θ || π_teacher) = E_{x~π_θ}[ log π_θ(x_{t+1}|x_{1..t}) - log π_teacher(x_{t+1}|x_{1..t}) ]
```

**Reverse KL 的三个优良性质**：

1. **Mode-seeking**：学习教师的某一个特定行为模式，而非分散到多个次优选项
2. **"不可被 hack"**：低 KL 一定对应教师视角下的高概率行为（与 reward model 可能被 exploit 不同）
3. **减少 exposure bias**：学生始终在自身分布上被评估

**为什么 discount factor = 0**：实践中，只优化 immediate next token（不考虑未来 token）效果足够好，且计算更简单。

### 伪代码

```python
# 1. 初始化教师 client
teacher_client = create_sampling_client(teacher_model)

# 2. 学生采样 rollout
trajectories = student.sample(prompts)
sampled_logprobs = trajectories.logprobs  # 学生的 log probs

# 3. 教师对学生的 token 计算 log probs（单次前向传播！）
teacher_logprobs = teacher_client.compute_logprobs(trajectories)
reverse_kl = sampled_logprobs - teacher_logprobs

# 4. 用负 reverse KL 作为 advantage，做 RL 风格的更新
trajectories["advantages"] = -reverse_kl
student.train(trajectories, loss_fn="importance_sampling")
```

**关键效率优势**：
- 不需要等 rollout 完成就能计算 reward（可以训练更短/部分 rollout）
- 教师只需一次 forward pass 计算 logprobs
- 轨迹由更小/更便宜的学生模型生成

---

## 五、实验一：数学推理（Reasoning）

### 设置
- 学生：Qwen3-8B-Base
- 教师：Qwen3-32B（实际用 Qwen3-8B 作为 teacher 也略好）
- 数据：OpenThoughts-3（400K prompts，QwQ-32B 生成）
- 任务：AIME'24（数学竞赛题）

### 结果

| 方法 | AIME'24 | 算力节省 |
|------|---------|---------|
| SFT-400K（初始化） | 60% | — |
| SFT-2M（外推） | ~70%（外推） | 1× baseline |
| RL（Qwen3 报告） | 67.6% | ≈1×（17,920 GPU hours） |
| **OPD（~150 步）** | **70%** | **9-30×** |

- OPD 在 ~150 步内达到 SFT-2M 外推的同等性能
- 当 SFT 数据已存在时，算力节省 **9×**（GPU hours 约 **18×**）
- 当 SFT 数据不存在、需计入教师采样成本时，节省 **30×**
- LoRA 在 SFT 后落后 full finetune 13%，但在 OPD 后仅落后 6% —— OPD 缩小了 LoRA 与 full finetune 的差距

---

## 六、实验二：个性化 + 持续学习（Personalization）

### 问题
训练一个内部公司助手，需要同时满足：
1. **知识**：公司文档（预训练模型从未见过）→ 内部 QA 评测
2. **行为**：指令遵循（instruction following）→ IF-Eval 评测

### 两难困境

直接 fine-tune Qwen3-8B（已 post-trained）到公司文档上 → **灾难性遗忘**：
- 更多文档数据 → 知识提升
- 但指令遵循能力崩溃
- 混合 chat 背景数据（最高 70%）能减缓但**无法阻止**退化
- LoRA 也不行：学得更少，忘得还是一样多

### OPD 解决方案

**Phase 1 — Mid-training**：70% 文档 + 30% chat 数据混合 fine-tune

**Phase 2 — OPD 恢复行为**：用**原始 Qwen3-8B 作为教师**，在 Tulu3 prompts 上做 OPD（此阶段与文档数据无关，只恢复指令遵循能力）

| 阶段 | 内部 QA（知识） | IF-Eval（行为） |
|------|:---:|:---:|
| Qwen3-8B（原始） | 18% | 85% |
| + Mid-train（纯文档 100%） | 43% | 45% |
| + Mid-train（70% 文档） | 36% | 79% |
| **+ Mid-train + OPD 蒸馏** | **41%** | **83%** |

**核心效果**：OPD 几乎完全恢复了指令遵循能力（85%→83%），且不损失知识（甚至略有提升 36%→41%）。

### OPD 作为持续学习工具

这一范式可推广：
- **交替训练**：fine-tune 新知识 → OPD 恢复行为 → fine-tune 新知识 → ...
- 用模型自身作为 reward model（高概率行为 = 被奖励行为）
- 任何经过 instruction-tuned 的开源模型都可通过 `compute_logprobs` 作为 reward model
- 与 Inverse RL 的联系：高概率行为对应底层偏好模型中的有利 reward

---

## 七、核心洞察

### 1. 密集监督极大提升算力效率

**Head-to-head 对比实验**：
- Qwen3-8B-Base → RL 训练（DeepMath, LoRA rank 128）→ 得到教师
- 从 RL 训练的模型蒸馏回 Qwen3-8B-Base
- **结果：OPD 仅需 7-10× 更少梯度步数达到教师性能，总计算效率 50-100×**

| 维度 | RL | OPD |
|------|-----|-----|
| 信息量 | O(1) bits/episode | O(N) bits/episode（N=token 数） |
| 上下文长度 | 需训练在评估长度（避免格式惩罚） | 可用更短上下文训练 |
| Batch size | 需要大 batch | 小 batch 即可（更多 bits/episode 减少梯度噪声） |

### 2. 数据效率：同一 prompt 可多次训练

RL 在同一 prompt 上多 epoch 训练 → 记忆最终答案。OPD 学习的是教师的完整分布 → 可重复采样同一 prompt。

**极端实验**：仅用 1 个 prompt，20 个 consecutive step，256 samples/step（共 5120 条被评分的序列）→ 几乎匹配教师性能。

### 3. RL 的本质是搜索，不是学习

关键理解框架：

| | Pre-training | RL | OPD |
|------|------------|----|-----|
| 探索空间 | 参数空间 | 语义策略空间 | — |
| 计算花在哪里 | 梯度更新 | 搜索（rollout + credit assignment） | 模仿最终策略 |

- RL 的每一步 = 对已有策略做微小变异 + 随机采样 → "碰运气"发现新策略
- **一旦 RL 发现了好的最终策略，OPD 可以直接学会它**，无需经历中间 curriculum
- 类比：科研探索 vs 教学——发现答案很慢，一旦发现，教给别人很快

### 4. On-Policy Learning 防止 SFT 的退化

**关键实验**：用 Qwen3-32B 自己的采样（KL=0 by definition）做 SFT → 仍然退化！

原因：虽然期望 KL=0，但每个 finite batch 分布略有不同 → 非零梯度更新 → 更新后的模型策略偏离原始状态 → "训练自己样本"逐渐变成 off-policy → 同样的复合误差问题。

OPD 始终 on-policy（学生采样），教师固定 → 学生收敛到教师，不会像 SFT 那样退化。

---

## 八、总结

> **OPD = On-Policy 的相关性 + 密集奖励的成本效率**

| 对比维度 | Off-Policy 蒸馏 (SFT) | RL | OPD |
|------|:---:|:---:|:---:|
| 采样来源 | 教师 | 学生 | 学生 |
| 奖励密度 | 密集 | 稀疏 | 密集 |
| 学习效率 (bits/episode) | O(N) | O(1) | O(N) |
| Exposure bias | 严重 | 无 | 无 |
| 持续学习能力 | 退化 | 仅塑形，不教知识 | 可恢复行为 + 保持知识 |
| 数据复用 | 有限 | 记忆答案 | 可多 epoch |
| 算力成本 | 高（需教师生成大量数据） | 极高（17,920 GPU h） | 低（1,800 GPU h） |

**OPD 为什么现在可行且重要**：
1. 小模型 + 强 post-training 在许多领域超越大模型
2. 开源 instruction-tuned 模型可作为现成的"reward model"（只需 `compute_logprobs`）
3. 对持续学习、个性化部署等场景，OPD 是目前最经济有效的选择

---

## 关键参考文献

- Agarwal et al., "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes" (ICLR 2024) — OPD 开创性工作
- Gu et al., "MiniLLM: Knowledge Distillation of Large Language Models" (ICLR 2024) — Reverse KL for LLM KD
- Qwen3 Technical Report (2025) — OPD 74.4% AIME'24 at 1/10 cost of RL
- Ross et al., "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (DAGGER, 2010) — OPD 的 inspiration
- Lightman et al., "Let's Verify Step by Step" (2023) — Process reward modeling
- Rafailov et al., "Direct Preference Optimization" (2023) — 模型自身作为 reward model
- Shenfeld et al., "RL's Razor: Why Online Reinforcement Learning Forgets Less" (2025)
