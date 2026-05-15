# On-Policy Distillation (OPD)：起源、发展路线与当今现状

> 综合来源：Thinking Machines Lab Blog (2025.10) · 知乎 OPD 综述 (2026.01) · Qwen3/3.5 · MiMo-V2 · GLM-5 · DeepSeek-V4 技术报告 · 21 篇学术论文

---

## 零、Preliminary：Forward KL 与 Reverse KL——OPD 的理论根基

> 本节综合自 Emiliano Penaloza et al., "Privileged Information Distillation for Language Models" (arXiv:2602.04942, 2026) 及其配套博客。该工作提出了 π-Distill 方法，同时也是理解 OPD 中 Forward KL vs Reverse KL 的最佳入门材料。

### 为什么 KL 方向是 OPD 一切讨论的起点

OPD 的目标是让学生模型的输出分布逼近教师。但"逼近"有两种截然不同的方式，取决于 KL 散度的方向：

```
Forward KL:  KL(教师 || 学生)  →  Mode-covering（覆盖模式）
Reverse KL:  KL(学生 || 教师)  →  Mode-seeking （寻找模式）
```

**这个选择决定了学生学到什么、丢掉什么、付出什么代价**。OPD 用 Reverse KL 不是随意的——它是从 RL-as-inference 框架中数学推导出来的必然结果。

---

### RL as Variational Inference：Reverse KL 从哪里来

RL 的标准目标是最大化期望奖励：

```
π* = arg max_π  E_{y~π(·|x)} [R(y, x)]
```

但在实践中，直接用这个目标会导致策略熵崩塌（entropy collapse）——模型把所有概率质量堆到一个高奖励输出上，失去泛化能力。

RL-as-inference 换个视角看这个问题。定义**目标分布** π* 为一个 reward-tilted 分布：

```
π*(y|x) ∝ π_ref(y|x) · exp(R(y,x) / β)
```

其中 β 是 KL 正则化系数，π_ref 是初始参考策略（通常是 base model 或 EMA）。

> **直观理解**：π* 是一个"理想分布"——它在 π_ref 的基础上，给高奖励的输出加权，但不偏离参考策略太远。β 越小，奖励的影响越大；β 越大，越接近参考策略。

问题是 π* 的归一化常数 Z(x) 无法计算（需要对所有可能的输出求和）。所以我们**参数化一个近似策略 π_θ 去逼近 π***。关键在于：**我们只能从 π_θ 采样，不能从 π* 采样**——这决定了只能用 Reverse KL：

```
Reverse KL:  KL(π_θ || π*)  =  E_{y~π_θ} [log π_θ(y|x) - log π*(y|x)]
                              ↑
                              期望在 π_θ（可采样）上取
```

展开并去除常数项，得到**标准 KL 约束的 RL 目标**：

```
max_θ  E_{y~π_θ}[R(y, x)]  -  β · KL(π_θ || π_ref)
       ↑ 最大化奖励             ↑ 不偏离参考策略太远
```

**这里有一个对理解 OPD 至关重要的点**：RL 的 KL penalty 项 `KL(π_θ || π_ref)` 本质上是 Reverse KL——它把 π_ref 当作"目标"（类似于 OPD 中的教师）。当 β > 0 时，策略在奖励最大化和"不远离参考"之间平衡。

---

### Forward KL = SFT（Off-Policy 蒸馏）

**定义**：

```
Forward KL:  KL(π_T || π_S)  =  E_{y~π_T} [log π_T(y|x) - log π_S(y|x)]
                                  ↑
                                  期望在教师分布 π_T 上取
```

因为 π_T 固定，最小化 Forward KL 等价于最大化教师样本上的对数似然：

```
J_SFT(θ) = max_θ  E_{y~π_T} [log π_S(y|x)]
```

这就是标准 SFT（Supervised Fine-Tuning）。

**Forward KL 的行为——Mode-Covering（覆盖模式）**：

Forward KL 惩罚学生对教师任何 mode 分配过低概率的行为。

> 教师有 A（60%）、B（30%）、C（10%）三个 mode。
> Forward KL 逼迫学生覆盖 A、B、C 全部三个 mode，即使其中某些对最终任务无用。
> → 学生分布"摊大饼"，在高维 token 空间中扩散到教师的整个支撑集。
> → 优点：多样性高。缺点：学到教师的低质量/噪声输出。

**核心风险**：学生输出教师不常输出的 token——在 LLM 生成场景中，这意味着学生可能生成教师几乎不会生成的序列，导致事实性错误（hallucination）。

**为什么 OPD 不用 Forward KL**：Forward KL 的期望在**教师分布**上取 → 训练数据必须来自教师 → **Off-Policy** → 学生推理时走的是自己的分布，不是教师的 → distribution mismatch → compounding error。

---

### Reverse KL = Self-Distillation / OPD（On-Policy 蒸馏）

**定义**：

```
Reverse KL:  KL(π_S || π_T)  =  E_{y~π_S} [log π_S(y|x) - log π_T(y|x)]
                                  ↑
                                  期望在学生分布 π_S 上取
```

**Reverse KL 的行为——Mode-Seeking（寻找模式）**：

Reverse KL 只需要学生抓住教师最确定的那个 mode。

> 教师有 A（60%）、B（30%）、C（10%）三个 mode。
> Reverse KL 允许学生把所有概率质量堆到 A 上——对 B 和 C 只给出微量概率。
> 惩罚 = 学生概率 × log(学生/教师)：学生对 C 给低概率 → 乘以前面的低概率 → 惩罚近乎零。
> → 学生"挑选"教师最容易拟合的 mode 进行精确拟合。
> → 优点：精确、稳定、不学噪声。缺点：可能挑到低奖励的 mode（非全局最优）。

**为什么 OPD 必须用 Reverse KL**：

1. **采样约束**：OPD 的期望必须在**学生分布**上取（学生自己生成 rollout）——只有 Reverse KL 能做到
2. **On-Policy**：学生在自己实际犯错的位置被纠正 → 无 distribution shift
3. **Mode-Seeking 是 feature 不是 bug**（在特定场景下）：教师的低概率 mode 往往是噪声/错误 → Reverse KL 天然过滤
4. **与 RL 统一**：Reverse KL 让 OPD 可以直接嵌入 RL 训练管线（同一个 loss 形式）

---

### Forward KL vs Reverse KL：直观对比

| 维度 | Forward KL（Off-Policy SFT） | Reverse KL（On-Policy OPD） |
|------|------------------------------|------------------------------|
| **公式** | KL(教师 ‖ 学生) | KL(学生 ‖ 教师) |
| **采样来源** | 教师分布（固定数据集） | 学生分布（on-policy rollout） |
| **行为** | Mode-covering（覆盖所有 mode） | Mode-seeking（追求单一 mode） |
| **学生分布** | 宽而模糊，覆盖教师所有可能性 | 窄而尖锐，只学教师最确定的部分 |
| **对教师低概率 region** | 强制覆盖 → 学到噪声 | 自然忽略 → 过滤噪声 |
| **对高概率 region** | 可能欠拟合（分布太分散） | 精确拟合 |
| **多样性** | 高（但可能包含低质量输出） | 低（但输出质量更可控） |
| **OOD 风险** | 高（学生可能在教师不去的区域犯错） | 低（学生始终在自身分布上） |
| **训练稳定性** | 高（静态数据集） | 中等（需 on-policy 采样） |
| **适用场景** | 初始化阶段（SFT cold start） | 精细化阶段（OPD/R L） |

**可视化直觉**（来自 Penaloza et al. 博客的简化 reward space 可视化）：

```
Forward KL (SFT):
  教师分布:  ████░░░  ░░███░  ░░░░██
             mode A   mode B   mode C
  学生分布:  ██░░░░░  ░░██░░  ░░░░█░
             ↑ 覆盖全部 mode，但每个都不精确 ↑

Reverse KL (OPD):
  教师分布:  ████░░░  ░░███░  ░░░░██
             mode A   mode B   mode C
  学生分布:  █████░░  ░░░░░░  ░░░░░░
             ↑ 只追求 mode A，但精确拟合 ↑
```

---

### 从 Reverse KL 到 Reward-Tilted Distillation

纯 Reverse KL 自蒸馏有一个核心问题：学生倾向于拟合教师**最容易**的那个 mode，但这个 mode 可能对应**低奖励**（即教师也经常犯错的那个输出模式）。

解决方法——**Reward-Tilted Reverse KL**（Penaloza et al., 2026；Shenfeld et al., 2026）：

```
π*_tilted(y|x) ∝ π_T(y|x) · exp(R(y,x) / β)
                         ↑
                 在教师分布上乘以奖励的指数项
```

目标变为：

```
max_θ  E_{y~π_θ}[R(y,x)]  -  β · KL(π_θ || π_T)
       ↑ 往高奖励区域推          ↑ 但不偏离教师
```

这就是 π-Distill 和 OPD+ORM（Outcome Reward Model）组合的理论基础——在 MiMo-V2 的 MOPD 和 DeepSeek-V4 的 GRM 中都有体现。

**与标准 RL 的关键区别**：
- 标准 RL：KL penalty 的参考是**静态**的 π_ref（base model）
- Reward-Tilted OPD：KL penalty 的参考是**动态**的 π_T（teacher model，可以随着训练改善）

---

### 为什么这很重要

所有 OPD 变体——从最基础的 Agarwal et al. (2023) 到最复杂的 MiMo-V2 MOPD 和 DeepSeek-V4 多专家 OPD——都是在 Forward KL 和 Reverse KL 之间的光谱上做选择：

```
纯 Forward KL  ←————————————→  纯 Reverse KL
    (SFT)                          (Self-Distillation)
      │                                  │
      ├─ Off-policy ────────────────── On-policy
      ├─ Mode-covering ─────────────── Mode-seeking
      ├─ 覆盖教师全部模式 ──────────── 精确拟合教师最强模式
      ├─ 学到噪声 ──────────────────── 可能错过高奖励 mode
      └─ 初始化的好选择 ────────────── 精细化的好选择
```

**实践中**（Qwen3, MiMo-V2, GLM-5, DeepSeek-V4 的共同模式）：

1. Phase 1：Forward KL（SFT on teacher data）→ 初始化，扩展学生支撑集
2. Phase 2：Reverse KL（OPD on student rollouts）→ 精细化，mode-seeking 到最优解
3. Phase 3（可选）：Reward-Tilted Reverse KL → OPD + Outcome Reward

---

## 一、雏形：OPD 的思想源头

### 1.1 DAGGER（2010-2011）——交互式模仿学习

**论文**: A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning
**作者**: Stéphane Ross, Geoffrey J. Gordon, Drew Bagnell (CMU)
**时间**: AISTATS 2011（arXiv 2010）

**核心问题**：行为克隆（Behavioral Cloning）中，学生策略在专家示范数据上训练 → 部署时遇到训练分布外的状态 → 一个错误导致连锁崩塌（compounding error）。

**解决方案**——DAGGER（Dataset Aggregation）：

```
1. 学生策略 rollout，收集自己生成的状态序列
2. 专家对学生的每一步状态标注正确动作
3. 将标注数据加入训练集
4. 迭代 1-3
```

**关键洞察**：学生必须在**自己诱导的状态分布**上被纠正，而非在专家的状态分布上。这正是 13 年后 OPD 的核心逻辑。

### 1.2 GKD / On-Policy Distillation of Language Models（2023-2024）——移植到 LLM

**论文**: On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
**作者**: Rishabh Agarwal, Nino Vieillard 等 (Google DeepMind)
**时间**: arXiv 2023.06 → ICLR 2024
**链接**: https://arxiv.org/abs/2306.13649

**核心贡献**——GKD（Generalized Knowledge Distillation）：

将 DAGGER 的 on-policy 范式从 imitation learning 移植到 LLM 知识蒸馏：

| 概念 | DAGGER (2011) | GKD / OPD (2023) |
|------|---------------|-------------------|
| 领域 | Imitation learning（机器人/RL） | Knowledge distillation（LLM） |
| 学生 | Learner policy | 小模型（student） |
| 教师 | Expert demonstrator | 大模型（teacher） |
| On-policy 数据 | Learner rollout states | Student 自生成的 token 序列 |
| 反馈 | Expert 标注正确动作 | Teacher 提供 logit probabilities |
| 解决的问题 | Distributional shift | Train-inference distribution mismatch |

**三个关键创新**：
1. **On-policy 采样**：学生自己生成序列，而非学习教师生成的数据集
2. **灵活散度**：支持 reverse KL、Jensen-Shannon 等多种散度（标准 KD 仅用 forward KL）
3. **RL 兼容**：GKD 可自然嵌入 RLHF 训练管线

**与标准 Off-Policy KD 的本质区别**：

```
Off-Policy KD（传统）:
  教师生成数据 → 固定数据集 → 学生模仿
  问题：学生推理时遇到的状态 不在数据集中 → compounding error

OPD / GKD:
  学生自己生成 → 教师对学生每一步打分 → 学生被纠正
  优势：学生始终在自己的分布上被评估 → 无 distribution shift
```

### 1.3 MiniLLM（2023）——Reverse KL 的引入

**论文**: MiniLLM: Knowledge Distillation of Large Language Models
**作者**: Yuxian Gu 等 (清华)
**时间**: arXiv 2023.06 → ICLR 2024

从另一个角度论证了 on-policy + reverse KL 的必要性：

- Forward KL（标准 KD）：mode-covering——学生必须覆盖教师所有输出模式，包括噪声和教师不确定的低概率区域
- Reverse KL（MiniLLM/OPD）：mode-seeking——学生只需学到教师最确定的推理路径，避免被教师的噪声分布污染

**OPD 的 loss 函数由此定型**：
```
Reverse KL(学生 || 教师) = Σ 学生(token) × log(学生(token) / 教师(token))
在学生自己生成的前缀上计算
```

---

## 二、发展路线：从学术论文到工业范式

### 时间线总览

```
2010-2011  DAGGER (Ross et al.)
            └─ 交互式模仿学习的思想奠基

2023.06    Agarwal et al. / Gu et al. 分别投稿 arXiv
            └─ OPD 与 MiniLLM 几乎同时提出

2024.05    ICLR 2024 正式发表
            └─ OPD 作为独立研究方向确立

2024.06    Google Gemma 2 技术报告
            └─ 首个在 production 模型中使用 OPD（9B 模型从更大教师蒸馏）

2025.05    Qwen3 技术报告 ← 分水岭
            └─ 首个系统性阐述 OPD 的头部实验室报告
            └─ Strong-to-Weak 两阶段蒸馏：SFT → OPD
            └─ 仅需 RL 的 1/10 GPU 时间达到可比推理能力

2025.10    Thinking Machines Lab Blog
            └─ 最全面的 OPD 工程实践指南
            └─ 数学推理 9-30× 算力节省 + 个性化/持续学习

2025.11    DeepSeek-R1-0528 系列发布
            └─ OPD 用于大规模推理模型蒸馏

2025.12    MiMo-V2-Flash (小米) 开源 + GLM-5 (智谱) 发布
            └─ MOPD：多教师 OPD 首次工业化
            └─ Cross-Stage OPD：OPD 用于防止多阶段 RL 遗忘

2026.01    Qwen3.5 系列分波发布 (0.8B~397B)
            └─ Gated Delta Networks + OPD 后训练

2026.02    学术论文爆发期开始
            └─ DeepMind 内在维度度量化 CoT → OPD 质量评估
            └─ CODI / Coconut 在 latent space reasoning 取得进展

2026.03    论文 2 (Revisiting OPD: Failure Modes) 上线
            └─ 系统化 OPD 三类失败模式 + 修复方案

2026.04    里程碑月
            ├─ DeepSeek-V4：OPD 完全替代 mixed RL，10+ 专家模型蒸馏为一
            ├─ 论文 5 (Tsinghua)：OPD 机制深度分析，分布不可区分性，97-99% 概率质量
            ├─ MAD-OPD：多智能体辩论突破单教师天花板
            ├─ Lightning OPD：完全离线 OPD，速度 4×
            ├─ StableOPD / SCOPE：训练稳定性与采样策略改进
            └─ Qwen3.5-Omni：OPD 第二阶段训练用于全模态模型

2026.05    MAD-OPD (arXiv:2605.01347) + 更多跟进研究
```

### 关键转折点详解

#### 转折点 1：Qwen3（2025.05）——OPD 工业化的起点

Qwen3 是最早将 OPD 系统化为后训练核心组件的头部实验室报告：

| 阶段 | 内容 |
|------|------|
| Off-Policy SFT | 教师（235B-A22B/32B）在大规模数据集上生成混合思考/非思考输出，学生（0.6B~30B）模仿 |
| On-Policy Distillation | 学生自主生成 rollout，教师对每个 token 给出 logit 反馈，最小化 reverse KL |

**关键数据**：OPD 仅需完整 RL pipeline 的 **1/10 GPU 时间**，达到可比推理性能。AIME'24 74.4% @ 1,800 GPU hours vs RL 67.6% @ 17,920 GPU hours。

#### 转折点 2：MiMo-V2（2025.12）——多教师 OPD

小米将单教师 OPD 扩展为 **MOPD（Multi-Teacher On-Policy Distillation）**：

```
Stage 1: SFT → 基础指令遵循
Stage 2: 独立训练 5 个专家教师（Code Agent / Search Agent / Math / General Reasoning / Safety）
Stage 3: MOPD → 学生从自身分布采样，5 个领域教师分别提供 token 级监督
         Advantage = Σᵢ wᵢ × KL(π_domainᵢ || π_student) + α × ORM
```

**核心发现**：在 logit 空间而非参数空间中整合能力——避开了 weight merge 中常见的能力干扰。学生甚至**超越最强单教师**：AIME 2025 94.1% vs teacher 93.9%。

#### 转折点 3：GLM-5（2025.12-2026.01）——OPD 防遗忘

GLM-5 面临独特问题：多阶段 RL（SFT → Reasoning RL → Agentic RL → General RL）中，最后的 General RL 会冲刷前面的推理和代码能力。

**Cross-Stage OPD** 的解决方案：

```
取前序阶段 checkpoint（Reasoning RL 结束 + General RL 结束）作为教师
→ 当前模型作为学生
→ OPD 将前序能力"蒸馏回"当前模型
→ 不需要重新训练，不需要 weight merge
```

工程细节：
- GRPO group size 从 32 降至 **1**（教师 log ratio 直接充当 advantage，无需组内对比）
- Batch size 扩大至 **1024**
- 纯 KL 信号，不叠加 outcome reward——目标纯粹是"缝合"回各阶段能力

#### 转折点 4：DeepSeek-V4（2026.04）——OPD 替代 RL

DeepSeek-V4 做出了最激进的选择：**完全用 OPD 替代 V3.2 中的混合 RL 阶段**。

| 维度 | V3.2 | V4 |
|------|------|-----|
| 整合方法 | Mixed RL (hybrid GRPO) | OPD |
| 专家数量 | ~5 个 | 10+ 个 |
| 蒸馏方式 | Off-policy（教师生成数据→学生 SFT） | On-policy（学生 rollout→教师反馈） |
| KL 方向 | Forward KL | Reverse KL |

DeepSeek-V4 的论证：
1. RL 的 scalar reward 方差高；OPD 的 full logit distribution 方差低
2. Teacher logits 已代表 Pareto-optimal 的能力平衡——无需在多目标间做 trade-off
3. Reverse KL 天然替代 KL penalty，不需要 value network 和 GAE
4. Mode-seeking 保留学生原有能力，防止灾难性遗忘

---

## 三、当今发展现状

### 3.1 OPD 已成为后训练新范式

来自知乎综述（2026.01）的判断：

> "On-Policy Distillation 正在成为事实上的后训练新范式。从 Qwen3 到 GLM-5，从 MiMo-V2 到 DeepSeek-V4，各家技术报告都指向同一个趋势：后训练不再只依赖昂贵的 RL 探索，而是越来越重视如何稳定、高效地把已有强模型的能力迁移到目标模型中。"

**OPD 相对于混合 RL 的核心优势**：

| 维度 | Mixed RL | OPD |
|------|----------|-----|
| 能力整合空间 | 参数空间（weight merge）→ 能力干扰 | Logit 空间（KL alignment）→ 无干扰 |
| 信号密度 | O(1) bits/episode | O(N) bits/episode（50-100×） |
| 信号类型 | Scalar reward（高方差） | Full distribution（低方差，"全息梯度"） |
| 多目标冲突 | 需要 tuning KL penalty/Reward weights | Teacher logits 本身是 Pareto-optimal |
| 灾难性遗忘 | 需要额外机制 | Reverse KL mode-seeking 天然保留已有能力 |
| GPU 成本 | 17,920 GPU h（Qwen3 RL） | 1,800 GPU h（Qwen3 OPD） |

### 3.2 当前主流 OPD 变体

| 变体 | 提出者 | 核心思路 | 适用场景 |
|------|--------|---------|---------|
| **Standard OPD** | Agarwal et al. (ICLR 2024) | 单教师 → 单学生，token 级 reverse KL | 通用能力迁移 |
| **Strong-to-Weak OPD** | Qwen3 (2025) | 两阶段：SFT 初始化 + OPD fine-tune | 大模型蒸馏小模型 |
| **Multi-Teacher OPD** (MOPD) | MiMo-V2 (2025.12) | 多领域专家教师，logit 空间加权融合 | 整合多个专项能力 |
| **Cross-Stage OPD** | GLM-5 (2025.12) | 前序 checkpoint 作教师，蒸馏回当前模型 | 多阶段 RL 抗遗忘 |
| **Multi-Expert OPD** | DeepSeek-V4 (2026.04) | 10+ 专家模型 + Generative Reward Model | 全面替代 mixed RL |
| **MAD-OPD** | arXiv:2605.01347 (2026.05) | 多教师辩论驱动的 OPD，突破单教师能力天花板 | Agentic 任务 |
| **Lightning OPD** | arXiv (2026.04) | 完全离线 OPD，预计算教师 logits | 追求训练速度 |
| **StableOPD** | arXiv:2604.08527 (2026.04) | 解决 OPD 训练中的重复崩溃 | 训练稳定性 |
| **SCOPE** | arXiv:2604.10688 (2026.04) | 答对/答错轨迹差异化加权 | 样本效率 |

### 3.3 学术界的核心发现（2026 年集中爆发）

| 发现 | 来源 | 核心结论 |
|------|------|---------|
| **OPD 三���失败模式** | 论文2 (2026.03) | 不平衡单 token 信号、教师引导在 unknown prefix 不可靠、tokenizer 不匹配 |
| **分布不可区分性** | 论文5 (2026.04) | 同家族 7B 和 1.5B 在学生视角下分布不可区分——更大≠更好教师 |
| **97-99% 概率质量** | 论文5 (2026.04) | OPD 梯度信号源集中在 ~3% 的共享 token——token 空间对齐天然受限 |
| **Thinking-Pattern 必须兼容** | 论文5 (2026.04) | OPD 成功的首要条件不是 benchmark 分数高，而是思维模式 overlap |
| **长序列崩塌** | 论文5 (2026.04) | 教师准确率优势从 +0.37 (1K) 退化到 +0.02 (16K)；崩塌从后往前传播 |
| **Reverse KL Mode Collapse** | 论文7,8 (2026.03) | 学生只学教师最确定的那条路径，失去推理多样性 |
| **β=1 限制了 OPD 表现力** | 论文12 (2026.02) | OPD 是 β=1 的 KL-constrained RL 特例——奖励和 KL 必须等权重 |
| **学生可以超越教师** | 论文12 (2026.02) | ExOPD (λ>1) 通过 reward extrapolation 突破教师能力边界——唯一方案 |
| **参考模型选择关键** | 论文12 (2026.02) | Reward Correction: 用教师 pre-RL base 作为 π_ref 提供更准确奖励信号 |
| **Small Model Learnability Gap** | 论文3 (ACL 2025) | ≤3B 模型从长 CoT 蒸馏反而退步 |
| **Distribution Shift 伤害 CoT** | 论文6 (2025.06) | 首次理论证明：分布偏移下 CoT 比不推理更差 |
| **内在维度度量化 CoT** | DeepMind (2026.02) | LoRA 微调所需最小参数量与 OOD 泛化相关性 0.93，远超长度 |

### 3.3.1 论文5 详解：OPD 的机制、条件与边界

> **论文**: Li et al. (Tsinghua NLP), "Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe"
> **链接**: https://arxiv.org/abs/2604.13016 | **代码**: https://github.com/thunlp/OPD
> **规模**: 30 页，23 张图，2026 年 4 月

这是迄今为止对 OPD 训练动力学最系统的实证研究。以下是其五个核心发现的详细展开。

---

#### 发现一：OPD 成功的两个必要条件

**条件 1：思维模式兼容（Thinking-Pattern Consistency）**

学生和教师必须共享兼容的"思维模式"——由 top-k token 分布的 **Overlap Ratio** 衡量：

```
Overlap Ratio = |S_t^p ∩ S_t^q| / k
其中 S_t^p = 学生在 prefix 下的 top-k token 集合
     S_t^q = 教师在 prefix 下的 top-k token 集合
```

实验证据：Qwen3-1.7B-Base 学生 × 两个 Qwen3-4B 教师：

| 教师类型 | Benchmark 分数 | Thinking Pattern | OPD 结果 |
|---------|:---:|------|:---:|
| GRPO 训练 | 相近 | 与学生兼容 | ✅ 成功 |
| Non-thinking | 相近 | 与学生差异大 | ❌ 失败 |

**核心洞察**：benchmark 分数高 ≠ 好老师。overlap ratio 才是预测 OPD 成败的关键指标。

**条件 2：教师必须有真正的新能力**

即使思维模式一致且 benchmark 分数更高，教师必须提供学生在训练中**未见过**的能力。在两个模型家族（DeepSeek + Qwen）上验证：

| 教师类型 | 效果 |
|---------|------|
| Same-pipeline 教师（仅规模放大，相同数据/方法训练） | 改善有限——大模型只是"对相同数据拟合得更好" |
| Post-trained 教师（经过 RL 获得新能力） | 大幅提升，"gap recovery rate" 显著更高 |

**核心洞察**：更高 benchmark 分数可能只反映"对相同数据的拟合程度不同，而非真正的新能力"。

---

#### 发现二：逆蒸馏实验——分布不可区分性（最令人警醒的发现）

**实验设计**：用 **JustRL-1.5B**（经 RL 训练的强大 1.5B）作为学生，逆向蒸馏到更弱的教师。

**结果 1**：蒸馏到 **R1-Distill-1.5B**（学生自己 RL 前的 checkpoint）
→ 学生退化到**几乎精确等于 RL 前的性能**——"移除了 RL 获得的所有增益"

**结果 2**：蒸馏到 **R1-Distill-7B**（同家族更大的模型，benchmark 分数更高）
→ 训练轨迹与蒸馏到 1.5B 教师**几乎完全相同**

**结论**：R1-Distill-1.5B 和 R1-Distill-7B 在学生视角下是**分布不可区分的**（distributionally indistinguishable）。因为 OPD 的 reverse KL 发生在 student-visited states 上——两个教师在这些状态上诱导出几乎相同的局部目标分布。

**三层含义**：
1. OPD 本质上学的是**思维模式**——会主动获取教师模式并覆盖学生自己的
2. **Benchmark 性能不能预测 OPD 结果**——7B 分数更高但蒸馏无增益
3. 更高 benchmark 分数可能只是"对相同数据的过拟合"，不是真正的能力差异

---

#### 发现三：Token 级机制——97-99% 概率质量的秘密

对比成功 OPD vs 失败 OPD 的动态指标：

**成功 OPD（R1-Distill-1.5B ← JustRL-1.5B）**：
- Overlap Ratio 从 **~72% → ~91%**（持续上升）
- Entropy Gap 持续缩小
- Overlap-Token Advantage 趋向零

**失败 OPD（R1-Distill-1.5B ← R1-Distill-7B）**：
- Overlap Ratio **停滞**（从一开始就低）
- Entropy Mismatch 从始至终存在

**最关键的发现**：交集 token 承载了师生双方 **97-99% 的总概率质量**（整个训练过程中保持）。

这意味着 OPD 的梯度信号几乎完全来自 ~3% 的共享高频 token。其余 97%+ 的 token 不参与学习。

**因果验证（Ablation）**：
- **Overlap Top-k**（仅优化交集 token）→ 匹配完整 Student Top-k 的性能
- **Non-Overlap Top-k**（优化对称差集 token）→ 显著更弱

**自增强动力学**（解释为什么低初始 overlap 的训练会陷入停滞）：

```
某个 token 进入共享高概率区域
  → 被教师偏好
    → Reverse KL 将更多概率质量集中到该 token
      → Overlap Ratio 上升
        → 更多 token 进入共享区域
          → 循环加速
```

反向同理：低初始 overlap → 梯度信号弱 → overlap 无法提升 → 训练停滞。

---

#### 发现四：两种恢复策略

**策略 1：Off-Policy Cold Start**

```
Phase 1: SFT on teacher-generated rollouts (off-policy)
  → 提升学生初始 overlap ratio
  → 学生从"教师分布附近"开始

Phase 2: 切换到标准 OPD (on-policy)
  → 更高的起点 → 训练轨迹更平滑
  → 最终上限也更高（不止影响早期）
```

**策略 2：Teacher-Aligned Prompt Selection**

| 粒度 | 做法 | 效果 |
|------|------|------|
| Template 对齐 | 改用教师后训练的 prompt 模板（如 `\boxed{}` 替代 "Answer:"） | 三项 benchmark 提升 + overlap ratio 提升 |
| Content 对齐 | 使用教师后训练数据中的 prompt | 效果更强 |

**风险警告**：Teacher-aligned prompts 会降低学生训练时的 entropy——建议混合分布外 prompt 维持探索能力。

---

#### 发现五：OPD "免费午餐"的代价——长序列不可扩展

**核心实验**：在 0.5K 到 15K 六种最大生成长度下训练 OPD。

**发现**：
- 中等长度（3K, 7K）是最优区间
- 超过（10K, 15K）→ **性能平台化或下降**
- "后期崩塌"：Overlap Ratio 急剧下降，学生 Entropy 和梯度范数飙升
- 崩塌从**后往前传播**：高 entropy 首先出现在响应末尾 → 逐步向前扩散

**教师延续能力随前缀深度单调退化**：

| 前缀长度 | 教师准确率优势 |
|---------|:---:|
| 1K | **+0.37** |
| 4K | +0.21 |
| 8K | +0.11 |
| 16K | **+0.02**（几乎消失） |

**"全局信息量 ≠ 局部可利用性"**（最关键的理论洞察）：

- 成功和失败的教师都产生与 rollout 正确性相关的 token 级奖励（AUROC 0.73 vs 0.75，接近）
- 但失败的 7B 教师的奖励景观在学生策略周围是**局部平坦**的 → token 级梯度无效
- 假设：大教师的 per-token advantage 在序列内**各向异性**（anisotropic），梯度聚合时相互抵消

**结论**：OPD 在长序列场景下可能无法干净扩展。这对 extended CoT、agentic multi-turn 等场景提出了根本性挑战——**token 级 OPD 在长链上不是效率问题，而是可能根本不 work**。

---

#### 该论文对 OPD 研究方向的影响

这五个发现共同指向一个方向：**token 空间本身是 OPD 的瓶颈**。

```
Token 级 OPD 的三个天花板：
  1. 只有 ~3% 的 token 参与学习（97-99% 概率质量发现）
  2. 教师在长序列上不可靠（+0.37 → +0.02）
  3. 更大模型不一定是更好的教师（分布不可区分性）

→ Latent Space OPD 的三个对应机会：
  1. 连续 latent space 没有离散 token 集 → 100% 的表示维度参与学习
  2. Latent 推理天然适合压缩长链 → 降低有效序列长度
  3. Latent 表示可能提取与模型规模无关的推理结构
```

### 3.3.2 论文7 详解：OPD 的三类失败模式与修复路径

> **论文**: Fu et al., "Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes"
> **链接**: https://arxiv.org/abs/2603.25562 | **博客**: http://yuqianfu.notion.site/revisiting-opd | **代码**: https://github.com/hhh675597/revisiting_opd
> **作者**: Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Jiacai Liu, Zhuo Jiang, Yuanheng Zhu, Dongbin Zhao
> **时间**: 2026 年 3 月

这篇论文与论文5（Tsinghua）互补：论文5 回答了"OPD 什么时候成功、为什么成功"，这篇回答了"OPD 什么时候失败、为什么失败、怎么修"。两者共同构成了理解 OPD 机制的完整图景。

---

#### 理论分析：Token 级 OPD 的 Bias-Variance Tradeoff

OPD 的标准实现是 **sampled-token OPD**——对每个采样到的 token 计算 reverse KL 的 log-ratio。但作者指出，这在理论上存在根本性的 tension：

| | Sequence-Level Reverse KL | Token-Level (Sampled) OPD |
|---|---|---|
| **偏差** | 无偏（Unbiased） | **有偏**（Biased）——将分布匹配退化为单 token 信号 |
| **方差** | 高方差（worst-case bound 宽松） | **低方差**（更紧的 worst-case bound） |
| **实践后果** | 理论上正确但不稳定 | 实践中可用但信号脆弱 |

**Future-Reward Coupling 分析**（受控合成实验）：

核心发现：**future-reward coupling 越强 → 梯度方差越高 → 学习越不稳定**。

```
含义：当前 token 的选择对远期 reward 的影响越大
  → 单 token 的 log-ratio 信号越不可靠
  → 在长推理链中，早期 token 的信号几乎被噪声淹没
  → 解释了为什么 OPD 在长序列上特别脆弱
```

---

#### 三类典型失败模式

**失败模式 1：不平衡的单 Token 信号（Imbalanced One-Token Signal）**

```
Sampled-token OPD 在每个位置只评估一个 token（实际采样到的那个）
  → 信号集中在高频 token（占概率质量 97-99%，论文5 验证）
  → 长尾但关键的低频推理 token（如数学符号、逻辑连接词）几乎收不到梯度
  → 学生学会了"看起来像在想"但没学会"真正在想"
```

**失败模式 2：教师引导在学生生成前缀上不可靠**

这是所有 OPD 变体共同面临的核心矛盾：

```
学生自行生成 rollout
  → 随着生成进行，前缀逐渐漂移到教师很少访问的区域
    → 教师在这些 OOD 前缀上的 log-prob 评估越来越不可靠
      → 教师给出"垃圾反馈"
        → 学生被垃圾反馈引导，漂移加速
          → 恶性循环 → 训练崩塌
```

**关键洞察**：这不是教师的错，也不是学生的错——**是 OPD 的结构性矛盾**。学生必须走到新区域才能进步，但教师在新区域无法提供可靠指导。

**失败模式 3：Tokenizer / 特殊 Token 不匹配失真**

```
不同模型的 tokenizer 行为不一致:
  - <think>, </think>, <|assistant|> 等特殊 token
  - 数字的分词方式（"123" vs "1" "2" "3"）
  - 代码缩进、换行符的处理

→ 教师和学生对这些 token 的概率评估天然不可比
→ 直接计算 KL 在这些位置引入虚假的梯度信号
```

---

#### 修复方案：Teacher Top-K Local Support Matching

作者提出了一套组合修复策略，核心思想是**限制对齐范围，只在对齐有意义的地方做对齐**：

**组件 1：Truncated Reverse-KL**

```
标准 Reverse KL:  在整个词表 V 上计算 KL(学生 || 教师)
Truncated Reverse KL: 只在教师支持的 token 集合上计算

做法：在每个 prefix，取教师 top-K token，重归一化后计算 KL
效果：排除教师几乎不会生成的 token → 去噪声
```

**组件 2：Top-p Rollout Sampling**

```
标准采样:  学生按照自己的完整分布采样（可能走到教师不支持的区域）
Top-p 采样: 学生仅从累积概率 ≥ p 的 nucleus 中采样

效果: 将 student rollout 约束在教师能可靠评估的区域
代价: 可能限制探索（需要合理选择 p）
```

**组件 3：Special-Token Masking**

```
识别师生 tokenizer 行为不一致的位置
  → 在 loss 计算中 mask 掉这些位置
  → 防止虚假梯度信号
```

**组合效果**：

> 在单任务数学推理和多任务 agentic+math 训练中，该方案实现了 **+19.8%** 的性能提升（vs 标准 sampled-token OPD），且优化稳定性显著改善。

---

#### 与论文5（Tsinghua）的互补关系

| 维度 | 论文7 (Fu et al.) | 论文5 (Li et al., Tsinghua) |
|------|-------------------|----------------------------|
| **核心问题** | OPD 为什么失败？怎么修？ | OPD 什么时候成功？机制是什么？ |
| **理论贡献** | Bias-Variance Tradeoff + Future-Reward Coupling | 两个成功条件 + 分布不可区分性 |
| **关键发现** | 三类失败模式（信号不平衡、教师不可靠、tokenizer 不匹配） | Token 级机制（97-99% 概率质量、overlap 自增强） |
| **修复方案** | Teacher Top-K Local Support Matching（截断 reverse KL + top-p 采样 + 特殊 token masking） | Off-Policy Cold Start + Teacher-Aligned Prompt Selection |
| **共同指向** | Token 空间本身是 OPD 瓶颈 → Latent space 可能绕过这些失败模式 | 同左——长序列崩塌 + 分布不可区分性 → 需要超越 token 的对齐方式 |

---

#### 该论文对研究方向的影响

失败模式 2（教师引导在学生前缀上不可靠）是 latent OPD 最强有力的 motivation：

```
Token 空间的矛盾:
  学生必须在自己的分布上采样（on-policy）
  → 但教师的 token 级反馈只在教师分布附近可靠
  → 学生越进步（分布变化越大），教师越不可靠
  → 这是一个内在矛盾

Latent Space 的可能出路:
  → 连续 latent 表示天然比离散 token 更平滑
  → 即使学生的 latent state 偏离教师，教师的 latent 反馈仍然有意义
  → "漂移到 OOD" 在连续空间中可能是一个渐变而非突变
```

---

### 3.3.3 论文12 详解：G-OPD —— 超越教师边界的广义 OPD 框架

> 论文：Yang et al., "Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation" (arXiv:2602.12125, 2026.02)
> 单位：中国人民大学 · Tencent LLM
> 代码：https://github.com/RUCBM/G-OPD

#### 核心贡献

这篇工作是 OPD 理论最深入的统一框架之一。它首先建立了 **OPD 与 KL-constrained RL 的等价关系**，然后推广出 G-OPD 框架，最后发现 reward extrapolation 可以让学生**超越教师的能力边界**。

---

#### 一、理论突破：OPD 是 KL-constrained RL 的特例

论文通过数学推导证明了 OPD 目标函数可以重写为 RL 形式：

```
J_OPD(θ) = min_θ E_{x~D, y~π_θ}[ D_KL(π_θ(y|x) || π*(y|x)) ]

         = max_θ E_{x~D, y~π_θ}[ log π*(y|x) / π_ref(y|x) − D_KL(π_θ(y|x) || π_ref(y|x)) ]
                                ↑                                    ↑
                          隐式奖励 r(x,y)                      KL 正则化 (β=1)
```

**关键洞察**：
- 把教师的 log-prob 与参考模型的 log-prob 的**差值**定义为 token 级**隐式奖励（implicit reward）**
- 标准 OPD 等价于 β=1 的 KL-constrained RL——即奖励和 KL 正则化**永远等权重**
- π_ref 可以是**任意模型**（不限于学生初始化）——这个自由度此前被完全忽略了

```
标准 RL:
  r_t = 0 (t=1,...,T−1), Outcome Reward (t=T)   ← 稀疏
  β 可调

OPD（等价形式）:
  r_t = log π*(y_t|x,y_<t) − log π_ref(y_t|x,y_<t)   ← 密集！
  β = 1 (固定，不可调)
```

OPD 与标准 RL 的两个本质差异：
1. **密集奖励**：每个 token 都有有效奖励，而非仅最后 token
2. **β=1 固定**：奖励项和 KL 正则项的权重永远相等——这一点限制了 OPD 的表现力

---

#### 二、G-OPD：广义化 OPD 框架

基于上述等价关系，论文引入两个新自由度，提出 **G-OPD**（Generalized OPD）：

```
J_G-OPD(θ) = max_θ E_{x~D, y~π_θ}[ λ · log π*(y|x)/π_ref(y|x) − D_KL(π_θ(y|x) || π_ref(y|x)) ]
                                      ↑                                    ↑
                                 奖励缩放因子 λ                      灵活的参考模型
```

| 参数 | 标准 OPD | G-OPD | 含义 |
|------|:------:|:-----:|------|
| λ | 1 (固定) | **可调** | 控制奖励相对 KL 的权重（对应 RL 中的 1/β） |
| π_ref | 任意（被 cancels out） | **任意（有实际影响）** | 定义隐式奖励的基线 |

**最优解形式**：

```
log π_θ(y|x) = λ · log π*(y|x) + (1−λ) · log π_ref(y|x)
             = log π*(y|x) + (λ−1) · (log π*(y|x) − log π_ref(y|x))
```

这揭示了 λ 的三种工作区间：

| λ 区间 | 名称 | 学生行为 |
|--------|------|---------|
| 0 < λ < 1 | **Reward Interpolation** | 学生行为介于 π_ref 和 π* 之间（向教师靠拢但不到达） |
| λ = 1 | **标准 OPD** | 学生直接匹配教师分布 |
| λ > 1 | **Reward Extrapolation (ExOPD)** | 学生在匹配教师的基础上，沿 (π* − π_ref) 方向**继续外推** |

---

#### 三、ExOPD（λ > 1）：超越教师边界

这是论文最惊人的发现。

**直觉**：当 λ > 1 时，G-OPD 不仅要求学生匹配教师，还要求学生**额外放大**教师相对于参考模型的优势方向：

```
学生 = 教师 + (λ−1) · (教师 − 参考)

若 λ = 1.25 → 学生 = 教师 + 0.25 × (教师 − 参考)
```

这意味着学生在教师已经很好的 token 上**更加确信**，在教师也不确定的 token 上**更加远离**——学生**放大教师的偏好**，实现"比教师更教师"。

**实验结果（Multi-Teacher Distillation）**：

Teacher: Qwen3-4B-Non-Thinking 的领域特化 RL 变体（Math / Code 两个 teacher）

| 方法 | Math Avg (4 benchmarks) | Code Avg (3 benchmarks) | 能否超越教师？ |
|------|:------:|:------:|:------:|
| SFT | 44.3 | 60.8 | ❌ |
| ExPO (weight extrapolation) | 45.0 | 62.6 | ❌ 不可控 |
| OPD | 46.4 | 60.6 | ❌ |
| **ExOPD (λ=1.25)** | **47.7** | **62.0** | ✅ **唯一超越所有 domain teacher** |

- Teacher (Math): 46.0 → ExOPD: **48.0** (+2.0，50 步)
- Teacher (Code): 61.2 → ExOPD: **62.1** (+0.9)

**对比 continued RL——"不是教师训得不够"**：

| 方法 | Steps | Avg Acc |
|------|:-----:|:------:|
| Teacher (Qwen3-4B-RL-Math) | — | 46.0 |
| Teacher + 继续 RL 100 步 | 100 | 46.9 (+0.9) |
| **ExOPD** | **50** | **48.0 (+2.0)** |

给教师额外 100 步 RL 训练只带来 +0.9，ExOPD 用一半步数带来 +2.0——增益来自 reward extrapolation 机制本身，不是"教师训得不够"。

**λ 的敏感度**：

| λ | 效果 |
|:---:|------|
| 0.25, 0.5, 0.75 | 学生行为介于 π_ref 和 π* 之间（已验证） `[verified: 论文 Fig 3]` |
| 1.0 (OPD) | 匹配教师但不超过 |
| **1.25** | **一致最优，所有实验均使用此值** |
| 1.5 | 可能不稳定——过度放大隐式奖励噪声，学生可能 reward hack |

论文建议：**固定 λ = 1.25，无需逐任务调参**。

---

#### 四、Reward Correction：强→弱蒸馏中的奖励修正

在 strong-to-weak distillation 中，π_ref 有两个选择：

| 选择 | π_ref | 隐式奖励 r = log π*/π_ref | 优劣 |
|------|-------|--------------------------|------|
| **默认** | π_student_base | log π*/π_student_base | 只需两个模型，但奖励混入师生知识差距噪声 |
| **Reward Correction** | π_teacher_base（教师的 pre-RL variant） | log π*/π_teacher_base | 奖励=教师 RL 的真正隐式奖励，更准确；但需额外模型 + 更大计算开销 |

**核心逻辑**：
- log π*/π_teacher_base = 教师通过 RL 学到的"真正增量"
- log π*/π_student_base = 教师增量 + (学生 base − 教师 base) 的交叉项 = 含噪声
- 在 strong-to-weak 设置中，学生比教师弱 → 学生 base 与教师 base 存在知识差距 → 默认奖励混入了这个差距的噪声

**Strong-to-Weak 结果（Teacher: Qwen3-30B-A3B-Instruct-2507 → Student）**：

| 学生 | SFT | OPD | ExOPD | Δ vs OPD |
|------|:---:|:---:|:-----:|:--------:|
| Qwen3-1.7B-Non-Thinking | 13.5 | 23.1 | **25.4** | +2.3 |
| Qwen3-4B-Non-Thinking | 35.1 | 42.6 | **45.3** | +2.7 |

**Reward Correction 的额外收益**（Teacher: Qwen3-4B-RL → Student: Qwen3-1.7B）：

| Setting | SFT | OPD | ExOPD | ExOPD + Reward Correction |
|---------|:---:|:---:|:-----:|:-------------------------:|
| Math (4 benchmarks avg) | 22.7 | 27.5 | 28.1 | **28.7** |
| Code (3 benchmarks avg) | 47.0 | 50.5 | 51.3 | **52.3** |

Reward correction 一致提升，但代价明确：
- 需要访问教师的 pre-RL variant（开源模型通常有，闭源模型没有）
- π_teacher_base 比 π_student_base 更大 → 计算 log-prob 成本更高

---

#### 五、ExOPD 训练动态

```
对比 OPD 的训练曲线:

  Training Reward:   ExOPD > OPD（验证 reward extrapolation 确实提供了更强学习信号）
  Response Length:   ExOPD > OPD（生成更长回复，隐式奖励可能存在长度偏差）
  Entropy:           ExOPD > OPD（更高多样性，由更长的回复自然带来）
```

---

#### 六、与其他论文的互补关系

| 维度 | 论文12 (Yang et al.) | 论文5 (Li et al., Tsinghua) | 论文7 (Fu et al.) |
|------|:---:|:---:|:---:|
| **核心问题** | 如何超越教师边界？ | OPD 什么时候成功？ | OPD 为什么失败？怎么修？ |
| **理论贡献** | OPD = β=1 RL；引入 λ 和 π_ref 两个新自由度 | 两个成功条件 + 分布不可区分性 | Bias-Variance Tradeoff + Future-Reward Coupling |
| **关键发现** | λ>1 的外推可以超越教师；π_ref 选择影响奖励质量 | 97-99% 概率质量；overlap 自增强；长序列崩塌 | 三类失败模式及 Teacher Top-K Local Support Matching 修复 |
| **与 latent OPD 的关联** | λ 与 latent compression 率可能存在统一框架——都是一种"强度调节器" | Token 空间瓶颈 → 需要 latent | Token 空间失败模式 → 需要 latent |

---

#### 七、该论文对研究方向的影响

1. **超越教师边界现在有了理论和实验依据**：ExOPD 证明了"学生超越教师"不仅可能，而且在 multi-teacher 设置中是唯一可行方案——这与论文5（Tsinghua）"更大≠更好教师"形成互补：教师不必更大，学生也未必不能超越 `[opinion]`

2. **Multi-Teacher + ExOPD 的组合**为方向三（Thinking-Pattern 自适应压缩）提供了新可能性 `[opinion]`：
   ```
   多个领域教师 ──ExOPD──→ 统一学生（超越所有教师）
                          ↓
           如果 latent compression 能消除跨域教师间的 thinking-pattern 差异
           → ExOPD 的 reward extrapolation 在 latent space 中可能更有效
   ```

3. **Reward Correction 的思路暗示了参考模型选择的关键性**：在 latent OPD 中，用教师 base model 的 latent representation 还是学生 base model 的作为参考？这个选择可能与 token 空间中的 reward correction 一样重要 `[opinion]`

4. **λ 与 latent compression 率的理论对应** `[opinion]`：
   ```
   Token 空间: λ 控制"学多少额外的东西"（extrapolation strength）
   Latent 空间: 压缩率控制"保留多少信息"（compression strength）
   → 两者都是一种"强度调节器"，可能存在统一的理论框架
   ```

5. **G-OPD 的隐式奖励公式可以直接迁移到 latent space** `[opinion]`：
   ```
   Token 空间: r_t = log π*(y_t|x,y_<t) − log π_ref(y_t|x,y_<t)
   Latent 空间: r_step = MSE(ℓ_student, ℓ_teacher) − MSE(ℓ_student, ℓ_ref)
   → 用 latent 距离替换 log-prob 差值，ExOPD 可以在 latent space 中运行
   ```


### 3.4 仍未解决的问题

| 问题 | 严重程度 | 可能的方向 |
|------|:---:|------|
| **Token 空间瓶颈**（97-99% 概率质量只在 3% token） | 高 | Latent space OPD（连续空间无 token 限制） |
| **长序列不可扩展**（>10K 崩塌，教师准确率退化到 0） | 高 | 多步 latent rollout 减少有效序列长度 |
| **更大≠更好教师**（分布不可区分） | 中高 | Latent space 可能提取规模无关的推理结构 |
| **Reverse KL mode collapse**（推理多样性丧失） | 中 | Entropy-aware adaptive KL |
| **Thinking-pattern 兼容性**（跨家族蒸馏困难） | 中 | 自适应压缩率调和 thinking-pattern 差异 |
| **Off-policy cold start 退化**（SFT 初始化后策略漂移） | 低中 | On-policy 数据混合 + iterative OPD |
| **计算成本**（仍需 on-policy 采样 + 教师推理） | 低 | Lightning OPD（离线预计算）、speculative distillation |

### 3.5 OPD 与其他技术的关系

```
Post-training 技术栈（2026 年 5 月现状）:

         Pre-training (通用能力)
              │
         Mid-training (领域知识)
              │
    ┌─────────┴─────────┐
    │   SFT (指令遵循)    │  ← 仍然必要，off-policy
    └─────────┬─────────┘
              │
    ┌─────────┴─────────┐
    │  Reasoning RL      │  ← 探索新策略（搜索语义空间）
    │  (GRPO / DeepMath) │
    └─────────┬─────────┘
              │
    ┌─────────┴─────────┐
    │  OPD ← 这一层      │  ← 整合能力、防遗忘、廉价放大
    │  (替代 mixed RL)   │     ├─ Multi-Teacher → 多专家融合
    │                   │     ├─ Cross-Stage    → 多阶段能力缝合
    │                   │     └─ Standard       → 通用蒸馏
    └─────────┬─────────┘
              │
    ┌─────────┴─────────┐
    │  General RL        │  ← 对齐人类偏好（可能仍是 RLHF/DPO）
    └───────────────────┘
```

**OPD 与 RL 的分工**：
- **RL** = 搜索（在已有策略空间中随机变异，"碰运气"发现新策略）
- **OPD** = 教学（一旦好策略被发现，廉价高效地教会其他模型）
- RL 烧钱找答案，OPD 省钱教答案——这就是为什么 DeepSeek-V4 说 OPD 可以替代 mixed RL

**OPD 与 Latent Space Reasoning 的结合机会**：
- Latent OPD 直接解决 token 空间的 97-99% 瓶颈
- 多步 latent rollout 可能解决长序列崩塌
- Latent CKA 可能弥合 thinking-pattern 不兼容问题
- 详见：`研究方向_Latent_OPD.md`

---

## 四、完整参考文献与链接

### 博客 / 综述 / 在线资源

| # | 标题 | 链接 | 时间 | 说明 |
|---|------|------|------|------|
| B1 | Thinking Machines Lab: On-Policy Distillation | https://thinkingmachines.ai/blog/on-policy-distillation/ | 2025.10 | 最全面的 OPD 工程实践指南，本文核心来源 |
| B2 | 知乎：从技术报告看 On-Policy Distillation 的崛起 | https://zhuanlan.zhihu.com/p/2031101471563962191 | 2026.01 | OPD 成为后训练新范式的全景分析 |
| B3 | Penaloza et al.: Understanding Self-Distillation and Privileged Information Distillation | https://emilianopp.github.io/Privileged-Information-Distillation-and-Self-Distillation/ | 2026.02 | Forward KL vs Reverse KL 的最佳入门（本文 Preliminary 来源） |
| B4 | AwesomeOPD (GitHub) | https://github.com/thinkwee/AwesomeOPD | 持续更新 | OPD 论文与技术报告合集 |
| B5 | Awesome Latent Space Reasoning | https://github.com/YU-deep/Awesome-Latent-Space | 持续更新 | Latent space 推理论文合集 |

### 创始论文（OPD 思想源头）

| # | 论文 | 链接 | 会议/时间 | 贡献 |
|---|------|------|----------|------|
| P1 | Ross et al., "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (DAGGER) | https://arxiv.org/abs/1011.0686 | AISTATS 2011 | OPD 的思想源头：交互式模仿学习 + on-policy 数据聚合 |
| P2 | **Agarwal et al., "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"** (GKD) | https://arxiv.org/abs/2306.13649 | ICLR 2024 | **OPD 开创性工作**：将 DAGGER 移植到 LLM KD，引入 on-policy + reverse KL |
| P3 | Gu et al., "MiniLLM: Knowledge Distillation of Large Language Models" | https://arxiv.org/abs/2306.08543 | ICLR 2024 | Reverse KL 在 LLM 蒸馏中的首次系统分析 |

### 旗舰模型技术报告（OPD 工业化的里程碑）

| # | 模型 | 链接 | 时间 | OPD 用法 |
|---|------|------|------|---------|
| T1 | **Gemma 2** (Google) | https://arxiv.org/abs/2408.00118 | 2024.06 | 首个在 production 中使用 OPD 的模型，9B 从更大教师蒸馏 |
| T2 | **Qwen3** (Alibaba) | https://arxiv.org/abs/2505.09388 | 2025.05 | **分水岭报告**：首个系统性阐述 Strong-to-Weak OPD，仅 1/10 GPU 时间达 RL 同等性能 |
| T3 | **DeepSeek-R1-0528** | https://huggingface.co/deepseek-ai | 2025.11 | OPD 用于大规模推理模型蒸馏 |
| T4 | **MiMo-V2-Flash** (Xiaomi) | https://github.com/XiaomiMiMo/MiMo-V2-Flash | 2025.12 | **MOPD（多教师 OPD）**首创：5 个领域教师在 logit 空间融合，学生超越最强单教师 |
| T5 | **GLM-5** (Zhipu) | https://arxiv.org/abs/2512.16173 | 2025.12 | **Cross-Stage OPD**：多阶段 RL 后通过 OPD 恢复被遗忘的能力 |
| T6 | **Qwen3.5** (Alibaba) | https://huggingface.co/Qwen | 2026.01 | Gated Delta Networks + OPD 后训练，0.8B~397B |
| T7 | **DeepSeek-V4** (DeepSeek) | https://arxiv.org/abs/2604.16296 | 2026.04 | **OPD 完全替代 Mixed RL**：10+ 专家模型蒸馏为一，Codeforces 3206 |
| T8 | **Qwen3.5-Omni** (Alibaba) | https://arxiv.org/abs/2604.15804 | 2026.04 | OPD 第二阶段训练用于全模态旗舰模型 |

### OPD 机制分析与改进（学术论文）

| # | 论文 | 链接 | 时间 | 核心发现 |
|---|------|------|------|---------|
| A1 | Li et al., "Small Models Struggle to Learn from Strong Reasoners" | https://arxiv.org/abs/2502.12143 | ACL 2025 | ≤3B 模型从长 CoT 蒸馏反而退步，Mix Distillation 方案 |
| A2 | Yin et al., "Data Shifts Hurt CoT: A Theoretical Study" | https://arxiv.org/abs/2506.10647 | 2025.06 | 首次理论证明 distribution shift 下 CoT 比不推理更差 |
| A3 | Shen et al., "Merge-of-Thought Distillation" | https://arxiv.org/abs/2509.08814 | 2025.09 | 不同学生需要不同教师，不存在万能教师 CoT |
| A4 | Jin et al., "Entropy-Aware On-Policy Distillation of Language Models" | https://arxiv.org/abs/2603.07079 | 2026.03 | Reverse KL 导致 mode collapse，entropy-aware 自适应方案 |
| A5 | Wu et al., "Rethinking Kullback-Leibler Divergence in Knowledge Distillation for Large Language Models" (AKL) | https://arxiv.org/abs/2404.02657 | COLING 2025 | 自适应组合 Forward + Reverse KL |
| A6 | "Reasoning Path Divergence" | https://arxiv.org/abs/2510.26122 | 2025 | 提出 RPD 度量化 CoT 推理路径间的语义距离 |
| A7 | **Fu et al., "Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes"** | https://arxiv.org/abs/2603.25562 | 2026.03 | **OPD 三类失败模式**：不平衡单 token 信号、教师引导不可靠、tokenizer 不匹配 |
| A8 | **Li et al. (Tsinghua), "Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe"** | https://arxiv.org/abs/2604.13016 | 2026.04 | **OPD 机制最系统分析**：两条件、分布不可区分性、97-99% 概率质量、长序列崩塌 |
| A9 | Song et al., "A Survey of On-Policy Distillation for Large Language Models" | https://arxiv.org/abs/2604.00626 | 2026.04 | OPD 综述：三维分类法 + 开放问题 |
| A10 | **MAD-OPD: Breaking the Ceiling in OPD via Multi-Agent Debate** | https://arxiv.org/abs/2605.01347 | 2026.05 | 多教师辩论驱动 OPD，突破单教师能力天花板 |
| A11 | Lightning OPD: Efficient Post-Training with Offline On-Policy Distillation | https://arxiv.org/abs/2604.09745 | 2026.04 | 完全离线 OPD，速度 4×，AIME 2024 达 69.9% |
| A12 | StableOPD | https://arxiv.org/abs/2604.08527 | 2026.04 | 解决 OPD 训练中的重复崩溃问题 |
| A13 | SCOPE: Sample-Conditioned On-Policy Distillation | https://arxiv.org/abs/2604.10688 | 2026.04 | 区分答对/答错轨迹的差异化 OPD 加权策略 |
| A14 | **Yang et al., "Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation" (G-OPD / ExOPD)** | https://arxiv.org/abs/2602.12125 | 2026.02 | **OPD = β=1 RL 特例**；引入 λ 和 π_ref → G-OPD 框架；ExOPD (λ>1) 唯一能超越所有领域教师；Reward Correction 提升 strong-to-weak 蒸馏 |

### Privileged Information + Self-Distillation 系列

| # | 论文 | 链接 | 时间 | 核心发现 |
|---|------|------|------|---------|
| S1 | **Penaloza et al., "Privileged Information Distillation for Language Models"** (π-Distill) | https://arxiv.org/abs/2602.04942 | 2026.02 | Forward/Reverse KL 理论基础，Reward-Tilted Distillation，Variational EM → π-Distill，本文 Preliminary 核心来源 |
| S2 | Shenfeld et al., "Self-Distillation Enables Continual Learning" | https://arxiv.org/abs/2601.19897 | 2026.01 | 自蒸馏防止遗忘，持续学习框架 |
| S3 | Zhao et al., "Self-Distilled Reasoner" | https://arxiv.org/abs/2601.18734 | 2026.01 | On-policy self-distillation for reasoning |
| S4 | Hübotter et al., "Reinforcement Learning via Self-Distillation" | https://arxiv.org/abs/2601.20802 | 2026.01 | Jensen-Shannon Divergence 改进纯 Reverse KL |
| S5 | Ye et al., "On-Policy Context Distillation for Language Models" | https://arxiv.org/abs/2602.12275 | 2026.02 | On-policy context distillation |

### Latent Space Reasoning 相关（与 OPD 结合的前景）

| # | 论文 | 链接 | 时间 | 核心发现 |
|---|------|------|------|---------|
| L1 | DeepMind, "Effective Reasoning Chains Reduce Intrinsic Dimensionality" | https://arxiv.org/abs/2602.09276 | 2026.02 | 内在维度与 OOD 泛化相关性 0.93，远超长度的 0.31 |
| L2 | Coconut: Training Large Language Models to Reason in a Continuous Latent Space | https://arxiv.org/abs/2412.06769 | ICLR 2025 | 首次 latent space 连续思维 |
| L3 | CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation | https://arxiv.org/abs/2502.21074 | EMNLP 2025 | Hidden state 对齐自蒸馏，压缩 CoT |
| L4 | Liang & Pan, "Do Latent-CoT Embeddings Learn to Reason Step-by-Step?" | https://arxiv.org/abs/2601.08847 | 2026.01 | CODI 不是逐步推理，长链/分布外崩塌 |
| L5 | Li et al., "Chain Of Thought Compression: A Theoretical Analysis" | https://arxiv.org/abs/2601.21576 | 2026.01 | 理论证明跳步导致高阶依赖信号指数衰减 |
| L6 | Lin et al., "Implicit Reasoning = Shortcut Learning" | ACL 2025 | ACL 2025 | 实验证明隐式推理本质是捷径学习 |
| L7 | CoLaR: Compressing Long Chain-of-Thought into Continuous Space via Latent Reasoning | https://arxiv.org/abs/2505.16552 | NeurIPS 2025 | 动态 latent 压缩推理链 |

### RL 与一般理论

| # | 论文 | 链接 | 说明 |
|---|------|------|------|
| R1 | Levine, "Reinforcement Learning and Control as Probabilistic Inference" | https://arxiv.org/abs/1805.00909 | RL-as-inference 框架的经典教程 |
| R2 | Lightman et al., "Let's Verify Step by Step" | https://arxiv.org/abs/2305.20050 | ICLR 2024 | Process reward modeling |
| R3 | Rafailov et al., "Direct Preference Optimization" (DPO) | https://arxiv.org/abs/2305.18290 | NeurIPS 2023 | 模型自身作为 reward model 的理论基础 |
| R4 | Shah et al., "A Comedy of Estimators: On KL Regularization in RL Training of LLMs" | https://arxiv.org/abs/2512.21852 | 2026 | β 选择的系统性研究 |

---



## 五、一句话总结

> OPD 在 18 个月内完成了从 ICLR 论文到后训练新范式的跨越。2026 年的核心命题从"要不要用 OPD"变成了"如何用 latent space / multi-teacher / adaptive KL / offline precompute 来突破 OPD 的 token 空间瓶颈和长序列限制"。
