# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a research knowledge base documenting **On-Policy Distillation (OPD)** and its intersection with **Latent Space Reasoning**. It contains no executable code — it is analysis, literature review, and research direction planning.

## Key Documents

| File | Role |
|------|------|
| [OPD介绍_ThinkingMachines_Blog.md](OPD介绍_ThinkingMachines_Blog.md) | OPD primer: what it is, how it works, key experiments (math reasoning, personalization/continual learning). Source: Thinking Machines Lab blog (Kevin Lu, 2025.10). |
| [OPD_发展路线_起源与现状.md](OPD_发展路线_起源与现状.md) | Full OPD development roadmap: DAGGER (2010) → GKD/MiniLLM (2023) → Qwen3/MiMo-V2/GLM-5/DeepSeek-V4 industrial adoption (2025-2026). Deep-dives into 3 key papers (Tsinghua mechanism analysis, failure modes, G-OPD/ExOPD). Complete reference list. |
| [研究方向_Latent_OPD.md](研究方向_Latent_OPD.md) | Four proposed research directions combining OPD with latent space reasoning to overcome OPD's token-space bottlenecks. Includes priorities, roadmaps, experimental designs, and sanity checks. |

## Document Relationships

- `OPD介绍` — entry point, explains the core OPD concept
- `OPD_发展路线` — comprehensive history + current state, references both other docs
- `研究方向_Latent_OPD` — forward-looking, depends on understanding both OPD mechanics (from the other two docs) and latent space reasoning literature

## Research Context

The central thesis across all three documents: **token space is OPD's bottleneck** (only ~3% of tokens carry gradient signal; teacher reliability decays on long sequences; larger models aren't always better teachers). The proposed escape route is **latent space OPD** — continuous representations instead of discrete tokens, multi-step latent rollout instead of single-point alignment, and selective compression instead of lossy full-sequence encoding.

## Key Technical Terms

- **OPD** = On-Policy Distillation (student generates its own rollouts, teacher scores each token)
- **GKD** = Generalized Knowledge Distillation (Agarwal et al., ICLR 2024)
- **Reverse KL** = KL(student ‖ teacher), mode-seeking — the OPD loss function
- **Forward KL** = KL(teacher ‖ student), mode-covering — the SFT loss function
- **MOPD** = Multi-Teacher OPD (MiMo-V2)
- **ExOPD** = OPD with reward extrapolation (G-OPD framework, λ > 1)
- **CODI** = Compressing CoT into Continuous Space via Self-Distillation
- **Coconut** = Continuous latent space reasoning (ICLR 2025)
