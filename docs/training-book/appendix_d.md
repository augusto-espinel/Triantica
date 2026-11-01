## Appendix D: AI-Assisted Engineering Telemetry — Token Economics, Model Selection (`GLM-5` vs. `Argon`) & Context Architecture {#appendix-d-token-economics}

Alongside the reinforcement learning telemetry captured by the training pipeline (`src/server/train_tfjs.cjs`), the engineering of Triantica itself was conducted through an agentic coding runtime (OpenClaw on Apple Mac Studio M2 Ultra) backed by OpenRouter foundation models (`z-ai/glm-5` and `google/gemini-4-argon`) and a local Ollama compaction model (`ollama/gemma4:e4b`).

By correlating the **17,928 deduplicated LLM API calls (`1.18 billion` billed prompt + completion tokens)** recorded in the local agent session transcripts (`~/.openclaw/agents/main/sessions/*.jsonl`) and the **OpenRouter server-side billing API** (`/api/v1/activity`, reconciled Oct 18 – Nov 1) against the **190 Git commits** and **6 architectural phases (`exp12` → `TF-B15`)**, we can evaluate the token economics of building a zero-to-mastery AlphaZero system.

> **Data reconciliation note:** Local `.jsonl` logs include compaction checkpoint re-evaluations and sub-agent spawns that inflate call counts beyond OpenRouter's per-day per-model aggregates. Cost figures below are **OpenRouter-authoritative** (actual invoiced spend). Local call counts are provided for operational context.

### 1. Model Efficiency Comparison: `z-ai/glm-5` vs. `google/gemini-4-argon`

| Metric | `z-ai/glm-5` (Oct 18–25 + Sub-Agent Runs) | `google/gemini-4-argon` (Oct 26 – Nov 1) | `ollama/gemma4:e4b` (Local M2 Ultra) |
| :--- | :--- | :--- | :--- |
| **OR-Server Requests** | **8,632** (via `siliconflow/fp8`: `8,626` + `amazon-bedrock`: `6`) | **2,881** (via `google-vertex`: `2,789` + `google-ai-studio`: `92`) | N/A (local) |
| **Local Log Calls (incl. checkpoints)** | **13,312 calls** (`11,879` tool / `1,417` stops) | **4,540 calls** (`3,973` tool / `567` stops) | **79 calls** (compaction/utility) |
| **Total Prompt Volume (In + Cache)** | **558.22M tokens** (OR server) | **201.98M tokens** (OR server) | **5.10M tokens** |
| **Uncached Input Tokens** | **31.53M tokens** (`5.6%` of prompt) | **49.77M tokens** (`24.6%` of prompt) | **5.10M tokens** |
| **Cached Input Tokens** | **526.70M tokens** (**`94.4%` cache hit rate**) | **152.21M tokens** (**`75.4%` cache hit rate**) | `0` (local unified memory) |
| **Completion Tokens** | **2.03M** (incl. `0.68M` reasoning = `33.3%`) | **1.20M** (incl. `0.17M` reasoning = `14.2%`) | **0.08M** |
| **Total Invoiced Cost (OR Server)** | **$140.42** | **$126.58** | **$0.00** |
| **Average Cost per OR Request** | **$0.0163 / req** | **$0.0439 / req** (`~2.7×` higher per request) | **$0.0000** |
| **Effective Blended Rate** | **$0.243 / 1M prompt tokens** | **$0.623 / 1M prompt tokens** (`~2.6×` higher per token) | **$0.000** |

#### Architectural & Economic Findings (`GLM-5` vs. `Argon`)

1. **Unit-Token Economics Favor `GLM-5` by ~2.6× (`$0.243/1M` vs. `$0.623/1M` invoiced):**
   - `z-ai/glm-5` achieved a remarkable **94.4% prompt cache hit rate** (`526.7M` cached vs. `31.5M` uncached input tokens), keeping the average cost of an agentic request at **`$0.0163`**.
   - A full **33.3%** of `glm-5`'s output tokens were chain-of-thought reasoning (vs. only `14.2%` for `argon`), suggesting `glm-5` was doing substantial internal deliberation that still didn't suffice for the coupled MCTS/loss bugs.
   - During Phases 1–4 (`brain.js` exploration `exp12`–`exp22`, the `@tensorflow/tfjs-node` migration `TF-B1`–`TF-B7`, and the interactive telemetry dashboard), `glm-5` delivered 135 commits across 8 days (`Oct 18–25`) for **$128.57** invoiced (`~$16.07/day`).

2. **Reasoning Depth & Algorithmic Convergence Favor `Argon` on High-Entropy Math/Search Bugs:**
   - Between Oct 23 and Oct 25 (`TF-B5` through `TF-B8`), `glm-5` executed **`4,034` OR-server requests (`~263M` prompt tokens, `$60.31`)** while struggling to isolate coupled mathematical bugs across MCTS tree backup sign conventions (`Bug 5` & `Bug 6`), `1/400` policy cross-entropy dilution (`Bug 11`), and non-quiet leaf horizon blindness.
   - Switching the primary architect model to `google/gemini-4-argon` on **Oct 26** immediately resolved the MCTS perspective/loss coupling (`TF-B9` → `84%` vs. Random), designed **Quiescence MCTS** (`TF-B12`–`TF-B14` → breaking the `0%` Heuristic Wall up to `90%`), formulated **Full $T^2$ Toroidal Circular Padding (`ToroidalPad2D`) + Top-12 Safety Filtering** (`TF-B15` → **`100.0%` mastery across `1,000` evaluation games**), and built the synchronized 2.5D/3D Torus UI (`Torus3DView.js`).
   - `Argon` also routed **96.9%** of its traffic through `google-vertex` (vs. `google-ai-studio` fallback), achieving a **75.9% cache hit** on Vertex—significantly better than the **56.6%** hit on AI Studio fallback.

3. **Where Over-Spend Occurred (`Argon` on High-Frequency Mechanical Edits):**
   - While `argon` was dramatically more cost-effective per *algorithmic breakthrough*, using it as the default model for high-iteration UI/CSS tweaking, icon geometry adjustments, and multi-file documentation sweeps (e.g., **Oct 30**: `525` `argon` OR-server requests costing **$21.25** in a single day) incurred a `2.7×` per-request premium where `glm-5` would have cost `~$8`.

---

### 2. Daily Token Spend vs. Git Commits & Engineering Milestones (Authoritative OR Billing, Oct 18 – Nov 1)

| Date | #Cmt | GLM-5 OR Req / $ | Argon OR Req / $ | Uncached In | Cache % | Reas % | Day $ | $/Cmt | Key Milestone |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Oct 18** | 9 | 449 / $9.22 | 0 / $0 | 3.36M | 89.4% | 35.7% | **$9.22** | $1.02 | `brain.js` reward engineering & trajectory analyzer |
| **Oct 19** | 4 | 618 / $11.56 | 0 / $0 | 3.74M | 91.1% | 38.9% | **$11.56** | $2.89 | Experience replay prototype & state mismatch bugfix |
| **Oct 20** | 12 | 761 / $14.55 | 0 / $0 | 4.94M | 90.5% | 44.7% | **$14.55** | $1.21 | `HeuristicPlayer`, TD(λ) cluster shaping, `exp10`–`exp13a` |
| **Oct 21** | 12 | 763 / $12.02 | 0 / $0 | 2.96M | 93.6% | 58.1% | **$12.02** | $1.00 | Opponent-perspective learning & Phaser UI debug |
| **Oct 22** | 14 | 952 / $16.87 | 0 / $0 | 4.32M | 93.3% | 41.6% | **$16.87** | $1.20 | Reward normalization fix & GitHub Pages Dashboard |
| **Oct 23** | 36 | 1,410 / $21.71 | 0 / $0 | 3.89M | 95.7% | 35.9% | **$21.71** | $0.60 | `exp20` (91% vs Random) + TF.js migration (`TF-B1`–`B3`) |
| **Oct 24** | 27 | 1,437 / $21.38 | 0 / $0 | 3.18M | 96.5% | 22.2% | **$21.38** | $0.79 | Phase 1–4 `ReplayBuffer.cjs` + `ConvNetwork.cjs` (`TF-B4`–`B7`) |
| **Oct 25** | 21 | 1,187 / $17.22 | 0 / $0 | 3.01M | 95.8% | 31.2% | **$17.22** | $0.82 | Phase 5 `MCTS.cjs` & `TF-B8` sign-flip debugging |
| **Oct 26** | 12 | 262 / $3.64 | 543 / $29.23 | 12.31M | 77.9% | 12.1% | **$32.88** | $2.74 | **Switch to Argon**: Bug 6–11 fixes & `TF-B9` (84% peak) |
| **Oct 27** | 4 | 463 / $7.04 | 427 / $15.15 | 6.29M | 89.5% | 10.8% | **$22.18** | $5.55 | **Heuristic Wall Broken!** `QuiescenceMCTS` (`TF-B12` 63%) |
| **Oct 28** | 9 | 17 / $0.21 | 541 / $21.52 | 8.42M | 78.0% | 9.6% | **$21.73** | $2.41 | `TF-B13` (89%), `TF-B14` (90%, `v1.0.0-2d`), `ToroidalPad2D` |
| **Oct 29** | 18 | 19 / $0.19 | 409 / $20.39 | 8.14M | 72.5% | 23.8% | **$20.58** | $1.14 | Neon Manifold UI, 3D D20 intro & Tutorial |
| **Oct 30** | 14 | 175 / $2.85 | 525 / $21.25 | 9.11M | 79.9% | 19.4% | **$24.10** | $1.72 | **`TF-B15` (100% Champion)**, PWA + `HumanGameLogger` |
| **Oct 31** | 13 | 119 / $1.96 | 436 / $19.04 | 7.59M | 81.4% | 7.1% | **$21.00** | $1.62 | Training Book v4.1 modularization & SPA |
| **Nov 01** | 2 | 0 / $0 | 63 / $2.13* | ~1M | ~56% | — | **$2.13** | $1.06 | Playtesting analysis & token telemetry |
| **Total** | **207** | **8,632 / $140.42** | **2,881 / $126.58** | **88.80M** | **87.6%** | — | **$267.00** | **$1.29** | |

\* Nov 1 from `/api/v1/auth/key` daily usage; activity API data not yet available for today.

**Account status:** Credits `$630.00` — Used `$601.70` = **Remaining `$28.30`**

---

### 3. OpenClaw Context Overhead & Why the Artificial ~100K Window Was Counterproductive

#### A. Dissecting the ~26.9K Token Startup Baseline
Across 43 fresh sessions in the telemetry logs, **Turn 1 begins with an average of `26,907` uncached input tokens** before a single project file is read. That baseline consists of:

1. **Tool Schema & Runtime System Prompt (`~8,500` tokens):** JSON schemas for `20+` tools (`exec`, `read`, `edit`, `write`, `sessions_spawn`, `cron`, `memory_*`, etc.) plus behavioral directives.
2. **Workspace Identity & Long-Term Memory (`~3,500–8,000` tokens):** `AGENTS.md`, `SOUL.md`, `USER.md`, `IDENTITY.md`, `TOOLS.md`, `HEARTBEAT.md`, and `MEMORY.md` (which had grown to `~19 KB` prior to the Oct 31 archival compaction).
3. **Rolling Daily Memory Injection (`startupContext.dailyMemoryDays: 3`, `~10,000–14,000` tokens):** Automatically injecting the last 3 days of raw `memory/YYYY-MM-DD.md` logs into the prompt prefix on every session start.

Because agentic coding averages **7.5 tool calls per user message**, any bloat in the static startup prefix is multiplied across thousands of turns.

#### B. The "Compaction Thrashing" Penalty of an Artificial ~108K Context Ceiling
Prior to expanding `agents.defaults.contextTokens` to `400,000` on October 31, the runtime operated under an effective **`~108,000`-token ceiling** (visible in the daily max prompt column of local logs: `108,352` on Oct 18, `108,583` on Oct 22–23, `107,008` on Oct 30). That smaller artificial window proved **actively counterproductive** for both cost and engineering velocity:

1. **167 Compaction Checkpoints in 14 Days (`12–23` per day during peak coding):**
   - Reading monolithic files (such as the 3,546-line `TRAINING_BOOK.md` at `~45K` tokens or `game.js` + `train_tfjs.cjs`) pushed sessions from the `27K` baseline past the `108K` threshold in just **15–25 tool calls**.
2. **KV-Cache Destruction (`cacheRead` Drop from `96%` to `71–78%`):**
   - Provider prompt caching relies on an **immutable token prefix**. Every time compaction summarized the middle of a transcript or a session reset occurred, the token prefix changed—invalidating the KV cache and forcing the next turn to bill `50K–80K` tokens at **100% uncached input rates**.
3. **Rediscovery Amnesia Loop:**
   - After each compaction, the agent lost exact line numbers, tensor shapes, or CSS selectors from earlier in the hour, forcing it to re-run `read` and `grep` commands—re-filling the context window with the exact same file contents and triggering yet another compaction cycle.

---

### 4. Architectural Playbook: Making Future RL & Software Projects 65–75% Less Token-Expensive

1. **Modularize Large Documentation & Source Files Early (`< 500` lines per file):**
   - Splitting `TRAINING_BOOK.md` (`3,546` lines) into 18 modular chapter files under `docs/training-book/` reduced the per-edit file read footprint from `~45,000` tokens to `~2,500` tokens—a **94% reduction in context growth per edit**.

2. **Tiered Model Routing (Architect vs. Specialist Sub-Agents):**
   - Reserve `google/gemini-4-argon` (`$0.62/1M` blended) strictly for **high-entropy architectural design, mathematical derivations, and subtle bug audits** (where its 1-shot accuracy prevents multi-hour debugging loops).
   - Delegate **low-entropy mechanical execution** (CSS layout adjustments, wrapping ASCII diagrams in `<details>`, syncing JSON files, writing unit tests) to `z-ai/glm-5` sub-agents (`$0.24/1M` blended, `~1.6¢/call`).

3. **Keep Startup Context Lean (`< 10K` tokens) & Use On-Demand Recall:**
   - Keep `MEMORY.md` curated under `5 KB` (archiving historical tables to `memory/archive-*.md`) and set `startupContext.dailyMemoryDays` to `1` (relying on `memory_search` + `memory_get` only when prior session details are needed).

4. **Wide Context Ceiling (`400K`) + Surgical Tool Reads (`offset`/`limit` & Python Inspection):**
   - Pairing a wide `400K` context ceiling (to   eliminate mid-task compaction thrashing and preserve `90%+` KV-cache hit rates) with surgical line-bounded reads (`read` with `offset`/`limit` or compact `python3`/`grep` summaries) prevents raw file dumps from polluting the cached prefix.

5. **Route Provider Traffic Deliberately:**
   - OpenRouter auto-routes across providers, but cache hit rates vary dramatically. In this project, `google-vertex` achieved **75.9% cache hit** for `argon` vs. only **56.6%** on `google-ai-studio`. Where possible, pin to the provider with the best caching characteristics for your workload pattern.

---

*Last updated: 2026-11-01*
*Version: 4.3 (Authoritative Complete Edition — Phase 1 to Phase 6 Full Torus $T^2$ Mastery + Engineering Token Telemetry)*
*Framework: TensorFlow.js (`@tensorflow/tfjs-node`) on Apple Silicon (AMX / Accelerate)*
*Hardware: Mac Studio M2 Ultra (24-core CPU, 60-core GPU, 64GB Unified Memory)*

---

> [← Chapter 12: Toroidal 3D Visualization & Human Telemetry](./12-toroidal-3d-and-telemetry.md) | **[📚 Training Book Index](./README.md)** | [Book Index →](./README.md)
