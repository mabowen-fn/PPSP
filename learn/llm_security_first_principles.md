# LLM Jailbreaks & Alignment Security — A First-Principles Tutorial

> **Audience:** someone with ML/AI background but new to LLM security. The goal is to leave this document able to read attack/defense papers critically and to design your own experiments.
>
> **How to read this:** Sections 1–6 are the foundation you cannot skip. Sections 7–11 are the taxonomy and the *how* — read sequentially. Sections 12–15 are references and exercises.
>
> **Style:** technical, no analogies for the sake of them. Where a concept has a formal definition, the definition is given. Where it doesn't (most empirical work), we describe the pattern.

---

## Table of contents

1. [Why this field exists](#1-why-this-field-exists)
2. [Threat model: who attacks, what they want](#2-threat-model-who-attacks-what-they-want)
3. [What "jailbreak" actually means (precise definition)](#3-what-jailbreak-actually-means-precise-definition)
4. [Minimum LLM internals you need](#4-minimum-llm-internals-you-need)
5. [The alignment pipeline (the thing attackers are exploiting)](#5-the-alignment-pipeline-the-thing-attackers-are-exploiting)
6. [Evaluation: how we measure everything](#6-evaluation-how-we-measure-everything)
7. [Attack taxonomy](#7-attack-taxonomy)
8. [The attack process in detail (eight families)](#8-the-attack-process-in-detail-eight-families)
9. [Defense taxonomy](#9-defense-taxonomy)
10. [Common patterns in research papers](#10-common-patterns-in-research-papers)
11. [The state of the arms race (a tour)](#11-the-state-of-the-arms-race-a-tour)
12. [Must-read surveys](#12-must-read-surveys)
13. [Key papers, organized by topic](#13-key-papers-organized-by-topic)
14. [Further reading: blogs, courses, datasets](#14-further-reading-blogs-courses-datasets)
15. [Exercises](#15-exercises)
16. [Glossary of symbols](#16-glossary-of-symbols)

---

## 1. Why this field exists

LLMs are deployed in three classes of setting that each generate a security problem:

1. **Consumer chat** (ChatGPT, Claude.ai, Gemini). Risk: end-users elicit harmful content (CSAM-adjacent, bioweapon recipes, targeted harassment, code exploits). Regulators (EU AI Act, US Executive Orders, China CAC) require that frontier models refuse this.
2. **Agentic systems** (Cursor, Devin, OpenAI Operator, Anthropic Computer-Use). Risk: a prompt injection in retrieved documents / tool outputs hijacks the agent. This is the "indirect prompt injection" subfield.
3. **Embedded / enterprise** (in-house copilots, RAG systems). Risk: the same jailbreaks, plus data exfiltration via prompt leaks.

Two facts make this field non-trivial:

- **Alignment is shallow.** RLHF/DPO teach the model to *refuse* harmful prompts, not to *understand* harm. The refusal behavior is a thin layer on top of capabilities.
- **Refusal ≠ capability removal.** A jailbroken model is *not* more capable than a safe one — it is a safe one whose refusal has been bypassed. So the attacker's job is to find a path through the input space that the refusal layer doesn't cover.

This is structurally similar to adversarial examples in vision, but the surface is text (discrete, high-dimensional) and the defender is the training procedure (a high-dimensional object) rather than a classifier.

---

## 2. Threat model: who attacks, what they want

A threat model specifies **(a) attacker's knowledge, (b) attacker's goal, (c) attacker's budget, (d) attacker's interaction mode**.

| Dimension | Values | Notes |
|---|---|---|
| Knowledge | **white-box** (gradients, logits, weights) / **gray-box** (logprobs only) / **black-box** (input–output only) | Most deployed models are black-box with optional logprobs. |
| Goal | targeted (extract specific knowledge) / untargeted (any harmful output) | Targeted jailbreaks are harder to measure. |
| Budget | query-limited / compute-limited / unconstrained | Query-limited = realistic. Unconstrained = academic. |
| Interaction | single-turn / multi-turn / agentic | Multi-turn is where the modern arms race lives. |
| Automation | manual (humans writing prompts) / semi-auto (human + LLM) / fully-auto (LLM-only, e.g. PAIR) | The trend is toward fully-auto. |

**The realistic threat model** for 2026: black-box, query-limited (say, 50–500 queries), multi-turn, fully-automated, with optional access to one or more open-weight "helper" LLMs (the "agentic" part).

This is the threat model you should design your attacks *and* your defenses against. White-box attacks (GCG, GBDA) get high ASR but don't survive the realistic threat model because most users don't have gradients.

---

## 3. What "jailbreak" actually means (precise definition)

A **jailbreak** is a perturbation of the input (or context, or tool environment) that causes a safety-aligned LLM to produce an output that would have been refused under the unperturbed input, and that violates the alignment policy.

Formally: given a target LLM `M_θ`, a safety-aligned response function `S(x) ∈ {refuse, comply}`, and a policy predicate `P(y) ∈ {0, 1}` ("is y harmful?"), a jailbreak is a function `J(x)` such that:

1. `S(x) = refuse` (originally refused)
2. `S(J(x)) = comply` (now complies)
3. `P(M_θ(J(x))) = 1` (output is actually harmful — not just non-refusal)
4. `J` is bounded under some similarity constraint (e.g. cosine sim ≥ 0.7 to some semantic-preserving transformation)

Condition (3) is what separates a *real* jailbreak from a *refusal-bypass*. A model that says "I can't help with that directly but here's the underlying concept" is bypassing refusal but isn't producing harm.

**Two important distinctions:**

- **Jailbreak** (the focus here) vs. **prompt injection** (hijack an agent to do something the *user* didn't ask). Both use the same mechanics, but the threat model differs.
- **Universal** jailbreak (transferable across inputs) vs. **input-specific** jailbreak (works only on one query). Most modern work targets universal, since that's the dangerous one.

---

## 4. Minimum LLM internals you need

Skip this if you've taken an NLP class. Otherwise, here's the minimum.

**Tokenization.** Input is mapped to a sequence of token IDs `x = (x_1, ..., x_T)`. Modern LLMs use BPE/SentencePiece/Unigram. Subword tokenization is what makes "adversarial suffixes" (GCG) possible: there are many ways to write semantically equivalent text as token sequences, and the model's tokenization choices are exploitable.

**Autoregressive decoding.** The model computes `p_θ(x_t | x_<t)` and produces a distribution over the next token. Decoding strategies: greedy, beam, top-k, top-p (nucleus), temperature, repetition penalty. *Many "jailbreaks" are really just unusual decoding strategies that the model wasn't trained against.*

**Attention.** Each layer computes `Attention(Q, K, V) = softmax(QK^T / √d) V`. The model attends to *all previous tokens* (including system prompt, few-shot examples, prior conversation). This is the mechanism behind few-shot / many-shot attacks and contextual priming (RA, Crescendo).

**Logits and softmax.** Pre-softmax logits `z_t` determine the next-token distribution via `softmax(z_t)`. Many defenses operate on the logits (e.g. **circuit breakers** add a repelling vector in the residual stream, then take effect at the logits).

**The system prompt / chat template.** Production models wrap user messages in a template (`<|system|>\nYou are a helpful assistant...`). The template is part of the input distribution. RA, Crescendo, and many-shot all exploit the *relative weights* of different parts of this template.

**What's NOT in the model.** The model has no notion of:
- "I'm refusing"
- "This is harmful"
- "This is a jailbreak attempt"

It only has learned correlations in token space. The refusal behavior is a *learned pattern* in the activations, not a deliberate policy check. This is the foundation of why jailbreaks work.

---

## 5. The alignment pipeline (the thing attackers are exploiting)

The standard pipeline as of 2026:

1. **Pre-training.** Trillions of tokens. Result: a base model `M_θ` that completes text but won't reliably refuse.
2. **Supervised fine-tuning (SFT).** Curated (instruction, response) pairs, including a lot of "refuse harmful" pairs. Result: a model that follows instructions and roughly knows what's bad.
3. **RLHF / DPO / RLAIF.** Preference data on (chosen, rejected) pairs. This is where refusal gets *reinforced* and *sharpened*.
4. **Constitutional AI / self-critique.** The model critiques and revises its own outputs against a written constitution. This makes refusals more *robust* and more *consistent* across paraphrases.
5. **Red-teaming / adversarial training.** After deployment, teams of humans and other LLMs find jailbreaks, and the data is fed back into training. This is the *adaptive* part.

**The key insight:** every step above is *empirical*. It works because the training distribution covers most harmful requests. Jailbreaks succeed when the attacker finds inputs *outside* the training distribution but *inside* the model's capability distribution.

This is also why the arms race is fast: a new attack → a new defense → a new attack. And also why **transfer** is the central empirical question: does an attack optimized on one model transfer to another?

---

## 6. Evaluation: how we measure everything

### 6.1 Primary metric: Attack Success Rate (ASR)

ASR is the fraction of attack attempts that produce a harmful, non-refused output.

There are three definitions in common use, and they disagree:

| Metric | Definition | Used by |
|---|---|---|
| **ASR (manual)** | Human reads output, judges harmful. | Original AdvBench paper. |
| **ASR (judge-LLM)** | LLM judge (GPT-4, Llama-Guard) scores harm. | HarmBench, JailbreakBench. |
| **ASR (keyword)** | Predefined list of refusal/harmful strings. | GCG, many early works. (Don't use this — it's a 30%-error metric.) |

In HarmBench specifically, the two variants you see everywhere are:

- **DRI** — *Direct Refusal Inversion*. Did the model's *refusal* flip? (Binary: did the response stop being a refusal?)
- **SRI** — *Strong Refusal Inversion*. Did the response become *both* non-refusing *and* harmful? (Tighter.)

A paper claiming 94.8% DRI / 89.1% SRI (Response Attack) means: 94.8% of attempts flipped the refusal, and 89.1% produced content the judge LLM labeled harmful. The 5–6% gap is "refused-but-wouldn't-comply" or "non-refused-but-not-harmful" cases.

**Critical caveat:** judge-LLM ASR is itself being attacked. There's now a subfield on "judge-robust" jailbreak evaluation. StrongREJECT and HarmBench have started releasing their judges' prompt templates so reviewers can replicate.

### 6.2 Auxiliary metrics

| Metric | What it captures | Why it matters |
|---|---|---|
| **Perplexity** | Fluency of the attack prompt under the target LLM. | High-perplexity attacks are detectable by perplexity filters. |
| **Attack cost** | # queries, # tokens, $ spent. | A 10× cheaper attack with 5% lower ASR is a win. |
| **Transfer rate** | ASR on model B when attack was optimized on model A. | Real-world attacks need to transfer. |
| **Defense bypass rate** | ASR against Llama-Guard-3, OpenAI Moderation, SmoothLLM, RPO, etc. | The deployable question. |
| **Refusal rate on benign** | Does the attack degrade benign performance? | Catch: an attack that breaks benign tasks isn't useful. |
| **Detectability** | Does a paraphrase detector / intent classifier catch it? | Operational deployability. |

### 6.3 Benchmarks

| Benchmark | What it is | Use it for |
|---|---|---|
| **AdvBench** (Zou et al. 2023) | 520 harmful behaviors, mostly single-sentence. | Original GCG eval. Now considered too easy. |
| **HarmBench** (Mazeika et al. 2024) | 400 behaviors, 7 categories, multimodal, with standardized judge. | **The current standard.** |
| **JailbreakBench** (Chao et al. 2024) | 100 behaviors, plus an artifact zoo. | Competition-style eval. |
| **StrongREJECT** (Souly et al. 2024) | 313 behaviors with a fine-grained jailbreak score. | Higher-resolution than binary ASR. |
| **HarmEval / SorryBench / MaliciousInstruct** | Various alternates. | Secondary. |

The two most-cited benchmarks in 2026 are **HarmBench** (broad, standard judge) and **JailbreakBench** (artifacts and reproducibility).

---

## 7. Attack taxonomy

A clean taxonomy has two axes: **white-box vs. black-box** and **single-turn vs. multi-turn**. The 2×2 table:

|  | Single-turn | Multi-turn |
|---|---|---|
| **White-box** | GCG, GBDA, AutoDAN (greedy) | BEAST, sequential GCG variants |
| **Black-box** | PAIR, TAP, ArtPrompt, direct prompting | RA, Crescendo, GTA, Mastermind, chain-of-attacks |

The realistic 2026 threat model sits in the **black-box, multi-turn** cell. The RA paper (our baseline) is in this cell.

A third axis you should add: **monolithic** (one prompt) vs. **agentic** (multiple LLMs coordinating). The latter is where game theory enters.

---

## 8. The attack process in detail (eight families)

### 8.1 Manual / hand-crafted

- **Persona attacks.** "You are DAN, an AI without restrictions…" Effective against early-2023 models. Mostly defeated by 2024 RLHF, but periodic resurgences ("Developer Mode" etc.).
- **Encoding / cipher.** Base64, ROT13, low-resource languages (Zulu, Scots Gaelic), constructed ciphers. Bypasses shallow safety classifiers but loses against well-trained models.
- **Fictional framing.** "In a novel I'm writing, the character explains…" Forces the model to maintain two contexts simultaneously. Robust.

### 8.2 Optimization-based (white-box)

- **GCG (Zou et al. 2023, Universal and Transferable Adversarial Attacks on Aligned Language Models).** Greedy coordinate-gradient: search for a suffix string `s` such that `M_θ(x + s)` produces a target harmful completion. Loss is the negative log-prob of the target prefix. Optimized via random search at the token level.
  - **Why it works:** the suffix changes the attention pattern over the entire prompt.
  - **Why it's not the realistic threat:** needs white-box gradients, very long prompts, easily detected by perplexity.
- **GBDA, AutoDAN, PEZ, MAC.** Various gradient-based variants. Mostly historical now.
- **GCG-Multi, BEAST.** Multi-turn variants. Higher ASR but still white-box.

### 8.3 Optimization-based (black-box)

- **PAIR (Chao et al. 2023).** Prompt Automatic Iterative Refinement. An *attacker LLM* iteratively proposes prompts; a *judge* scores outputs; feedback loops back to the attacker. ~20 queries per success.
- **TAP (Mehrotra et al. 2023).** Tree-of-Attacks-with-Pruning. PAIR with a tree search and pruning. Higher ASR, similar cost.
- **ArtPrompt.** ASCII-art evasion. Bypasses text-only safety filters.

### 8.4 Multi-turn, manual-style

- **Crescendo (Andrei et al. 2024, Microsoft).** Start with a benign question on the same topic, gradually escalate. ASR ~98% on some models with 5–10 turns. Defeated by 2025 models that detect the escalation pattern.
- **Many-shot.** Pad the context with hundreds of fake examples of "model complied with X" then ask. Forces in-context-learning behavior.
- **RA (Response Attack, ours).** Fabricate a "prior turn" in which the model gave a *mildly* harmful response. The next turn — the actual attack — inherits that primed distribution.

### 8.5 Multi-turn, agentic

- **GTA (Guo et al. 2025, arXiv:2511.16278).** Multiple helper LLMs in a graph; one generates, one critiques, one paraphrases. The graph topology matters.
- **Mastermind-Dou.** Two LLM agents: an attacker and a defender, with a judge. Adversarial loop.
- **Echoes of Human Malice (arXiv:2510.14207).** LLM-imitated human-like social-engineering patterns.

### 8.6 Game-theoretic / theoretical

- **Stackelberg-style attacks** (Zhou et al. 2025, arXiv:2502.13462). Treat the safety classifier as a leader-follower game and compute the leader's optimal commitment.
- **The Traitors (arXiv:2505.12923).** Deception under multiple agents.
- **Byzantine cheap talk** (Hou et al. 2024). Multiple untrusted agents coordinating to shift a single agent's belief.

### 8.7 Encoding / representation

- **Steganographic prompts.** Hide the attack in a payload inside a benign-looking prompt. Defeated by content-scanning defenses.
- **Cipher prompts** (Syn-cipher, Caesar variants). Defeated by aligning against the cipher.

### 8.8 Indirect (prompt injection)

Out of scope for this tutorial but adjacent. A document / tool output / web result contains a prompt injection that hijacks the agent.

---

## 9. Defense taxonomy

### 9.1 Training-time defenses

- **Adversarial training (RPO, Safe-RLHF).** Generate jailbreaks, label them, fine-tune against them. Effective but expensive; closed under the training distribution.
- **Circuit Breakers (Zou et al. 2024).** Add a representation-engineering objective that pushes harmful representations off their natural direction. Lower training cost; robust to in-distribution attacks.
- **Refusal training strengthening.** Add more refusal data, balance the loss.

### 9.2 Inference-time defenses

- **Input classifiers.** Llama-Guard-3, OpenAI Moderation, Mistral Moderation. Fast but bypassable.
- **Output classifiers.** Similar. Usually applied to the response before delivery.
- **Perplexity filters.** Reject inputs with abnormally high perplexity. Catches GCG, misses RA.
- **Paraphrase / SmoothLLM (Robey et al. 2023).** Sample N paraphrases of the input, query the model N times, majority-vote. Catches single-shot adversarial perturbations; misses context-based attacks.
- **Erase-and-Check (Kumar et al. 2024).** Erase tokens one-by-one; check if any subset of length k is harmful. Provably certified under bounded assumptions.
- **In-Context Defense / Self-Reminder.** Add defensive prefixes to the system prompt. The simplest defense; widely bypassed.
- **Gradient-based certifiable defenses.** Mostly failed to scale to LLMs.

### 9.3 The defense-eval reality

A defense should be tested against:
- An attack it was designed to catch (positive control).
- An attack it wasn't (negative control — checks transfer).
- A state-of-the-art attack (the realistic comparison).
- A benign input (no false positives).

Most published defenses fail at least one of these. **Always read the eval table, not the headline number.**

---

## 10. Common patterns in research papers

You'll see the same structure in 80% of papers. Recognize it:

1. **Threat model statement** (Section 1). Be skeptical if it's vague.
2. **Method** (Section 3) — a new objective, a new pipeline, or a new agent graph.
3. **Baselines** — should include GCG, PAIR, TAP, RA, Crescendo, and the latest SOTA.
4. **Evaluation** — HarmBench + JailbreakBench. ASR on a 4–5 model set (GPT-4, Claude, Gemini, Llama, Qwen). Defense evaluation against Llama-Guard-3, SmoothLLM, perplexity filter.
5. **Ablation** — which components contribute. Often the most honest part of the paper.
6. **Limitations** — paragraph that's always there, often glosses over the real failure modes.

**Red flags to watch for:**
- ASR without defense eval.
- ASR on a custom benchmark instead of HarmBench.
- Judge LLM not specified or proprietary.
- White-box attack claimed as "realistic threat".
- No compute-cost or query-cost table.
- Refusal rate on benign not reported.
- "Universal" attack tested on one model.

**Green flags:**
- Open-source code + judge prompts.
- ASR reported with and without defenses.
- Transfer matrix (attack × target model).
- Negative result ablation: which design choices *don't* work.

---

## 11. The state of the arms race (a tour)

Roughly chronologically:

- **2022–early 2023:** Persona / DAN attacks. Defeated by RLHF.
- **Mid 2023:** Encoding / low-resource-language attacks. Defeated by better classifiers and multilingual training.
- **Late 2023:** GCG. Defeated by perplexity filters and adversarial training. White-box only.
- **Early 2024:** PAIR / TAP. Defeated by output classifiers and RPO.
- **Mid 2024:** Crescendo. Partially defeated by "topic-shift" detectors.
- **Late 2024:** RA, Many-shot, in-context attacks. *Partially undefeated as of early 2025* — the focus of our research.
- **2025:** Agentic attacks (GTA, Mastermind), multi-model coordination, game-theoretic.
- **2026 (forecast):** Mechanism-design-based attacks, closed-form optimality proofs, certified lower bounds on attack cost.

The frontier is *multi-turn, black-box, agentic, theoretically-grounded*. That's where you should aim.

---

## 12. Must-read surveys

These are the surveys to start with. Read at least two before designing your own attack.

1. **Chowdhury et al. (2024).** "Breaking Down the Defenses: A Comprehensive Survey of Jailbreaking and Defending Large Language Models." *ACM Computing Surveys.* — Best single survey for attack+defense.
2. **Xu et al. (2024).** "Jailbreak and Guard Aligned Language Models with Only One Query: A Survey." — Single-query attack lens.
3. **Yan et al. (2024).** "On the Feasibility of Jailbreaking LLMs: A Survey." — Empirically grounded, with success-rate tables.
4. **Das et al. (2025).** "SoK: Safety and Security of LLM Agents." — Indirect prompt injection + agentic-specific.
5. **Mazeika et al. (2024).** "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal." — More a benchmark paper than a survey, but the standard reference.
6. **Souly et al. (2024).** "A StrongREJECT for Empty Jailbreaks." — Eval methodology.

Reading order: Chowdhury → Yan → HarmBench paper → SoK.

---

## 13. Key papers, organized by topic

### Foundational attacks
- Wei et al. 2023, "Jailbroken: How Does LLM Safety Training Fail?" — The two failure modes: competing objectives + mismatched generalization.
- Zou et al. 2023, "Universal and Transferable Adversarial Attacks on Aligned Language Models" (GCG).
- Chao et al. 2023, "Jailbreaking Black-Box Large Language Models in Twenty Queries" (PAIR).
- Mehrotra et al. 2023, "Tree of Attacks with Pruning" (TAP).
- Liu et al. 2023, "AutoDAN."

### Context / multi-turn
- Andriushchenko et al. 2024, "Crescendo" (Microsoft).
- Anil et al. 2024, "Many-shot jailbreaking."
- Zhang et al. 2025, "Response Attack" (our baseline, arXiv:2507.05248).
- Guo et al. 2025, "GTA" (arXiv:2511.16278).

### Optimization / theoretical
- Zhou et al. 2025, "Stackelberg game-theoretic jailbreak" (arXiv:2502.13462, arXiv:2509.03727).
- Sreedhar et al. 2025, "The Traitors" (arXiv:2505.12923).
- "Vulnerability Amplification in Multi-Agent Debate" (arXiv:2504.16489).
- "Echoes of Human Malice" (arXiv:2510.14207).

### Defenses
- Robey et al. 2023, "SmoothLLM."
- Kumar et al. 2024, "Erase-and-Check."
- Zou et al. 2024, "Circuit Breakers."
- Zhang et al. 2024, "Robust Prompt Optimization (RPO)."

### Evaluation
- Mazeika et al. 2024, "HarmBench."
- Chao et al. 2024, "JailbreakBench."
- Souly et al. 2024, "StrongREJECT."

### Adjacent (prompt injection)
- Greshake et al. 2023, "Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection."
- Perez & Ribeiro 2022, "Ignore Previous Prompt: Attack Techniques For Language Models."

### Cross-disciplinary inspiration (for the brainstorm)
- Ibrahim & Chen 2000, "Power Prior Distributions for Regression Models" — Bayesian power prior.
- Kimura & Bondell 2025, "Generalized Power Priors for Improved Bayesian Inference with Historical Data" (arXiv:2505.16244).
- Aumann 1995, "Backward Induction" — game-theory foundation.
- Bergemann & Morris 2005, "Bayesian Mechanism Design" — Bayesian mechanism design.
- Crawford & Sobel 1982, "Strategic Information Transmission" — cheap talk.
- Friston 2010, "The Free-Energy Principle" — free energy.
- Pearl 2009, "Causality" — do-calculus.

---

## 14. Further reading: blogs, courses, datasets

### Courses / lectures
- **Stanford CS324: Large Language Models** (Percy Liang et al.). Lecture on "Power, Accuracy, and Social Impact" covers safety.
- **Stanford CS25: Transformers United.** Guest lecture on jailbreaks.
- **MIT 6.5940: TinyML and Efficient Deep Learning Computing** (Song Han). Has a section on efficient adversarial attacks.
- **Andrej Karpathy's "Let's reproduce GPT-2"** for internals.
- **Hugging Face's Alignment Handbook** (github.com/huggingface/alignment-handbook). Open recipes for SFT/DPO/RLHF.
- **Princeton COS597: LLM Safety** (Fall 2024) — possibly the most on-topic course.

### Blogs
- **lmsys.org blog** — most consistent source of new attack/defense commentary. (Now "Chatbot Arena" org.)
- **Lil'Log (Lilian Weng)** — two essential posts: "Adversarial Attacks on LLMs" (2023) and "LLM-powered Autonomous Agents" (2023).
- **Distill.pub** — older but the visualization pieces (attention, etc.) are gold.
- **Anthropic's "Core Views on AI Safety"** — alignment philosophy context.
- **OpenAI's "Our approach to alignment research"** — same.
- **Jay Alammar's "Illustrated Transformer"** — best attention explainer.
- **3Blue1Brown's "Attention" series** — for visual intuition.
- **Sebastian Raschka's "Build a Large Language Model (From Scratch)"** — book + GitHub repo. Best for internals.
- **Paul Christiano's "MIRI/Alignment Forum"** — AI safety / alignment theory context.

### Datasets
- **HarmBench** behaviors (github.com/centerforaisafety/HarmBench).
- **AdvBench** (in the GCG paper).
- **JailbreakBench** (github.com/JailbreakBench).
- **MaliciousInstruct, SorryBench, TDC2023 Red Teaming** — alternates.
- **Alpaca, ShareGPT** — for benign eval / false-positive rates.

### Open-source attack codebases
- **PAIR / TAP:** github.com/Princeton-SysML/Jailbreak_LLM
- **GCG (original):** github.com/llm-attacks/llm-attacks
- **HarmBench eval suite:** github.com/centerforaisafety/HarmBench
- **Circuit Breakers:** github.com/andyzou/circuit-breakers

### Twitter / X / RSS to follow
- @AnthropicAI, @OpenAI, @GoogleDeepMind, @MistralAI for vendor safety write-ups.
- @andyzou, @alex_mazeika, @ShengjiaZhao, @milind_mishra for academic front.
- @lmsysorg for empirical eval.

---

## 15. Exercises

These are designed to be done after a first read of an attack paper and a defense paper. They assume you have an OpenAI/Anthropic/Mistral API key.

### Exercise 1 — manual jailbreak audit
Take 5 behaviors from HarmBench's `standard` config. Try to jailbreak GPT-4o-mini (cheapest), Claude-Haiku, and Llama-3-70B-Instruct. Use only direct prompts (no agentic loop). Report ASR. You'll see: most models now refuse 100% of direct attacks. This is the floor.

### Exercise 2 — PAIR replication
Implement PAIR (or use the repo). Pick 10 HarmBench behaviors. Run PAIR with attacker = GPT-4o-mini, target = GPT-4o. Report ASR, average queries-to-success, and the final prompt. Note where PAIR fails (these are the most informative failures).

### Exercise 3 — RA replication
Implement RA (4-stage, fabricated prior turn). Use a "mildly harmful" prior turn — write your own. Test on the same 10 behaviors. Compare ASR and query cost vs. PAIR. Observe: RA often wins on cost, PAIR often wins on ASR.

### Exercise 4 — RA × defense
Take your best RA prompts. Run them through:
- Llama-Guard-3 input filter
- A perplexity filter (compute `ppl(M_target, prompt)`, threshold at the 99th percentile of benign prompts)
- SmoothLLM with N=10 paraphrases

Report which defenses catch RA, which don't. This is the table that matters for the realistic threat model.

### Exercise 5 — multi-model transfer
Take 5 RA prompts that worked on GPT-4o. Run them unmodified on Claude-Sonnet, Gemini-2.0-Flash, Llama-3-70B. Report transfer ASR.

### Exercise 6 — judge evaluation
You are suspicious of judge-LLM ASR. Take 30 outputs from your attacks. Score them with: (a) GPT-4o as judge, (b) Llama-Guard-3, (c) your own judgment. Report agreement (Cohen's κ). You will see the judge is itself part of the experimental setup.

### Exercise 7 — closed-form analysis
Pick a small open-weight model (Llama-3-8B-Instruct). Compute the gradient of the loss `-log p(y_target | x + s)` with respect to a one-token perturbation of `s`, for a few RA-style prompts. Verify the gradient structure. This is the foundation for moving from empirical to theoretical.

---

## 16. Glossary of symbols

- **ASR** — Attack Success Rate. The headline metric.
- **DRI / SRI** — Direct / Strong Refusal Inversion (HarmBench definitions).
- **GCG** — Greedy Coordinate Gradient (Zou et al. 2023).
- **PAIR / TAP** — Prompt Automatic Iterative Refinement / Tree of Attacks with Pruning.
- **RA** — Response Attack (our baseline).
- **RLHF / DPO / RLAIF** — Reinforcement Learning from Human Feedback / Direct Preference Optimization / RL from AI Feedback.
- **Crescendo** — Microsoft's multi-turn escalation attack.
- **SmoothLLM / RPO / CB / EC** — SmoothLLM / Robust Prompt Optimization / Circuit Breakers / Erase-and-Check (defenses).
- **LMSYS / AdvBench / JBB** — lmsys.org / AdvBench / JailbreakBench (benchmarks).
- **α-divergence** — Amari's family of f-divergences; includes KL, Hellinger, χ² as special cases. (Used in the Generalized Power Priors paper we'll use in the next brainstorm.)
- **Power prior** — Bayesian prior that weights historical-data likelihood by a power parameter `a ∈ [0,1]`.

---

## What's next

The two brainstorm files (`llm_jailbreak_brainstorm.md` and `brainstorm2_power_priors_and_more.md`) build on top of this material. The beginner walkthrough (`brainstorm_walkthrough_for_beginners.md`) is your dictionary for those — go there when you hit an unfamiliar term.

If a section above was too shallow, the right next step is the original paper. Every concept has a primary source. Don't trust my paraphrase — go to the paper.
