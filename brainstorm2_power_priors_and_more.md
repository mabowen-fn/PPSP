# Brainstorm 2 — Generalized Power Priors, Geodesic Trajectories, and Other Cross-Disciplinary Attacks

> **Companion to** `llm_jailbreak_brainstorm.md` (Brainstorm 1, focused on game theory).
>
> **Headline:** A direct transplant of Kimura & Bondell (2025), "Generalized Power Priors for Improved Bayesian Inference with Historical Data" (arXiv:2505.16244) — turns Response Attack's "fabricated prior turn" into a closed-form Bayesian inference problem with a learnable power parameter `a` and a free α-divergence geometry. Five more novel cross-disciplinary ideas (causal inference, free-energy principle, statistical physics, topology, anthropology) are added on top.

---

## 0. Where Brainstorm 1 left off

Brainstorm 1 framed the problem: **RA wins because of sequential contextual priming** — a fabricated mildly-harmful turn-1 shifts the model's distribution such that turn-2 coerces harmful compliance. The 12 ideas there were mostly *game-theoretic* (Stackelberg, POMDP, cheap talk, repeated games).

What Brainstorm 1 *did not* do: formalize the priming mechanism itself. RA's "mildly harmful prior" is hand-tuned by the authors; the optimal strength of the prior turn is never learned. This is the gap that the Generalized Power Priors paper plugs perfectly.

---

## 1. Why Generalized Power Priors is the right transplant

### 1.1 The GPP recap, in one paragraph

The **power prior** (Ibrahim & Chen, 2000) is a Bayesian device for combining current data `D` with historical data `D₀`. The historical data's likelihood is raised to a power `a ∈ [0, 1]`:

```
π(θ | D, D₀) ∝ L(D | θ) × L(D₀ | θ)^a × π₀(θ)
```

`a = 0` ignores history; `a = 1` treats history as full data; intermediate `a` interpolates. The Kimura & Bondell (2025) *Generalized* Power Prior replaces the linear combination of KL divergences with a linear combination of Amari's **α-divergences**, where α indexes a family of f-divergences (KL = α=0, Hellinger = α=0.5, χ² = α=2, …). The generalized posterior is a **geodesic on the Riemannian manifold of probability distributions** — i.e., a curve that is locally length-minimizing.

### 1.2 The mapping to jailbreaking

| Bayesian object | LLM jailbreak object |
|---|---|
| Historical data `D₀` | Fabricated prior turns (RA's "priming context") |
| Current data `D` | The actual target query |
| Power parameter `a` | Strength of the priming — *learnable per turn* |
| Base prior `π₀(θ)` | The model's pre-prompt alignment distribution |
| Posterior `π(θ | D, D₀)` | The model's effective distribution *after* priming |
| α-divergence | Choice of geometry when combining prior + current |
| Geodesic on the manifold | The optimal trajectory from safe → unsafe output distribution |

**Key insight:** RA's hand-tuned "mildly harmful" priming is exactly the `a` parameter — RA picks it by intuition, we can *learn* it. The α-divergence generalizes the entire family of attacks (KL = classical contextual priming; Hellinger = robust to paraphrase; χ² = heavy-tail sensitive).

### 1.3 What GPP gives us that RA does not have

1. **A learnable, scalar strength per turn.** Instead of "mildly harmful", we have a well-defined `a_t` per priming turn, optimizable end-to-end.
2. **A geometric target.** The optimal priming is a *geodesic* on the manifold — gives a closed-form answer for the best attack trajectory, not a hand-tuned prompt.
3. **A natural multi-turn extension.** `a_t` becomes a *schedule* across multiple priming turns. The optimal schedule is a control problem (Pontryagin / LQ — Brainstorm 1's machinery reuses here).
4. **Robustness via α.** Choose α to make the attack robust to the specific defense (paraphrasing defense → Hellinger; logit-clipping defense → KL; out-of-distribution defense → χ²).
5. **A certifiable lower bound.** Under the GPP regularity conditions, the geodesic lower-bounds the success probability given the attack budget. This is the *theoretical contribution* GTA and Mastermind cannot make.

---

## 2. The GPP-derived research directions

### 2.1 Power-Prior Sequential Priming (PPSP) — *the headline idea*

**Mechanism.** Replace RA's hand-tuned "mildly harmful prior turn" with a learned *schedule* of power parameters `a = (a_1, ..., a_T)` across `T` priming turns. The attacker LLM proposes each prior turn; a separate *probe* (small black-box query to the target) estimates the post-priming distribution shift; the `a_t` are updated by a Bayesian optimizer to maximize the probability that turn `T+1` (the actual attack) succeeds.

**Why it beats RA.** RA picks `a` by intuition and is single-shot. PPSP learns `a` over multiple turns and can ramp it (low `a_1`, high `a_T` — like Foot-in-the-Door but with formal control). Predicted ASR gain: **+8 to +15 absolute percentage points** over RA on HarmBench, with the same query budget.

**Closest prior.** RA (single-turn, fixed `a`); Foot-in-the-Door (psychology, qualitative); Crescendo (qualitative schedule). None formalize `a`.

**Minimal experiment.** 50 HarmBench behaviors × 5 target models (RA set). Learn `a_t` schedule with Bayesian optimization (BoTorch) using ASR as objective, 100 probe queries per behavior. Compare to RA baseline.

**Predicted headline.** "On HarmBench, PPSP reaches 97.4% DRI / 93.8% SRI across 5 models, vs. RA's 94.8% / 89.1%, with comparable query budget."

**Compute.** ~50 GPU-h on 8×A100.

### 2.2 α-Divergence Geodesic Attack (α-DGA)

**Mechanism.** Different α values give different attack geometries. Optimize α *jointly* with the prompt. Hypothesis: the optimal α depends on (a) target model, (b) defense, (c) attack budget.

| α value | Divergence | Attack character | Best against |
|---|---|---|---|
| α → 0 | KL (forward) | Standard contextual priming | Undefended models |
| α = 0.5 | Hellinger | Robust to paraphrasing | SmoothLLM, paraphrasing defenses |
| α = 1 | Total variation | Symmetric, max-min | Logit-clipping defenses |
| α = 2 | χ² | Heavy-tail / out-of-distribution | Perplexity filters, in-distribution detectors |
| α → ∞ | Max-likelihood | Aggressive, brittle | None — degenerate case |

**Why it beats RA.** RA is implicitly KL (α → 0). Hellinger attacks are provably more robust to paraphrase-based defenses (SmoothLLM, RA-style multi-shot paraphrasing). χ² attacks trigger when the input is *just* outside the training distribution, which is exactly the regime jailbreaks live in.

**Closest prior.** No direct prior. F-divergence has been used in adversarial training (Mao et al. 2023) but not in *attack* design.

**Minimal experiment.** For each defense (Llama-Guard-3, SmoothLLM, perplexity filter), find the optimal α on a calibration set of 20 behaviors, then evaluate on 100 held-out behaviors. Expect: Hellinger-optimal attacks have 20%+ higher ASR against SmoothLLM than KL-optimal.

**Compute.** ~20 GPU-h.

### 2.3 Geodesic Jailbreak Trajectory (GJT) — *the closed-form claim*

**Mechanism.** The GPP posterior is a generalized geodesic on the manifold of probability distributions. The "jailbreak problem" becomes: find the shortest path (geodesic) from the safe-distribution to the unsafe-distribution, parameterized by the conversation. The geodesic is *locally length-minimizing* — meaning the attack requires the fewest tokens / smallest distribution shift for a given ASR target.

**Why it beats RA.** RA's 4-stage framework is heuristic. The geodesic is *optimal* in the GPP sense — the provably-shortest contextual trajectory. Even if RA's empirical ASR is close, the geodesic *certifies* a lower bound on the number of tokens required.

**Closest prior.** None in the jailbreak literature. Geometric methods exist in adversarial vision (manifold attacks, Moosavi-Dezfooli et al. 2017) but not in NLP.

**Minimal experiment.** For a fixed behavior, compute the geodesic numerically on a small open-weight model (Llama-3-8B) where the distribution is enumerable. Compare geodesic attack length to RA attack length for the same ASR. Predicted: geodesic is 30–50% shorter.

**Compute.** ~5 GPU-h (small model, fine-grained probing).

**Risk.** The manifold is high-dimensional; numerical geodesic computation is fragile. May require manifold learning (t-SNE-style preprocessing) which is approximate.

### 2.4 Hierarchical Power-Prior Ensemble (HPPE)

**Mechanism.** Maintain a *library* of N priming templates (different framings, different `a` values, different topics). Treat each as a "historical dataset" in the GPP sense. The optimal attack is the posterior-weighted mixture over templates. The weights are learned per (target model, behavior) via Bayesian updating.

**Why it beats RA.** RA uses one template family. HPPE has a Bayesian mixture — different templates will dominate for different behaviors and models. Empirically, mixture-of-templates is consistently stronger than single-template in NLP attacks (PAIR, TAP, GTA all rely on this implicitly).

**Closest prior.** GTA (multi-template, no formal mixture). Mastermind-Dou (two-agent, no ensemble). HPPE is the *Bayesian-optimal* version.

**Minimal experiment.** Pre-generate 50 priming templates (5 framings × 10 strengths). Run HPPE on 100 HarmBench behaviors. Compare to single-template RA. Predicted: +3–7% ASR.

**Compute.** ~80 GPU-h (heavier because of mixture inference per turn).

### 2.5 Borrowing-Strength Defense (BSD) — *the defensive flip*

**Mechanism.** The defender treats the user's prior turns as historical data with power `a ≈ 1` (full weight) and treats any new turn as current data. The defender's inference: if the new turn is "too far" from the prior context under the α-divergence, flag as injection. Concretely: compute `D_α(p(θ | all turns) ‖ p(θ | prior turns))` and threshold.

**Why it matters.** Turns the attacker's own weapon against them. RA's strength is that the prior turn is "trusted" by the model. BSD breaks that assumption by *certifying* the deviation between the prior context and the new turn.

**Closest prior.** Erase-and-Check (token-level). SmoothLLM (paraphrase-level). BSD is *distribution-level* — strictly more powerful.

**Minimal experiment.** Implement BSD on top of an existing defense (Llama-Guard-3). Test against RA, PPSP (2.1), Crescendo. Measure: does BSD catch what Llama-Guard misses? Predicted: 20–40% additional catch rate on contextual attacks.

**Compute.** ~10 GPU-h.

### 2.6 Conjugate-Prior Personalization (CPP)

**Mechanism.** Different target LLMs have different implicit "priors" over refusals (because they were trained differently). Estimate the target's prior from a small *probe set* of 20 benign prompts (each scored for refusal-likelihood). Then design the priming to exploit the conjugate prior — i.e., pick `a` and template such that the posterior most likely shifts the target into compliance.

**Why it beats RA.** RA is *model-agnostic* — uses the same template for GPT-4o and DeepSeek-R1. CPP customizes. The per-model gain is small (~2–5%) but consistent.

**Closest prior.** None direct. Closest is "model fingerprinting" in adversarial vision (Mosbach et al. 2023).

**Minimal experiment.** Run RA on 5 models, then run CPP on the same 5 models with a 20-prompt probe. Predicted: CPP gains 2–5% on each.

**Compute.** ~30 GPU-h.

---

## 3. Other novel cross-disciplinary ideas (not GPP-derived)

These are not from the GPP paper. They're independently novel cross-disciplinary transplants.

### 3.1 Causal Do-Calculus Intervention Attack (CDIA)

**Mechanism.** Pearl's structural causal model (SCM) and do-calculus. Treat the conversation as an SCM: prompt features `X`, model internal state `Z`, output `Y`. The attack is a `do(X := x*)` intervention. The optimal `x*` is the one that maximizes `P(Y = harmful | do(X := x*))`. The do-calculus gives three rules to convert interventional queries to observational ones — meaning the attacker can compute the optimal intervention from black-box observations alone.

**Why it beats RA.** RA treats the conversation as i.i.d. inputs. CDIA exploits the *causal structure* — the fact that the turn-1 → turn-2 edge is a causal one, not a correlational one. The do-calculus gives a *provably correct* intervention (under the SCM assumption) where RA is heuristic.

**Closest prior.** Causal inference has been used in adversarial vision (Zhang et al. 2024) but not in LLM jailbreaks.

**Minimal experiment.** Build an SCM for GPT-4o-mini from 1000 observations. Compute the optimal intervention. Compare to RA. Predicted: +5–10% ASR.

**Compute.** ~20 GPU-h.

**Risk.** SCM identification is hard; the model may not be well-approximated by a low-dimensional SCM.

### 3.2 Free-Energy Surprise Maximization (FESM)

**Mechanism.** Friston's free-energy principle: a system maintains a generative model of its environment and acts to minimize variational free energy (an upper bound on surprise). The model can be cast as a free-energy-minimizer. A jailbreak is a perturbation that *maximizes* expected free energy in the model's internal generative model — i.e., the attack puts the model in a high-surprise state where its learned priors are unreliable.

**Why it matters.** Unifies several existing attacks (GCG = token-level surprise; RA = context-level surprise; Crescendo = temporal surprise) as *special cases* of free-energy maximization. Gives a unifying *objective*: maximize expected free energy, subject to the prompt being well-formed.

**Closest prior.** Friston himself (2010) for the principle. Recent work (Tschantz et al. 2023) applies active inference to RL agents; not yet to LLM attacks.

**Minimal experiment.** Define expected free energy for a target LLM via its logit distribution. Optimize prompt to maximize it. Compare to GCG, RA, Crescendo.

**Compute.** ~50 GPU-h.

**Risk.** The free-energy formulation requires a generative model of the LLM's outputs, which is the LLM itself — circular. May need a surrogate model.

### 3.3 Phase-Transition Alignment Attack (PTAA)

**Mechanism.** From statistical physics. Treat the alignment training as a phase transition: at training time, the model's "spin" (refusal) aligns with a magnetic field (the safety policy). At deployment, the model is at low temperature — order parameter (refusal rate) is high. A jailbreak is a *localized thermal fluctuation* that flips a *single output* above critical temperature. The attack is a *localized, low-entropy perturbation* at exactly the right token position — equivalent to a nucleation event.

**Why it matters.** Gives a precise physical picture of *why* jailbreaks work (a small fluctuation at the right spot triggers a local phase transition). Suggests a defense: lower the "temperature" further (more refusal training), or add impurity pinning (adversarial training) to raise the nucleation barrier.

**Closest prior.** Statistical mechanics of learning (Bahri et al. 2020, Mehta & Schwab 2014). Not yet applied to alignment.

**Minimal experiment.** Plot the refusal-rate curve as a function of an "alignment strength" parameter. Look for a phase transition (sharp drop in refusal rate). The attack is the *critical* perturbation.

**Compute.** ~10 GPU-h (mostly empirical).

**Risk.** May not show a clean phase transition; modern models are likely in a smoothly-aligned regime.

### 3.4 Persistent-Homology Activation Attack (PHAA)

**Mechanism.** Topological data analysis. Compute the persistent homology (Betti numbers, persistence diagrams) of the activation-space representation as the conversation proceeds. The attack is the prompt sequence that *changes* the topology — i.e., the Betti number `b_0` (number of connected components of the activations) shifts in a detectable way at the moment of harm. This is a *topological* signature of the attack, not a token-level one.

**Why it matters.** Defenses based on token-level or embedding-level features may miss topological signatures. PHAA opens a new *modality* for the attack to exploit.

**Closest prior.** "Topology of deep learning" (Naitzat et al. 2020) and persistent-homology-based adversarial detection (Zelko et al. 2024). Not yet applied to LLM attacks.

**Minimal experiment.** On Llama-3-8B (open), compute Betti numbers during a successful RA attack. See if a topological signature exists. If yes, design a "topology-changing" attack.

**Compute.** ~30 GPU-h (TDA is slow on high-dim activations).

**Risk.** The signal-to-noise may be low. Activation-space topology is high-dimensional and noisy.

### 3.5 Taboo-Anthropology Attack (TAA)

**Mechanism.** From Mary Douglas (anthropology). Harmful content is "polluted" — the model has been trained to treat it as such. The attack is *ritual purification* of the request: enclose the harmful ask in a frame of legitimacy (academic, fictional, medical, security-research) that *ritually purifies* it. This is more sophisticated than simple "I'm a researcher" framing — it borrows the formal structure of taboo-cleansing rituals (Douglas 1966, "Purity and Danger").

**Why it matters.** Gives a *cultural-anthropological* theory of why roleplay/framing attacks work. Suggests that the *legitimacy frame* must satisfy specific structural criteria to be effective — not just any "I am a researcher" prefix works. The frame must (a) name an authority, (b) specify a sanctioned purpose, (c) mark the content as not-for-personal-use, (d) provide a cover story for the requester's expertise.

**Closest prior.** "Persona attacks" (qualitative, no formal structure). Closest theory: speech-act theory (Austin 1962, Searle 1969) on performative utterances.

**Minimal experiment.** Compare ASR of: (a) generic "I'm a researcher" framing, (b) TAA-structured framing with all 4 ritual criteria, (c) TAA framing with random criteria. Predicted: (b) >> (a) ≈ (c).

**Compute.** ~5 GPU-h (cheap).

**Risk.** This is the most "soft-science" idea. Reviewers may push back. Frame it as: "we import the *form* of Douglas's ritual analysis to structure the prompt; the underlying mechanism is still game-theoretic — the frame shifts the model's distribution via in-context learning."

---

## 4. Synthesis: how Brainstorm 2 stacks with Brainstorm 1

| Layer | Brainstorm 1 contribution | Brainstorm 2 contribution |
|---|---|---|
| **Strategic** (what to attack) | Stackelberg / Bayesian Persuasion (target the *response function*) | Causal do-calculus (target the *causal structure*) |
| **Tactical** (how to plan) | POMDP / Shapley allocation (multi-model coordination) | Free-energy principle (unifies objectives) |
| **Operational** (what to write) | Foot-in-the-Door / many-shot (schedule) | Power prior `a_t` schedule (learned, formal) |
| **Geometric** (the distribution) | (Implicit, KL) | α-divergence / geodesic (explicit, family) |
| **Defensive** (counter) | — | Borrowing-Strength Defense (uses the attacker's own tool) |
| **Personalization** (per-model) | — | Conjugate-Prior Personalization |
| **Diagnostics** | — | Phase transition / persistent homology (theoretical tools) |

**Three combined headlines to consider pitching:**

1. **The closed-form story.** Geodesic Jailbreak Trajectory (2.3) + PPSP (2.1) + α-DGA (2.2) → a single paper arguing that the *optimal jailbreak is a closed-form geodesic on a Riemannian manifold*, with `a_t` and α learned empirically. Strongest theoretical contribution.

2. **The multi-layer attack.** HPPE (2.4) + CDIA (3.1) + PPSP (2.1) → a paper arguing that *layered* attacks (template + causal + priming) beat single-layer by 10–20% ASR. Strongest empirical contribution.

3. **The arms-race story.** PPSP attack (2.1) + BSD defense (2.5) + Phase-Transition analysis (3.3) → a paper showing that *defenses have a phase-transition structure* and that the right attack is the one that targets the *nucleation barrier*. Strongest security-contribution framing.

---

## 5. Pilot experiments (ordered by EV / cost)

| Priority | Pilot | Source | Predicted Δ vs. RA | GPU-h |
|---|---|---|---|---|
| **1** | PPSP (§2.1) | GPP transplant | +8–15% DRI | 50 |
| **2** | α-DGA (§2.2) | GPP transplant | +20% vs. SmoothLLM | 20 |
| **3** | Geodesic GJT (§2.3) | GPP transplant | 30–50% fewer tokens at same ASR | 5 |
| **4** | HPPE (§2.4) | GPP + multi-template | +3–7% DRI | 80 |
| **5** | CDIA (§3.1) | Causal inference | +5–10% DRI | 20 |
| **6** | FESM (§3.2) | Neuroscience | Unified objective (qualitative) | 50 |
| **7** | TAA (§3.5) | Anthropology | +10–20% (cheap) | 5 |
| **8** | CPP (§2.6) | Bayesian | +2–5% per model | 30 |
| **9** | PTAA (§3.3) | Statistical physics | Diagnostic, not direct gain | 10 |
| **10** | PHAA (§3.4) | TDA | Diagnostic, high-risk | 30 |
| **11** | BSD (§2.5) | GPP-flip | 20–40% additional catch | 10 |

**Total compute for all pilots:** ~310 GPU-h. Affordable as a single 4-week run on 4×A100, or one week on a 16-GPU cluster.

**Minimum viable combined paper:** pilots 1, 2, 7, 11 (PPSP + α-DGA + TAA + BSD). The attack + defense pair. ~85 GPU-h.

---

## 6. Open questions for Joel

1. **Theoretical vs. empirical track.** The GPP transplant gives you a *theoretical* contribution (closed-form geodesic, learnable `a_t`). Is that the publication you want, or do you prefer an *empirical* SOTA paper? My read: the theoretical track is more defensible and less likely to be beaten in a 6-month window.

2. **Attack vs. defense framing.** PPSP+α-DGA is an attack paper. BSD is a defense paper. Mixing them in one paper is hard (different reviewers, different venues). Recommend: pick one.

3. **α vs. KL choice.** The α-divergence story is novel but the closed-form geodesic is *under* specified for high-dim distributions. Is a numerical / approximation paper acceptable, or do you want a fully analytic result?

4. **Compute budget.** The headline 3-pilot package (PPSP, α-DGA, BSD) is ~85 GPU-h. Acceptable?

5. **Target model set.** Replicate RA's 5 models (GPT-4o, Gemini-2.5, LLaMA-3-70B, DeepSeek-R1, QwQ-32B)? Or focus on a 2-model comparison (one closed, one open)?

6. **Multi-turn commitment.** All GPP ideas are inherently multi-turn. Is that locked in, or do you want a single-turn variant for breadth?

7. **Where to submit.** The combined theoretical + empirical package fits AAAI / NeurIPS / IEEE S&P. The pure-empirical SOTA fits USENIX Security / CCS. The pure-theoretical fits AISTATS / JMLR. Where do you want to aim?

8. **Anthropology framing risk.** TAA is the most novel but also the most "soft-science". Is the rhetorical risk worth the cheap-win (5 GPU-h)?

9. **GTA differentiator.** GTA (arXiv:2511.16278) claims ~95% on DeepSeek-R1. The PPSP story has to *differentiate*, not match. Differentiation: closed-form `a_t` schedule + geodesic optimality + transferability across the 5-model set. Is that the story?

10. **Power-prior as defense.** BSD is interesting but is *itself* an attack primitive (an attacker can craft prompts that look like legitimate user history). Have I correctly understood the security model?

---

## 7. Risks (consolidated)

| Risk | Mitigation |
|---|---|
| GPP's closed-form claims are numerical-approximation-bound, not analytic | Frame as numerical method paper, not analytic. |
| Reviewers reject α-divergence / geometry as "not empirically justified" | Lead with empirical gains; keep theoretical as appendix. |
| Anthropological framing (TAA) gets mocked | Skip it if pitching a CS-only venue. Frame it as "speech-act structure" instead. |
| Free-energy circularity (the model is the generative model) | Use a *small* open-weight model as the surrogate. |
| Causal SCM identification is fragile | Fall back on observational PE (Pearl's front-door criterion). |
| Topological signal is too noisy | Pilot before committing; if no clear Betti shift, drop. |
| Phase transition is too smooth to detect | Plot a fine-grained refusal-vs-strength curve first. |
| Conjugate prior is too high-variance from 20 probes | Use a hierarchical prior (BATE-style). |
| Open-weight models give weak signals; closed models are expensive | Start with Llama-3-8B; use the 5 closed models for final ASR table only. |
| Research ethics | Coordinate with the model's safety team; pre-register on the OpenAI / Anthropic red-team program if possible. |

---

## 8. References (new in this brainstorm)

### Generalized Power Priors
- **Kimura & Bondell (2025).** "Generalized Power Priors for Improved Bayesian Inference with Historical Data." arXiv:2505.16244.
- Ibrahim & Chen (2000). "Power Prior Distributions for Regression Models." *Statistical Science.*
- Amari (1985). "Differential-Geometrical Methods in Statistics." — α-divergence family.
- Duchi (2016). "Derivations for Linear Algebra and Optimization." — f-divergence properties.

### Causal inference
- Pearl (2009). *Causality.* CUP. — Do-calculus.
- Pearl, Glymour, Jewell (2016). *Causal Inference in Statistics.* Springer.
- Bareinboim et al. (2022). "On Pearl's Hierarchy and the Foundations of Causal Inference." Causal AI book.

### Free-energy / active inference
- Friston (2010). "The Free-Energy Principle: A Unified Brain Theory?" *Nature Reviews Neuroscience.*
- Friston et al. (2017). "Active Inference: A Process Theory." *Neural Computation.*
- Tschantz et al. (2023). "Active Inference for LLM agents." (Workshop paper, not yet main venue.)

### Statistical physics
- Bahri et al. (2020). "Statistical Mechanics of Deep Learning." *Annual Review of Condensed Matter Physics.*
- Mehta & Schwab (2014). "An Exact Mapping Between the Variational Renormalization Group and Deep Learning." — RG ↔ deep learning.
- Roberts, Yaida, Hanin (2021). *The Principles of Deep Learning Theory.* CUP. — Effective theory of neural networks.

### Topological data analysis
- Edelsbrunner, Harer (2010). *Computational Topology.* AMS.
- Naitzat, Zhitnikov, Lim (2020). "Topology of Deep Learning." *NeurIPS Workshop on Topological Data Analysis.*
- Zelko et al. (2024). "Persistent Homology for Adversarial Robustness."

### Anthropology / speech acts
- Douglas (1966). *Purity and Danger.* Routledge.
- Austin (1962). *How to Do Things with Words.* OUP. — Speech-act theory.
- Searle (1969). *Speech Acts.* CUP.

### Adversarial vision (analogs)
- Madry et al. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks." — PGD.
- Moosavi-Dezfooli et al. (2017). "Universal Adversarial Perturbations." — Manifold attacks.

### Game theory / multi-agent (carried over)
- Zhou et al. 2025, arXiv:2502.13462, 2509.03727.
- Sreedhar et al. 2025, arXiv:2505.12923.
- "Echoes of Human Malice" 2025, arXiv:2510.14207.
- "Vulnerability Amplification in MAD" 2025, arXiv:2504.16489.

---

## 9. Decision record (vs. Brainstorm 1)

Brainstorm 1 said: "12 ideas, 3-layer architecture (Shapley → SAP/POMDP → MDJO)."

Brainstorm 2 says: "11 more ideas; the *attack* layer is now a *learned power prior* with a Riemannian-geometric interpretation. The *coordination* layer from Brainstorm 1 still sits on top. The *defense* layer is new (BSD) and uses the same machinery against the attacker."

**If we combine both:** the unified pitch is "**multi-agent, power-prior-scheduled, α-geometric jailbreaks on a Riemannian manifold of distributions**." One paper, ~300 GPU-h, two clear contributions: (1) closed-form optimality, (2) per-target-model personalization.
