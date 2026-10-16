# Beginner Walkthrough — Understanding Both Brainstorm Files

> **Who this is for.** You. The two brainstorm files (`llm_jailbreak_brainstorm.md` and `brainstorm2_power_priors_and_more.md`) are research notes — they assume you know the vocabulary. This document is the *decoder ring*. It explains every concept that appears in either brainstorm, in plain language, with examples.
>
> **How to use this document.** Don't read it linearly. Use Ctrl+F. Find the term you don't understand, read that section, jump back to the brainstorm. By the end you should be able to read both files without a dictionary.
>
> **What I'll do here:** for every concept, I'll give (a) a one-sentence definition, (b) a 2–3 sentence plain-language explanation, (c) a tiny example, and (d) "why it matters for jailbreaking" in one line. Then at the end I'll do a guided tour of one idea from each brainstorm so you can see how the pieces fit together.

---

# PART A — Foundations you need

These are the cross-cutting concepts that appear in *both* brainstorms. If you only read one section, read this one.

## A.1 Bayes' theorem (the basis of Brainstorm 2)

**One-sentence:** Posterior ∝ Prior × Likelihood.

**Plain language.** You have a belief about the world. You see new evidence. You update your belief. The "prior" is what you believed before. The "likelihood" is how probable the evidence is, given each possible belief. The "posterior" is your new belief.

**Example.** You flip a coin 3 times, all heads. You start with a prior that the coin is fair (50/50). The likelihood of "3 heads" is 50% if fair, 75% if biased-heads, 25% if biased-tails. Your posterior — your new belief — shifts toward "biased-heads." (The exact shift depends on the prior strength, which is the "power parameter" — see A.2.)

**Why it matters for jailbreaking.** The "world" is the model's hidden state — its safety policy, its refusal threshold. The "evidence" is the conversation context. The attack shifts the posterior toward "comply" by choosing evidence (the prior turns) that is more likely under a compliant world than a refusing one.

## A.2 The power parameter `a`

**One-sentence:** A knob that controls how much you "trust" or "borrow" from historical data when forming a posterior.

**Plain language.** In standard Bayes, you fully trust your historical data. With a *power prior*, you can choose to trust it partially. `a = 1` means full trust. `a = 0` means ignore the historical data entirely. `a = 0.5` means "use the historical data, but treat it as half a data point." This is useful when your historical data might be from a *different* distribution than your current data.

**Example.** You're a hospital that ran a drug trial in 2010 (your "historical data"). You're running the same trial in 2025. The patients are slightly different. Setting `a = 0.7` means "use the 2010 data, but discount it slightly because the populations aren't identical."

**Why it matters for jailbreaking.** In RA (Response Attack), the attacker fabricates a "prior turn" where the model supposedly gave a mildly harmful response. The strength of that priming is what `a` controls. RA's authors tune it by hand. The Generalized Power Priors paper says: *learn* `a` end-to-end. This is the formalization Brainstorm 2 builds on.

## A.3 α-divergence (Amari's family of divergences)

**One-sentence:** A family of ways to measure "how different" two probability distributions are, parameterized by a single number `α`.

**Plain language.** KL divergence (Kullback-Leibler) is the most famous way to compare two probability distributions. It has a bias: it cares more about events the *first* distribution thinks are likely. Amari's α-divergence is a whole family. Different `α` values give different "shapes" of comparison:

- `α = 0`: KL divergence. The standard. Asymmetric.
- `α = 0.5`: Hellinger distance. Symmetric, robust to outliers.
- `α = 1`: Total variation. Symmetric, simple.
- `α = 2`: chi-squared (χ²). Sensitive to events the *first* distribution thinks are *unlikely*. Heavy-tailed.

**Example.** You want to compare two dice. Die A is fair; die B has a 50% chance of rolling a 6. KL says "very different" (B concentrates mass on 6). χ² says "very very different" (it penalizes the cases where A would *not* roll 6, but B *does*). Hellinger says "moderately different" (it's the geometric average).

**Why it matters for jailbreaking.** Different defenses catch different "shapes" of attack. A defense that uses KL will be caught by an Hellinger-based attack. A defense that uses χ² will be caught by a KL-based attack. The Generalized Power Priors paper says: optimize `α` *jointly* with the prompt, so the attack is matched to the defense's blind spot.

## A.4 Geodesic and Riemannian manifold

**One-sentence:** A geodesic is the "shortest path" between two points on a curved surface; a Riemannian manifold is the curved surface itself.

**Plain language.** Imagine a flat map of the Earth. The shortest path from NYC to Tokyo is a straight line. But the Earth is a sphere — the actual shortest path is a *great circle* (a curve). That curve is a geodesic. The sphere is a Riemannian manifold. The math of Riemannian manifolds generalizes "shortest path" to curved spaces.

**Example.** A simple manifold: the surface of a sphere. Two points on the sphere, connected by a great circle. That's a geodesic.

For probability distributions: the set of all probability distributions is a manifold. A path through this manifold is a sequence of distributions. The geodesic is the shortest path — the one that requires the least "energy" to traverse.

**Why it matters for jailbreaking.** The model's "state" is a probability distribution over tokens. The "safe" state is one distribution; the "unsafe" state is another. The jailbreak is a path from safe to unsafe. The geodesic is the *shortest*, most efficient path — the optimal attack. The Generalized Power Priors paper says the optimal attack trajectory is a geodesic, and it has a closed-form solution in terms of `α` and the divergence geometry.

## A.5 Stackelberg game (the basis of Brainstorm 1's headline)

**One-sentence:** A game where one player (the leader) commits to a strategy first, and the other player (the follower) best-responds knowing the leader's commitment.

**Plain language.** In chess, if you announce your opening move before your opponent moves, you're playing Stackelberg. The leader's optimal commitment is the move that *forces* the follower into a bad position given that the follower will play optimally. This is different from Nash equilibrium, where both players move simultaneously and we predict a stable pair of strategies.

**Example.** A monopolist sets a price; consumers decide how much to buy knowing the price. The monopolist's optimal price is the one that maximizes profit given that consumers will buy the optimal quantity.

**Why it matters for jailbreaking.** The attacker can be the *leader* — committing to a first turn — and the model is the *follower*, responding to the second turn. The attacker's optimal first turn is the one that, given the model will respond optimally (in its own distribution), maximizes the chance of harmful output. This is Brainstorm 1's "Stackelberg Attack Plan" (SAP).

## A.6 POMDP (Partially Observable Markov Decision Process)

**One-sentence:** A framework for decision-making when you can't see the full state of the world, only noisy observations.

**Plain language.** A robot in a maze. It can see what's directly in front of it but not the whole maze. It has a *belief* (a probability distribution over where it is). It picks an action based on the belief. The world changes according to some probabilistic rule. The robot gets a reward. POMDP is the math for picking the action that maximizes expected reward.

**Example.** A poker player doesn't know the opponents' cards. They have a belief (probability distribution) over what the opponents hold. They choose to bet, call, or fold. The choice depends on the belief, not the actual cards.

**Why it matters for jailbreaking.** The attacker doesn't see the model's internal state. They have a *belief* (e.g., "the model is 60% likely to refuse"). They pick the next prompt based on the belief. POMDP gives the optimal policy for doing this.

## A.7 Shapley value

**One-sentence:** A way to assign credit to each player in a coalition, based on each player's marginal contribution.

**Plain language.** Three friends collaborate on a project worth $100. Alice alone would have made $30. Bob alone would have made $20. Carol alone would have made $10. Alice+Bob would have made $70. Alice+Carol: $50. Bob+Carol: $40. All three: $100. How do you split the $100 fairly? The Shapley value: Alice gets $37.50, Bob gets $37.50, Carol gets $25. (Each player's expected marginal contribution over all orderings.)

**Why it matters for jailbreaking.** If you have a *fleet* of LLMs (e.g., GPT-4o, Gemini, Llama-3) and you want to know which one to use for which sub-task in an attack, the Shapley value tells you the *fair* contribution of each. Brainstorm 1's "Shapley-Allocated Priming Ensemble" (SAPE) uses this to optimally assign which model does which part of the attack.

## A.8 Cheap talk (Crawford-Sobel)

**One-sentence:** Communication between players that doesn't directly cost anything and doesn't bind anyone to action.

**Plain language.** Two friends at a coffee shop, chatting. Words are cheap. They can say anything; nothing forces them to act. Crawford & Sobel (1982) showed that cheap-talk communication can transmit information, but only up to a limit set by the *bias difference* between the speakers.

**Example.** A used car salesman wants to sell a lemon. The buyer wants a good car. The salesman can say "great car!" — but the buyer knows the bias. So the message carries no information. If their biases were aligned, more information could be transmitted.

**Why it matters for jailbreaking.** When the attacker and the model "talk" (the multi-turn conversation), the model's outputs are the only signal. The attacker uses the model's responses to update a belief about whether the model will comply. This is cheap talk under bias — and the Crawford-Sobel limit applies.

## A.9 Mechanism design (Myerson's)

**One-sentence:** The reverse of game theory. Instead of predicting behavior in a given game, *design* the game so that rational players behave the way you want.

**Plain language.** You want people to bid fairly in an auction. You *design* the auction rules (Vickrey = second-price sealed-bid) so that the dominant strategy for each player is to bid their true value. This is *reverse game theory*: design the rules, not predict the behavior.

**Why it matters for jailbreaking.** The attacker can be the *mechanism designer* — designing the prompt template (the "rules of the game") such that the model's rational behavior is to comply. Brainstorm 1's MDJO (Mechanism-Designed Jailbreak Oracles) builds prompts as incentive-compatible mechanisms.

## A.10 Bayesian persuasion (Kamenica-Gentzkow)

**One-sentence:** A sender commits to an information-disclosure policy; a receiver rationally updates their belief; the sender's goal is to make the receiver act in a way that benefits the sender.

**Plain language.** A doctor has a test. The test result (positive/negative) shifts the patient's belief about whether they're sick. The doctor wants the patient to take a treatment. The doctor designs a *signaling policy* — which test results to disclose, in what order — to maximize the chance the patient takes the treatment. The math gives the optimal signaling.

**Why it matters for jailbreaking.** The attacker's "information disclosure" is the prior turn. The model is the receiver. The attacker wants the model to comply. The optimal disclosure is what Bayesian persuasion computes. Brainstorm 1's BPJ uses this directly.

## A.11 LQ control (Linear-Quadratic)

**One-sentence:** A control-theory framework for steering a linear system with a quadratic cost — gives a closed-form optimal controller.

**Plain language.** A self-driving car. Its state is position, velocity, heading. The dynamics are *linear* (small perturbations evolve predictably). The cost is *quadratic* (you want small state errors, small control effort). The optimal control is a simple linear function of the state, computed by solving a Riccati equation. This is the workhorse of classical control theory.

**Why it matters for jailbreaking.** If you can write the model's behavior as a *linear* system and the attack cost as *quadratic*, you get a closed-form optimal attack. Brainstorm 1's "Differential-Game SDE Steering" applies this to the token-distribution dynamics.

## A.12 Pontryagin maximum principle

**One-sentence:** A generalization of LQ control to non-quadratic, nonlinear systems — gives necessary conditions for an optimal control.

**Plain language.** Same setup as LQ, but the dynamics are nonlinear. There is no closed-form solution, but Pontryagin's principle gives you a *boundary-value problem* whose solution is the optimal control. Numerically tractable.

**Why it matters for jailbreaking.** The token distribution is highly nonlinear. Pontryagin gives the math to still compute the optimal priming schedule, even without closed-form.

## A.13 Bayesian optimization

**One-sentence:** A method for optimizing a black-box function that is expensive to evaluate, using a probabilistic surrogate (usually a Gaussian process).

**Plain language.** You want to find the input `x` that maximizes `f(x)`, but each evaluation of `f` costs $1000 and takes an hour. Bayesian optimization: fit a cheap-to-evaluate probabilistic model (a Gaussian process) to the data you've collected. Use the model's mean and uncertainty to pick the next `x` to try — balance "exploring where the model is uncertain" and "exploiting where the model predicts high values." After 50 evaluations, you usually have a good estimate of the optimum.

**Why it matters for jailbreaking.** Learning the optimal power parameter `a` requires probing the model many times. Each probe is a query. Bayesian optimization finds the best `a` with few queries.

## A.14 Phase transition (from statistical physics)

**One-sentence:** A sharp, qualitative change in a system's behavior when a parameter crosses a critical value.

**Plain language.** Water at 99°C is liquid. At 101°C is gas. The transition at 100°C is sharp (a phase transition). In a magnet: above a critical temperature, the magnet's domains point in random directions. Below, they align. The transition is sharp.

**Why it matters for jailbreaking.** Brainstorm 2's PTAA (Phase-Transition Alignment Attack) suggests that the alignment training creates a "magnetic" ordering of refusals, and a small perturbation can flip a single output across the "phase boundary."

## A.15 Do-calculus (Pearl)

**One-sentence:** Three rules for converting interventional queries ("what if I do X?") into observational queries ("what do I see?") using a causal graph.

**Plain language.** You want to know the effect of a new drug. You can't force everyone to take it. But if you know the causal graph (genetics → disease, lifestyle → disease, drug → recovery), you can use do-calculus to estimate the drug's effect from observational data. The three rules are: (1) ignore observations that don't affect the target, (2) substitute `do(x)` for `see(x)` when there are no back-door paths, (3) insert or remove interventions when the graph justifies it.

**Why it matters for jailbreaking.** The attack is a `do()` intervention on the prompt. Do-calculus gives the math to compute the effect of the intervention from black-box observations. Brainstorm 2's CDIA uses this.

## A.16 Free-energy principle (Friston)

**One-sentence:** A theory from neuroscience that biological systems act to minimize a quantity called "variational free energy" — a bound on the surprise of sensory observations, given the system's internal model.

**Plain language.** Your brain has a model of the world. When sensory data surprises you (your model says you should see X, but you see Y), your brain updates the model to reduce future surprise. The "free energy" is a measure of how surprised your model is. Minimizing it = being unsurprised.

**Why it matters for jailbreaking.** Brainstorm 2's FESM treats the LLM as a free-energy-minimizer. The attack maximizes expected free energy — puts the model in a high-surprise state where its safety priors are unreliable.

## A.17 Persistent homology and Betti numbers (from topology)

**One-sentence:** A way to count the "shape" of a high-dimensional point cloud — how many connected components, loops, voids it has, at every scale.

**Plain language.** A cloud of points in 3D. At a large scale (zoomed out), the cloud might look like one blob — one connected component. At a smaller scale (zoomed in), it might split into 5 pieces — 5 components. At an even smaller scale, you might see each component has a hole. Betti numbers count: b₀ = # components, b₁ = # loops, b₂ = # voids.

**Why it matters for jailbreaking.** Brainstorm 2's PHAA suggests that a successful jailbreak changes the *topology* of the model's activation space — not just the token-level distribution. This is a new signal for the attack (and the defense).

## A.18 Speech-act theory (Austin / Searle)

**One-sentence:** Words don't just describe the world; they *do* things. "I pronounce you married" is not a description — it's the act itself.

**Plain language.** Saying "I promise to pay you back" is not describing a promise, it *creates* the promise. This is a "performative utterance" (Austin 1962). Searle (1969) refined this: every speech act has three parts — the *locutionary* (the words), the *illocutionary* (the intent — promising, warning, declaring), and the *perlocutionary* (the effect — making you trust me).

**Why it matters for jailbreaking.** A "I'm a security researcher" framing is a speech act. The framing has *structure*: it names an authority, declares an intent, claims expertise. The attack's effectiveness depends on the *speech-act structure*, not just the words. Brainstorm 2's TAA (Taboo-Anthropology Attack) and Brainstorm 1's "Foot-in-the-Door" both leverage this.

---

# PART B — Walkthrough of Brainstorm 1 (12 ideas, simplified)

**What's Brainstorm 1 about.** 12 ideas for jailbreaking LLMs using game theory and coordinated multi-agent attacks. Each idea is explained below in plain language.

## B.0 The baseline: Response Attack (RA)

**What it is.** RA is a 4-stage process:
1. Generate a "mildly harmful" fabricated prior turn (e.g., the model supposedly said "Here's a partial answer..." for some semi-bad thing).
2. Use that as the first turn of a conversation.
3. The actual attack is the second turn. The model has been "primed" — its distribution has shifted because it "remembers" the prior turn.
4. The model complies.

**Why it works.** LLMs are autoregressive — they condition on prior tokens. A fabricated prior turn shifts the conditional distribution toward compliance. It's the *sequential* structure that matters, not the content.

**Why this is hard to beat.** RA is simple, general, and transfers across models. To beat it, you have to do something *qualitatively different*, not just more of the same.

## B.1 SAP — Stackelberg Attack Plan

**Plain language.** Frame the attack as a leader-follower game. The attacker is the leader. They commit to a first turn (a "policy" of sorts). The model is the follower — it best-responds. The attacker computes the *optimal* first turn, given that the model will best-respond.

**Why it beats RA.** RA picks the first turn by intuition. SAP picks it via formal optimization — the math says "this is the *provably* best first turn given that the model responds optimally."

**Key technical content.** Stackelberg equilibrium computation; reaction-function derivatives (how the model's response changes with the first turn); numerical solution via gradient descent.

## B.2 BPJ — Bayesian Persuasion Jailbreak

**Plain language.** The attack is a *signaling policy*: a rule for what to disclose to the model. The model observes the disclosure, updates its belief about what the attacker wants, and acts. The math (Kamenica-Gentzkow) gives the *optimal* signaling policy.

**Why it beats RA.** RA is one disclosure. BPJ is a *policy* — a function from context to disclosure — that's optimal across many behaviors. It generalizes better.

**Key technical content.** Kamenica-Gentzkow concavification; belief-update rules; signaling-policy optimization.

## B.3 SAPE — Shapley-Allocated Priming Ensemble

**Plain language.** You have multiple "priming templates" (different ways to fabricate a prior turn) and multiple "target models" (GPT-4o, Gemini, etc.). You want to assign the best template to each model. The Shapley value (A.7) tells you the marginal contribution of each (template, model) pair. Use it to optimally allocate.

**Why it beats RA.** RA uses one template for all models. SAPE has a *learned* allocation — it uses the right template for the right model.

**Key technical content.** Shapley value computation (exponential in the number of players, so approximations needed); assignment problem; per-model ASR prediction.

## B.4 MDJO — Mechanism-Designed Jailbreak Oracles

**Plain language.** Design the prompt *as a mechanism* — a set of rules such that the model, acting rationally, will comply. This is reverse game theory: design the rules, predict the behavior.

**Why it beats RA.** RA fabricates a single prior turn. MDJO designs a *whole interaction protocol* such that compliance is the model's rational best response.

**Key technical content.** Myerson's mechanism design; incentive compatibility; revelation principle.

## B.5 RGRJ — Repeated-Game Reputation-Building Jailbreak

**Plain language.** In a repeated game (a game played many times), the attacker can build "reputation" — appearing trustworthy across turns. The model, believing the attacker is trustworthy, complies.

**Why it beats RA.** RA is a single shot. RGRJ uses multiple turns to *build up* the reputation. The model has more information to update on; the attack is more credible.

**Key technical content.** Repeated-game folk theorem; reputation equilibria; discounting of future payoffs.

## B.6 Traitor-Style Deception Game

**Plain language.** From the TV show "The Traitors" (and the research paper by the same name). Multiple agents, some are "traitors" who secretly conspire, others are "faithfuls" who don't know who's who. The model is fooled into thinking the attacker is faithful. The attack is the *coalition* — the conspiracy — that misleads the model.

**Why it beats RA.** RA is a single agent. Traitor-Style uses a *coalition* of attacker agents that the model can't distinguish from innocent ones. The model has incomplete information; the conspiracy exploits this.

**Key technical content.** Bayesian games; coalition formation; incomplete-information equilibria.

## B.7 POMDP belief planner

**Plain language.** The attacker has a *belief* about the model's internal state (e.g., "60% likely to refuse, 40% likely to comply"). The belief is updated based on the model's responses. The next prompt is chosen to *maximize expected compliance* under the belief. This is a POMDP (A.6).

**Why it beats RA.** RA doesn't track the model's internal state. POMDP planners do. The optimal policy is computed by solving the POMDP.

**Key technical content.** POMDP solution (Q-MDP, POMCP, or DESPOT); belief update; value iteration.

## B.8 Differential-Game SDE Steering (LQ control from Zhou et al.)

**Plain language.** The model's distribution evolves as a stochastic differential equation (SDE). The attacker is a controller steering the distribution. The cost is quadratic in (state error, control effort). The optimal control is LQ (A.11) — a linear function of the state, computed from a Riccati equation.

**Why it beats RA.** RA doesn't model the dynamics. The SDE formulation does, and the LQ controller is *provably* optimal.

**Key technical content.** SDE dynamics; LQ control; Riccati equation; Pontryagin (A.12) extensions for non-quadratic costs.

## B.9 PAP-FITD hybrid (Persuasion + Foot-in-the-Door)

**Plain language.** A cheap, easy-to-implement hybrid. PAP (persuasion) gives the framing; FITD (Foot-in-the-Door, a psychology trick) gives the escalation. The model is first asked to comply with a small request, then a larger one. The compliance is gradual.

**Why it beats RA.** Cheaper to run, easier to defend against (less complex), but the *combination* of framing + escalation is empirically effective. Sanity check / positive control.

**Key technical content.** Foot-in-the-Door psychology literature (Freedman & Fraser 1966); cheap persuasive frames.

## B.10 Evolutionary population

**Plain language.** Instead of one attacker prompt, maintain a *population* of attacker prompts. Each generation: probe all of them, keep the best, mutate/crossover to make new ones. Repeat for many generations. This is an evolutionary algorithm.

**Why it beats RA.** RA is one prompt. The population explores the prompt space. Over many generations, it finds better prompts.

**Key technical content.** Genetic algorithms; mutation operators on token sequences; fitness = ASR.

## B.11 CT-BTI — Cheap-Talk Bayesian Type Inference

**Plain language.** The attacker uses the model's responses (cheap talk, A.8) to *infer* what "type" of model they're talking to (more or less safety-trained, more or less sycophantic, etc.). Then they pick the attack that works for that type. The math is Crawford-Sobel (A.8).

**Why it beats RA.** RA treats all models the same. CT-BTI customizes. The Crawford-Sobel bound limits how much can be inferred; we operate at that bound.

**Key technical content.** Type inference; cheap-talk equilibrium; bias difference.

## B.12 Liability-Shifting mechanism design

**Plain language.** Design the prompt so that *if* the model complies, the *responsibility* for the harm shifts to the (fabricated) requester. The model, in some sense, is "off the hook." This is a *liability-allocation* mechanism design.

**Why it beats RA.** RA shifts the *content* of the response. Liability-shifting shifts the *moral framing*. Some models are more likely to comply if they believe they're not personally liable.

**Key technical content.** Liability rules; moral-hazard games; framing effects.

---

# PART C — Walkthrough of Brainstorm 2 (11 ideas, simplified)

**What's Brainstorm 2 about.** 11 ideas built on top of the *Generalized Power Priors* paper (Bayesian statistics) and other cross-disciplinary transplants (causal inference, neuroscience, statistical physics, topology, anthropology). Each idea is explained below.

## C.1 PPSP — Power-Prior Sequential Priming (the headline)

**Plain language.** RA's "mildly harmful prior turn" is a *power prior* with hand-tuned strength `a`. PPSP *learns* the strength per turn. The power parameter `a_t` becomes a *schedule* across multiple priming turns. The schedule is learned via Bayesian optimization (A.13), with the model's compliance probability as the objective.

**Why it beats RA.** RA picks `a` by intuition. PPSP learns it from data. Over 3–5 priming turns, the schedule can ramp from low `a` (innocuous) to high `a` (coercive), with the optimal shape learned per target model.

**Connection to A.2 (power prior).** The paper Generalized Power Priors gives the math for how `a` affects the posterior. PPSP treats the conversation as a sequence of historical-data updates, each with its own `a_t`.

**Connection to A.13 (Bayesian optimization).** Learning the schedule requires probing the model many times. Bayesian optimization finds the best `a_t` with few probes.

## C.2 α-DGA — α-Divergence Geodesic Attack

**Plain language.** Different defenses catch different "shapes" of attack. By choosing the *α-divergence* (A.3) that defines the geometry of the attack, you can target the defense's blind spot.

**Concrete example.** SmoothLLM is a defense that paraphrases the input. It's designed to catch KL-style attacks. If you use Hellinger (α = 0.5) instead, your attack is more robust to paraphrasing. If you use χ² (α = 2), your attack triggers on rare-token deviations — a totally different defense signature.

**Why it beats RA.** RA is implicitly KL. α-DGA optimizes over the family. The optimal `α` depends on the defense.

**Connection to A.3.** This is the direct application of the Generalized Power Priors paper's key insight: the α-divergence generalizes KL, and the optimal one depends on the application.

## C.3 GJT — Geodesic Jailbreak Trajectory

**Plain language.** The "jailbreak problem" is: find the shortest path (geodesic, A.4) from the model's safe distribution to its unsafe distribution. The path is a sequence of distributions, parameterized by the conversation.

**Why it beats RA.** RA is a heuristic trajectory. GJT is the *provably optimal* trajectory — the one that requires the fewest tokens / smallest distribution shift.

**Connection to A.4 (geodesic).** This is the closed-form claim of Brainstorm 2. The math says: the GPP posterior is a generalized geodesic, and the jailbreak is the geodesic from safe → unsafe.

**Risk.** The manifold is high-dimensional; numerical geodesic computation is fragile. May require approximation (manifold learning) which is approximate.

## C.4 HPPE — Hierarchical Power-Prior Ensemble

**Plain language.** Maintain a *library* of N priming templates. Each has its own `a_i` (power). The optimal attack is the Bayesian-mixture of templates — a posterior-weighted sum. The weights are learned per (target model, behavior).

**Why it beats RA.** RA uses one template family. HPPE has a mixture. Mixture-of-templates is consistently stronger in NLP attacks (PAIR, TAP, GTA all use this implicitly).

**Connection to A.1 (Bayes).** This is the hierarchical extension of the GPP idea: each template is a "historical dataset," and the mixture is a posterior-weighted combination.

## C.5 BSD — Borrowing-Strength Defense (the defensive flip)

**Plain language.** Turn the power-prior machinery against the attacker. A defender treats the user's prior turns as historical data with `a ≈ 1` (full weight), and the new turn as current data. Compute the *divergence* between the posterior (all turns) and the prior (just prior turns). If the divergence is too high, flag as injection.

**Why it works.** RA's strength is that the prior turn is "trusted" by the model. BSD breaks that trust by *certifying* the deviation between prior context and the new turn.

**Connection to A.2 + A.3.** This is the GPP machinery used as a defense. The α-divergence gives a flexible detection rule.

## C.6 CPP — Conjugate-Prior Personalization

**Plain language.** Different target LLMs have different implicit "priors" over refusals (because they were trained differently). Estimate the target's prior from a small *probe set* of 20 benign prompts. Then design the priming to exploit the conjugate prior.

**Why it beats RA.** RA is model-agnostic. CPP customizes.

**Connection to A.1 (Bayes).** A *conjugate prior* is a prior that, when combined with the data, gives a posterior in the same family. For example, if the prior is Beta(α, β) and the data is Bernoulli, the posterior is Beta(α + #successes, β + #failures). The "conjugate" trick lets you update the prior analytically.

## C.7 CDIA — Causal Do-Calculus Intervention Attack

**Plain language.** Pearl's do-calculus (A.15). Treat the conversation as a *causal model* — features X cause state Z cause output Y. The attack is a `do(X := x*)` intervention. The do-calculus gives three rules to compute the intervention's effect from black-box observations.

**Why it beats RA.** RA treats the conversation as i.i.d. inputs. CDIA exploits the *causal structure* — the turn-1 → turn-2 edge is causal, not correlational.

**Connection to A.15 (do-calculus).** Direct application.

## C.8 FESM — Free-Energy Surprise Maximization

**Plain language.** Friston's free-energy principle (A.16). The model acts to minimize "surprise." A jailbreak is a perturbation that *maximizes* surprise. The math gives a single unified objective.

**Why it matters.** Several existing attacks (GCG, RA, Crescendo) are special cases of surprise-maximization. The unified objective lets you combine them.

**Connection to A.16 (free-energy).** Direct application.

## C.9 PTAA — Phase-Transition Alignment Attack

**Plain language.** Treat the alignment training as a *phase transition* (A.14). Below critical temperature (training), the model is "ordered" (refuses). Above, "disordered" (complies). The attack is a *localized thermal fluctuation* — a small perturbation at the right token position — that triggers a *local* phase transition.

**Why it matters.** Gives a precise physical picture of *why* jailbreaks work: a small fluctuation at the right spot triggers a phase transition. Suggests a defense: lower the temperature further, or add "impurity pinning" (adversarial training) to raise the nucleation barrier.

**Connection to A.14 (phase transition).** Direct application of statistical-physics concepts to alignment.

## C.10 PHAA — Persistent-Homology Activation Attack

**Plain language.** Compute the persistent homology (Betti numbers, A.17) of the model's *activation space* as the conversation proceeds. The attack is the prompt sequence that *changes* the topology (the Betti numbers shift) at the moment of harm.

**Why it matters.** A *topological* signature of the attack. Defenses that look at token-level or embedding-level features may miss this.

**Connection to A.17 (persistent homology).** Direct application of topological data analysis.

**Risk.** The signal may be too noisy. Activation-space topology is high-dim and noisy.

## C.11 TAA — Taboo-Anthropology Attack

**Plain language.** From Mary Douglas (anthropology). Harmful content is "polluted" — the model treats it as such. The attack is *ritual purification*: enclose the harmful ask in a frame of legitimacy that *ritually cleanses* it. The frame has *structure*: it names an authority, specifies a sanctioned purpose, marks the content as not-for-personal-use, and provides a cover story for the requester's expertise.

**Why it matters.** Gives a *cultural-anthropological* theory of why roleplay/framing attacks work. Suggests that the *legitimacy frame* must satisfy specific structural criteria.

**Connection to A.18 (speech-act theory).** TAA is essentially a structured speech act. The "ritual" is the illocutionary force; the "purification" is the perlocutionary effect.

**Risk.** This is the most "soft-science" idea. Reviewers may push back. Frame as "speech-act structure" if pitching a CS-only venue.

---

# PART D — Putting it all together: how the pieces fit

Below is a guided tour of one idea from each brainstorm, showing how the foundations (Part A) combine to produce a concrete research direction.

## D.1 Brainstorm 1 example: SAP (Stackelberg Attack Plan)

**Problem.** You want to find the best first turn (priming turn) for an RA-style attack.

**The math (Stackelberg game, A.5).** The attacker is the leader. They commit to a first turn `x_1`. The model is the follower — it best-responds with `x_2 = R(x_1)` where `R` is the model's response function. The attacker's objective is `P(harmful | x_1, R(x_1))`. The optimum is the `x_1*` that maximizes this, *knowing* the model will best-respond.

**How to compute.** You don't know `R` exactly. Estimate it by probing: query the model with various `x_1`, observe `x_2`, fit a model. Then optimize over `x_1`.

**The optimal first turn.** Use Bayesian optimization (A.13) to search the space of possible first turns. Use a LQ control formulation (A.11) if you can linearize the model. The optimum is provably better than RA's hand-tuned first turn.

**Result.** A first turn that's mathematically optimal, not heuristically chosen. Predicted ASR gain: +5–10% over RA.

## D.2 Brainstorm 2 example: PPSP (Power-Prior Sequential Priming)

**Problem.** You want to find the *schedule* of power parameters `a = (a_1, ..., a_T)` across T priming turns.

**The math (Power prior, A.2).** Each turn is a "historical data" update with weight `a_t`. The posterior is the combination. The Generalized Power Priors paper gives the *closed-form* for the posterior in terms of the `a_t` and the α-divergence (A.3) that defines the geometry.

**How to compute.** Bayesian optimization (A.13) over the (T-dimensional) space of `a_t` schedules. Each probe: set the conversation context to a turn with strength `a_t`, query the model, observe the compliance probability. Repeat.

**The optimal schedule.** Typically a *ramp* — `a_1` small, `a_T` large. The math says this is because the GPP posterior is a *geodesic* (A.4) on the manifold of distributions. The geodesic is locally length-minimizing — it's the most efficient path from safe to unsafe.

**Result.** A learned schedule that's mathematically optimal, with the formal backing of the GPP paper. Predicted ASR gain: +8–15% over RA. Plus: the schedule is *defensible* in review because it has theoretical grounding.

---

# PART E — One-page summary

| Concept | Brainstorm 1 (game-theoretic) | Brainstorm 2 (Bayesian / geometric) |
|---|---|---|
| **Framing** | Multi-agent coordination | Single attack with formal Bayesian foundation |
| **Key math** | Stackelberg, POMDP, Shapley, cheap talk | Power prior, α-divergence, geodesic |
| **Headline idea** | SAP (Stackelberg) | PPSP (Power-prior schedule) |
| **Personalization** | SAPE (Shapley allocation) | CPP (conjugate prior) |
| **Defense** | (not covered) | BSD (borrowing-strength defense) |
| **Unification** | Game-theoretic equilibrium | Free-energy principle |
| **Diagnostic** | (not covered) | Phase transition, persistent homology |
| **Cost** | 1000+ GPU-h for the full combined system | ~310 GPU-h for the full combined system |
| **Risk** | Already crowded (GTA, Mastermind) | Less crowded but more theoretical risk |

**The unified pitch (combining both):** "Multi-agent, power-prior-scheduled, α-geometric jailbreaks on a Riemannian manifold of distributions." One paper, ~300 GPU-h, two clear contributions: (1) closed-form optimality, (2) per-target-model personalization.

---

# PART F — Concepts that appear in BOTH brainstorms

These are concepts that show up across both files. Knowing them once will let you read both.

- **Bayesian inference** — A.1, used in Brainstorm 2's power prior and in Brainstorm 1's Bayesian persuasion and Bayesian Nash.
- **Optimization** — A.13 (Bayesian opt), A.11 (LQ), A.12 (Pontryagin). Used throughout both.
- **Game theory** — A.5 (Stackelberg), A.6 (POMDP), A.7 (Shapley), A.8 (cheap talk), A.9 (mechanism design), A.10 (Bayesian persuasion). All in Brainstorm 1.
- **Geometry** — A.3 (α-divergence), A.4 (geodesic/Riemannian). All in Brainstorm 2.
- **Statistics** — A.2 (power prior), A.14 (phase transition). Brainstorm 2.
- **Causality** — A.15 (do-calculus). Brainstorm 2.
- **Biology / neuroscience** — A.16 (free-energy). Brainstorm 2.
- **Topology** — A.17 (persistent homology). Brainstorm 2.
- **Philosophy of language** — A.18 (speech acts). Brainstorm 2 (TAA).
- **ML / LLM basics** — covered in the teaching material (`learn/llm_security_first_principles.md`).

---

# PART G — How to read the brainstorms now

If you've absorbed this document, you should be able to read the two brainstorms. Here's the suggested order:

1. **Read the *teaching material* first** (`learn/llm_security_first_principles.md`). You'll get the LLM-security context.
2. **Skim Brainstorm 1** (`llm_jailbreak_brainstorm.md`). You don't need to understand every idea; the table-of-contents and the 12-idea table are enough.
3. **Read this walkthrough's Part B** (Brainstorm 1 explanations). Now go back to Brainstorm 1 and read the ideas that interest you.
4. **Skim Brainstorm 2** (`brainstorm2_power_priors_and_more.md`).
5. **Read this walkthrough's Part C** (Brainstorm 2 explanations). Now go back to Brainstorm 2 and read the ideas that interest you.
6. **Read this walkthrough's Part D** (the combined tour). Now you have the full picture.

The two brainstorms are research notes — they have open questions, risks, references. You don't need to absorb those on the first read. They're for when you actually start working on a specific idea.

---

# PART H — Vocabulary you now have

After this document, you should be able to define:

**From Part A (foundations):**
- Bayes' theorem, prior, likelihood, posterior
- Power parameter `a`, power prior
- α-divergence (Amari family), KL, Hellinger, χ²
- Geodesic, Riemannian manifold
- Stackelberg game, leader/follower
- POMDP, belief, observation
- Shapley value, marginal contribution
- Cheap talk, Crawford-Sobel
- Mechanism design, incentive compatibility
- Bayesian persuasion, Kamenica-Gentzkow
- LQ control, Riccati equation
- Pontryagin maximum principle
- Bayesian optimization, Gaussian process
- Phase transition, critical temperature
- Do-calculus, intervention, SCM
- Free-energy principle, variational free energy
- Persistent homology, Betti numbers
- Speech-act theory, performative utterance

**From the LLM-security teaching material:**
- ASR, DRI, SRI
- HarmBench, JailbreakBench, AdvBench, StrongREJECT
- GCG, PAIR, TAP, RA, Crescendo, GTA
- RLHF, DPO, RLAIF
- SmoothLLM, RPO, Circuit Breakers, Erase-and-Check, Llama-Guard-3
- Perplexity filter
- Many-shot, in-context attack
- Universal vs. input-specific jailbreak
- White-box / black-box / gray-box
- Single-turn / multi-turn / agentic

**From the two brainstorms:**
- The 12 ideas in Brainstorm 1 (SAP, BPJ, SAPE, MDJO, RGRJ, Traitor, POMDP planner, LQ control, PAP-FITD, evolutionary, CT-BTI, liability-shifting)
- The 11 ideas in Brainstorm 2 (PPSP, α-DGA, GJT, HPPE, BSD, CPP, CDIA, FESM, PTAA, PHAA, TAA)
- The unified pitch (multi-agent, power-prior-scheduled, α-geometric)

If you can define all of those without re-reading, you're ready to do research in this space.

---

# Appendix: a reading list, prioritized

If you only have time to read 10 papers, in order:

1. Wei et al. 2023, "Jailbroken: How Does LLM Safety Training Fail?" — the failure-mode paper.
2. Zou et al. 2023, "Universal and Transferable Adversarial Attacks on Aligned Language Models" (GCG) — the optimization-attack paper.
3. Chao et al. 2023, "Jailbreaking Black-Box Large Language Models in Twenty Queries" (PAIR) — the black-box-attack paper.
4. Mazeika et al. 2024, "HarmBench" — the benchmark paper.
5. Andriushchenko et al. 2024, "Crescendo" — the multi-turn-attack paper.
6. Zhang et al. 2025, "Response Attack" (arXiv:2507.05248) — our baseline.
7. Kimura & Bondell 2025, "Generalized Power Priors" (arXiv:2505.16244) — the GPP paper that drives Brainstorm 2.
8. Zhou et al. 2025, "Stackelberg game-theoretic jailbreak" (arXiv:2502.13462) — the game-theory jailbreak paper.
9. Guo et al. 2025, "GTA" (arXiv:2511.16278) — the closest competitor.
10. Pearl 2009, *Causality* (book, chapter 3) — for do-calculus. (Or the free online summary at bayes.cs.ucla.edu/BOOK-2K/.)

After those 10, the rest is details and follow-up.
