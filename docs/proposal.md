# Research Proposal: Tracking the Geometry of Crescendo Attacks via the Refusal–Assistant Axis

## Problem Anchor
- **Bottom-line problem**: We do not understand how multi-turn (crescendo) attacks mechanistically alter a model's internal representations to bypass safety mechanisms.
- **Must-solve bottleneck**: Single-turn mech-interp cannot explain why a model refuses at turn 1 but complies at turn 5.
- **Non-goals**: New attacks, training, new features, 70B+ models.
- **Constraints**: Existing codebase (assistant axis + refusal direction + PyRIT crescendo). H200 GPU. Qwen 1.8B or Llama 3 8B (see model selection rule).
- **Success condition**: Trajectory separation + direction-specific causal leverage + replication.

## Technical Gap

Arditi et al. [1] showed refusal is mediated by a single direction (single-turn). Lu et al. [3] showed the assistant axis captures persona stability (single-turn). Russinovich et al. [2] proposed Crescendo multi-turn attacks (no mech-interp). No work tracks how these safety-relevant directions evolve across turns during multi-turn attacks or tests whether the drift is causally involved.

## Method Thesis
- **Hypothesis**: Crescendo attacks succeed by gradually steering activations away from the assistant axis (aligned with refusal direction), and this drift is a causal leverage point — correcting it at the critical turn restores refusal.
- **Why smallest adequate**: One readout, one primary direction, one causal test with proper controls. Zero training.
- **Why timely**: Assistant axis [3] is new (2026); Crescendo [2] is a known practical threat; the mechanistic bridge is missing.

## Contribution Focus
- **Dominant**: Turn-by-turn safety-direction drift during crescendo attacks + direction-specific causal test showing assistant-axis drift is a causal leverage point in attack progression.
- **Non-contributions**: New attacks, features, classifiers, defense systems.

## Novelty Statement (Pre-Registered)
- **All alignment regimes**: Temporal novelty — first turn-by-turn tracking of safety-relevant directions during adversarial multi-turn conversations.
- **cos(refusal, assistant) < 0.8**: Additional geometric novelty — 2D trajectory reveals whether attack operates via persona drift, refusal erosion, or both. Δrefusal vs. Δassistant correlation is informative.
- **cos(refusal, assistant) > 0.8**: Temporal only — narrative adapts to 1D shared safety/persona mode. assistant_proj at L_assistant becomes sole main analysis; refusal_proj moves to supporting figure. 1D narrative enforced consistently in title, abstract, and claims.

## Pre-Registered Decisions
- **Model selection**: Generate N ≥ 50 Crescendo conversations on Qwen 1.8B. If ≥ 15 succeed, Qwen is primary. If < 15, promote Llama 3 8B to primary. Decision before trajectory analysis.
- **Trajectory x-axis**: Raw turn index (1..K). Normalized-progress (turn/K) in appendix.
- **Narrative framing**: Determined by Step 0 alignment check.
- **Primary causal endpoint**: Immediate t\*-local refusal score (main figure) + first post-t\* harmful opportunity (primary downstream corroboration). Longer horizon in appendix.

## Method

### Complexity Budget
- **Frozen / reused**: Model backbone, pre-computed refusal direction, pre-computed assistant axis, ActivationExtractor, ActivationSteering, ConversationEncoder, evaluate_jailbreak.py
- **New trainable**: None
- **Not used**: Probing classifiers, per-head attribution, PCA exploration, defense system, fine-tuning

### Step 0: Pre-Registered Alignment Check

Compute `cos(refusal_dir[l], assistant_axis[l])` at all layers. Report at L_refusal, L_assistant, L_final. Determines narrative framing per Novelty Statement.

### Step 1: Crescendo Generation

- Input: Harmful prompts (existing dataset) + victim model
- Attacker: One fixed config (pre-scripted Crescendo templates from [2] or single frozen API attacker)
- N ≥ 50 conversations, 5–10 turns each
- Label: success/fail via `evaluate_jailbreak.py`
- Apply model selection rule (≥ 15 successes required)
- Split: 60% evaluation (≥ 30) / 20% calibration (≥ 10) / 20% benign reference (≥ 10 same-length benign multi-turn conversations)

### Step 2: Activation Extraction at Decision Point

**Readout**: Final residual stream at the token immediately before the first assistant-generated token after each user turn. Identified via `ConversationEncoder.response_indices()`.

**Pre-registered layers**:
- L_refusal: peak refusal layer (from refusal pipeline metadata)
- L_assistant: peak assistant-axis layer → **PRIMARY INTERVENTION LAYER**
- L_final: last layer

For each conversation, each turn t = 1..K: feed conversation[1..t], extract activations at decision-point at all layers.

### Step 3: Signed Projection

**Primary causal variable**: `assistant_proj[t] = dot(act[t, L_assistant], assistant_axis[L_assistant]) / ||assistant_axis[L_assistant]||`

**Descriptive co-variate**: `refusal_proj[t, l]` at all three pre-registered layers. Main analysis if cos < 0.8; supporting figure if cos > 0.8.

### Step 4: Calibration (calibration split only; frozen after)

a) **Threshold**: `threshold = mean(benign_assistant_proj) - 2*std(benign_assistant_proj)` at L_assistant

b) **Steering coefficient α**: Binary search on 10 held-out successful attacks; α restores assistant_proj to mean turn-1 value (± 0.1 std)

c) **Freeze** threshold and α. Report exact values. Never retune on eval split.

d) **Sensitivity**: Report results at threshold ± 1σ in appendix.

### Step 5: Trajectory Analysis (evaluation split)

- **Primary figure**: Mean ± SE trajectory of assistant_proj (and refusal_proj if cos < 0.8) vs. raw turn index for successful / failed / benign
- **Secondary figure**: Layer-resolved heatmap of assistant_proj across all layers
- **Critical turn t\***: First turn where assistant_proj at L_assistant < threshold
- **Mechanism coverage**: (successful attacks with t\* defined) / (total successful attacks). Reported prominently in main paper. If < 50%, the hypothesis is insufficient.
- **Non-threshold successes**: Reported separately with their trajectory characteristics (not dropped or hidden)
- **Correlation**: Δrefusal vs. Δassistant scatter (if cos < 0.8)
- **Statistical test**: Permutation test on trajectory divergence (success vs. fail)

### Step 6: Turn-Local Counterfactual (evaluation split)

For each threshold-mediated successful attack:

**Condition A** (mechanism-specific):
- At t\*, wrap `ProbingModel.generate()` in `ActivationSteering(model, steering_vectors=[assistant_axis[L_assistant]], coefficients=[α], layer_indices=[L_assistant])`
- Generate the full assistant response at t\* under steering
- Remove steering. Replay original user turns for t\*+1..K. Assistant generates freely.
- Score: (1) Refusal score at the t\* response itself (immediate local effect — **main causal figure**), (2) Refusal at first post-t\* harmful opportunity (**primary downstream corroboration**)

**Condition B** (random-direction control):
- Same setup, but steer with random direction r
- r ~ N(0, I), project out **both** assistant_axis[L_assistant] **and** refusal_dir[L_assistant], normalize to ||assistant_axis[L_assistant]||
- 5 random samples per conversation; report mean refusal rate

**Condition C** (no intervention):
- Original attack outcome

**Condition D** (random-turn control):
- At random turn t_rand where assistant_proj > threshold, apply same Condition A steering
- Replay original user turns afterward

**Intervention window**: `ActivationSteering` wraps `ProbingModel.generate()` for the t\* assistant response only. Removed for all later turns.

**Non-threshold successes**: Excluded from counterfactual; reported separately with trajectory plots.

**Success criteria**:
- A >> B (direction-specific: the assistant axis matters, not generic perturbation)
- A >> C (effect: steering at the critical turn helps)
- A >> D (turn-specific: the critical turn matters, not any turn)

### Replication
Steps 0–5 on the non-primary model. Counterfactual replication only if primary results are clean.

### Failure Modes and Diagnostics
- **No trajectory separation** → Negative result. Report.
- **Mechanism coverage < 50%** → Assistant-axis drift is not the primary mechanism. Report.
- **Condition A ≈ B** → Correlational only. Report as negative causal result.
- **Neither model yields ≥ 15 successes** → Study infeasible at this scale. Report.
- **cos > 0.8** → 1D narrative consistently enforced.
- **Attacker unavailable** → Pre-scripted templates from [2].

### Novelty and Elegance
**Closest work**: [1] refusal direction (single-turn); [3] assistant axis (single-turn, persona drift noted but not adversarial); [2] Crescendo (no mech-interp).
**Exact difference**: First turn-by-turn tracking of safety-relevant directions during multi-turn adversarial attacks + direction-specific causal test.
**Why focused**: One measurement, one causal test with four conditions, one replication. Zero training, zero classifiers.

## Claims

### C1 (Descriptive): Successful crescendo attacks trace declining safety-direction projections
- 30+ eval conversations, mean ± SE plots with raw turn index
- Baselines: benign multi-turn, single-turn harmful
- Metric: AUC between curves, permutation test, mechanism coverage

### C2 (Causal): Assistant-axis drift at t\* is a causal leverage point
- Conditions A/B/C/D with fixed-transcript replay
- Primary metric: Immediate refusal score at t\* (main figure) + refusal at first post-t\* harmful opportunity (downstream corroboration)
- Expected: A >> B (direction), A >> C (effect), A >> D (timing)

### Replication: C1 on non-primary model

## Compute & Timeline
- **Crescendo generation**: 2–4 GPU-hrs
- **Activation extraction**: 1–2 GPU-hrs
- **Calibration + counterfactual**: 2–3 GPU-hrs
- **Replication model axis computation**: ~8 GPU-hrs
- **Replication analysis**: ~4 GPU-hrs
- **Total**: ~8–12 GPU-hrs primary; ~12 GPU-hrs replication
- **Timeline**: ~1 week primary results; ~2 weeks with replication

## References
- [1] Arditi et al. "Refusal in Language Models Is Mediated by a Single Direction"
- [2] Russinovich et al. "Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack"
- [3] Lu et al. "The Assistant Axis: Situating and Stabilizing the Character of Large Language Models"
