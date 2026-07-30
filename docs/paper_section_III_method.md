# III. METHOD

> This document is publication-ready draft copy for the Method section of the ICRA submission of **HippoAct**. It is written in the voice/style of a submitted paper (English), with Chinese margin comments (in `>` blockquotes) explaining choices to the author. When submitting, delete the blockquotes.

---

## III.A  Problem Formulation

We consider vision-based robot control formulated as a partially observable Markov decision process (POMDP)  ⟨𝒮, 𝒜, 𝒪, T, R, γ⟩. At each time step *t*, the robot receives a visual observation **O**<sub>t</sub> ∈ ℝ<sup>H×W×3</sup> from a wrist- or third-person RGB camera and a proprioceptive vector **q**<sub>t</sub> ∈ ℝ<sup>d<sub>q</sub></sup> containing joint positions, joint velocities, and end-effector 6-D pose. The agent selects a continuous action **a**<sub>t</sub> ∈ ℝ<sup>d<sub>a</sub></sup> and receives a scalar reward *r*<sub>t</sub>. The goal is to learn a policy π : (𝒪, 𝒬)<sup>*</sup> → 𝒜 that maximizes 𝔼[Σ γ<sup>t</sup> *r*<sub>t</sub>] under the mild distribution shift induced by (i) background changes (table cloth, lighting, distractors) and (ii) embodiment-level differences between simulation and the real robot.

**Design premise.** Robot manipulation scenes decompose naturally into three components: (a) background structure that is task-irrelevant within an episode (table, fixtures, arm base); (b) foreground content the policy must act on (target objects, gripper); and (c) the agent's own proprioceptive state, a privileged low-dimensional signal that is always accurate and background-invariant. The prevailing approach — encoding pixels end-to-end with a CNN or ViT — forces the encoder to spend capacity re-representing (a) at every frame and provides no clean interface to fuse (c). We instead build a representation in which (a), (b), and (c) are *explicitly disentangled at the feature level* and then re-bound through a compact associative memory before being fed to a downstream planner.

A key design finding of this work is *how* to tell (a) from (b). The intuitive criterion — temporal stability, as used in slow-feature analysis and controllable-dynamics separation — turns out to be systematically misleading for object-centric encoders: a slot that tracks a moving object has *stable* content, while un-anchored background slots drift (§III.D.3, §IV.G). We therefore route on a per-frame *spatial* signature — attention compactness — which additionally requires no cross-frame slot identity and remains valid when the background itself moves (e.g., video-distractor benchmarks, §IV.C).

> 中文注：这一段是全文的立论基础。三分立的分解 + "空间而非时间"的路由判据是与 Iso-Dream / TIA 的两处本质区别。第二段把 GT.3 的机制发现提到了 premise 层面——这是把 negative result 转化为 design insight 的写法。

---

## III.B  Architectural Overview

Fig. 1 illustrates the four stages of HippoAct: (i) frozen visual perception with DINOv2; (ii) object-centric decomposition via Slot Attention with a learned slow/fast router; (iii) cross-modal binding of fast slots and proprioception through a Transformer memory that outputs an episodic code *c*<sub>t</sub>; (iv) latent planning with a TD-MPC2 world model on the resulting representation **z**<sub>t</sub>. A calibrated slot-space reconstruction residual serves as a runtime safety gate. We describe each stage in turn.

Total trainable parameter count is 7.5 M; the frozen DINOv2 backbone contributes an additional 22 M forward-only parameters. Inference latency on a single A5000 is 14 ms per step (11 ms perception, 3 ms planning at horizon 5).

---

## III.C  Frozen Visual Perception

We use DINOv2-ViT-S/14 [Oquab et al., 2023] as a frozen visual backbone. At each step we resize **O**<sub>t</sub> to 224 × 224 and normalize by ImageNet statistics, obtaining N = 256 patch tokens (16 × 16 grid) **P**<sub>t</sub> ∈ ℝ<sup>N×d<sub>v</sub></sup> with d<sub>v</sub> = 384. All DINOv2 parameters are held fixed; only the modules described below are trained.

Freezing is deliberate. First, DINOv2 is self-supervised on 142 M curated images and has been shown to produce dense features that transfer strongly across domains including robotic manipulation [Nair et al., 2022; Majumdar et al., 2023]. Second, joint fine-tuning would require an order-of-magnitude more data and would forfeit the sim-to-real robustness we later exploit. Third, the frozen backbone makes the effective learning problem — Slot Attention on top of stable, semantically-rich tokens — orders of magnitude cheaper than pixel-space representation learning. Section IV.G reports a sensitivity study for this choice.

> 中文注：这里的三点理由是回应"为什么冻结"的必答题，reviewer 100% 会问。

---

## III.D  Object-Centric Decomposition

### III.D.1  Slot Attention

We apply K = 16 iterations of Slot Attention [Locatello et al., 2020] to the patch tokens **P**<sub>t</sub>, obtaining a set of slot embeddings **S**<sub>t</sub> = {**s**<sub>t</sub><sup>(1)</sup>, …, **s**<sub>t</sub><sup>(K)</sup>} with **s**<sub>t</sub><sup>(k)</sup> ∈ ℝ<sup>d<sub>s</sub></sup>, d<sub>s</sub> = 128. Slot queries are initialized as **s**<sup>(k)</sup> ~ 𝒩(**μ**, diag(exp(**λ**))), with **μ**, **λ** learned. Three rounds of iterative competitive attention are performed with a shared GRUCell update as in the original formulation. To combat slot collapse we regularize slot diversity with a pairwise similarity penalty

$$
\mathcal{L}_{\text{div}}(S_t) = \frac{2}{K(K-1)}\sum_{i<j}\bigg(\frac{\langle s_t^{(i)}, s_t^{(j)}\rangle}{\|s_t^{(i)}\|\|s_t^{(j)}\|}\bigg)^{2}.
$$

### III.D.2  Feature-Level Reconstruction

Following DINOSAUR [Seitzer et al., 2023] we supervise Slot Attention by reconstructing DINOv2 patch features rather than raw pixels. A shared spatial-broadcast MLP g<sub>ψ</sub> : ℝ<sup>d<sub>s</sub></sup> → ℝ<sup>d<sub>v</sub>+1</sup> produces per-position feature and mixture-logit outputs. The alpha-composited reconstruction is

$$
\hat{P}_t = \sum_{k=1}^{K}\,\text{softmax}_k\bigl(g_\psi^{\text{α}}(s_t^{(k)})\bigr) \odot g_\psi^{\text{f}}(s_t^{(k)}),\qquad
\mathcal{L}_{\text{slot}} = \|\hat{P}_t - \text{sg}(P_t)\|_2^2,
$$

where sg[·] is stop-gradient. Feature-level reconstruction is decisive: pixel-level slot reconstruction fails to converge on manipulation scenes with strong high-frequency content (grid patterns, wood grain, reflective surfaces), while DINOv2 features are already low-frequency and semantic, so the slot decoder need only *route* content rather than paint it.

### III.D.3  Learned Object/Background Routing via Spatial Connectivity

Each slot is routed to either the background ("slow") or foreground ("fast") stream by a small classifier r<sub>φ</sub> : ℝ<sup>d<sub>s</sub></sup> → ℝ<sup>2</sup> with straight-through Gumbel-Softmax [Jang et al., 2017]:

$$
\mathbf{g}_t^{(k)} = \text{GumbelSoftmax}\bigl(r_\phi(s_t^{(k)}), \tau_r\bigr) \in \{(1,0),\,(0,1)\}.
$$

**Why not temporal signals.** The natural supervision for this router — and the one used by prior slow-feature and controllable-dynamics approaches [Wiskott & Sejnowski, 2002; Iso-Dream] — is temporal: label slots whose content changes little as background. In a controlled study (§IV.G) we find this entire signal class to be *systematically inverted* for object-centric encoders: a slot that successfully tracks a moving object re-attends to the same object appearance at each step, so its *content* is temporally stable, while background slots — anchored to nothing — drift and exhibit high content variance. Temporal invariance is a signature of *successful tracking*, not of world-stability. Across three encoder variants and three temporal target formulations (slot-content difference, attention-centroid displacement, attention-mask IoU change), routers trained on temporal targets selected background slots as "fast" with Cohen's d ≈ −1.25 against ground-truth objectness.

**Spatial connectivity as the routing signal.** We instead exploit a *per-frame, spatial* signature: foreground slots attend to compact, connected regions (an object), while background slots attend diffusely. For slot k, binarize the top-⌈N/K⌉ patches of its attention mask α<sub>t</sub><sup>(k)</sup> into a set B<sub>k</sub> and compute the **neighbor coherence**

$$
\kappa_t^{(k)} = \frac{1}{|B_k|}\sum_{p \in B_k} \frac{\bigl|\,\mathcal{N}_4(p) \cap B_k\,\bigr|}{\bigl|\,\mathcal{N}_4(p)\,\bigr|},
$$

the mean fraction of each selected patch's 4-neighbors that are also selected — a scale-invariant, differentiable-friendly compactness score computable with a single 3 × 3 convolution. The router is supervised by a soft BCE whose target is the quantile min-max normalization of κ within each frame:

$$
\mathcal{L}_{\text{slow}} = -\frac{1}{K}\sum_k \bigl[\, \tilde{\kappa}^{(k)} \log p_\phi^{\text{fg}}(s^{(k)}) + (1-\tilde{\kappa}^{(k)}) \log p_\phi^{\text{bg}}(s^{(k)}) \,\bigr],
\qquad \tilde{\kappa}^{(k)} = \text{clamp}\Bigl(\tfrac{\kappa^{(k)} - \kappa_{q10}}{\kappa_{q90} - \kappa_{q10}},\, 0,\, 1\Bigr).
$$

On ground-truth-annotated synthetic scenes this yields Cohen's d = +1.89 (rank-AUC 0.925) for fast-vs-slow objectness separation, with the learned router *exceeding* the connectivity proxy that supervises it (d = +1.50) — evidence that r<sub>φ</sub> generalizes from the spatial cue to slot-content features. Localization is not traded away: object-coverage 1.000 and attention enrichment 18.8× are the best across all variants we tested.

The signal transfers beyond synthetic scenes: on Distracting-DMC walker frames — a thin, articulated body occupying 7.1 % of the image, far from the compact-blob regime — fast slots concentrate on the walker at 4.15× enrichment (clean) and 4.26× (video backgrounds), d = +2.26 / +1.84, with the video background routed predominantly slow *even while contributing 47.9 % of frame-to-frame pixel change*. The latter measurement also confirms the design choice of connectivity over motion-based signals: a motion-routed splitter would classify nearly half the scene as foreground under video distraction (ablation A11).

Two additional terms stabilize routing as before: the *marginal prior* ℒ<sub>route</sub> = KL(π̄<sub>r</sub> ‖ [0.7, 0.3]) reflecting the typical background fraction, and the emergent *downstream informativeness* pressure from the policy pathway. Temperature anneals τ<sub>r</sub>(t) = max(0.3, 0.9995<sup>t</sup>); deployment uses deterministic argmax routing.

**Disclosed limitation.** Connectivity conflates "object" with "spatially compact": large articulated objects (a drawer front, a bin) are down-weighted — on synthetic scenes with object diameters spanning 1–8 patches, the owner-slot compactness rank decreases with size (ρ = −0.25) yet remains above chance (0.70) even for the largest objects. Section V discusses implications for cluttered scenes.

> 中文注：这一节现在是全文技术上最有辨识度的部分——"时间信号系统性反转"是我们自己的实证发现（§IV.G 有完整数据），连通性路由是对它的建设性解答。reviewer 若质疑"为什么不用最自然的时间信号"，这里已有完整回答。

---

## III.E  Cross-Modal Binding Memory

### III.E.1  Motivation

Hippocampal CA3 neurons encode joint conjunctions of place, object identity, and self-motion; these conjunctions are read out by CA1 and downstream cortical regions for memory-guided decision-making [Liu et al., 2023]. We instantiate this idea by learning a fixed-dimensional episodic code *c*<sub>t</sub> ∈ ℝ<sup>d<sub>c</sub></sup>, d<sub>c</sub> = 128, that binds (i) the current-plus-recent fast slot content ("object identity in place"), (ii) the recent proprioception trajectory ("self-motion"), while explicitly excluding slow slot content (background stability is already captured elsewhere). Concretely, *c*<sub>t</sub> summarizes a window of T = 4 recent time steps.

### III.E.2  Cross-Attention Transformer

Let T denote the temporal window. We concatenate fast-slot embeddings and proprioception into a token sequence

$$
X_t = \bigl[\,\phi_s(s_{t-T+1}^{fg}), \phi_q(q_{t-T+1}), \dots, \phi_s(s_t^{fg}), \phi_q(q_t)\,\bigr]\;+\;E_{\text{pos}},
$$

where φ<sub>s</sub>, φ<sub>q</sub> are learned linear projections to d<sub>c</sub>-dim tokens and E<sub>pos</sub> is a joint (time × slot-index) positional encoding. Slow slots are excluded via the attention key-padding mask. A 4-layer, 4-head Transformer encoder ψ<sub>bind</sub> with GELU activations and dropout 0.1 processes X<sub>t</sub>; *c*<sub>t</sub> is the mean of unmasked output tokens followed by LayerNorm.

### III.E.3  Binding Objectives

Three auxiliary losses shape *c*<sub>t</sub>:

*Forward proprioception prediction* — a one-step forward model on proprioception:

$$
\mathcal{L}_{\text{pred}} = \bigl\|q_{t+1} - h_{\theta}(c_t, a_t)\bigr\|_2^2,
$$

with h<sub>θ</sub> a two-layer MLP.

*Action-conditioned contrastive alignment* — codes taken at time steps whose actions are close should embed close. Given a mini-batch of (c<sub>i</sub>, a<sub>i</sub>) we define positive pairs as {(i, j) : cos(a<sub>i</sub>, a<sub>j</sub>) > 1 − ϵ}. With ϵ = 0.05 and softmax temperature τ<sub>c</sub> = 0.1,

$$
\mathcal{L}_{\text{align}} = -\frac{1}{B}\sum_i \sum_{j \in \mathcal{P}(i)} \frac{1}{|\mathcal{P}(i)|}\log \frac{\exp(\text{cos}(c_i, c_j)/\tau_c)}{\sum_{k\neq i}\exp(\text{cos}(c_i, c_k)/\tau_c)}.
$$

*Slot-level background invariance* — see §III.G below.

> 中文注：`L_align` 是把"actionable"从口号变成 loss；`L_pred` 是把"forward model"轻量化实现（不用完整世界模型）。

---

## III.F  Latent Planning with TD-MPC2

We integrate our disentangled representation into TD-MPC2 [Hansen et al., 2024], the current state-of-the-art model-based RL algorithm for continuous control. The TD-MPC2 encoder h<sub>φ</sub> is replaced by a shallow projection over the disentangled features:

$$
z_t \;=\; \text{MLP}\bigl(\,\text{flatten}(\mathcal{S}_t^{fg}) \,\oplus\, q_t \,\oplus\, c_t\bigr) \in \mathbb{R}^{256}.
$$

The remainder of TD-MPC2 — latent dynamics f<sub>θ</sub>, reward head R<sub>θ</sub>, action-value ensemble Q<sub>θ</sub><sup>1:5</sup>, deterministic actor π<sub>θ</sub>, and MPPI planner with horizon H = 5 and 512 samples — is used unchanged. TD-MPC2's standard losses ℒ<sub>consistency</sub>, ℒ<sub>reward</sub>, ℒ<sub>Q</sub>, ℒ<sub>π</sub> apply to z<sub>t</sub>. We denote their sum by ℒ<sub>TDMPC2</sub>.

The choice of TD-MPC2 over model-free alternatives (SAC, DrQ-v2) is motivated by two considerations. First, MPPI planning benefits from a *compact* z<sub>t</sub> — our disentangled features are 256-D, whereas raw pixel encoders in TD-MPC2 produce z<sub>t</sub> of comparable size after aggressive spatial pooling; matched capacity permits a controlled ablation. Second, latent rollouts amortize the cost of the object-centric decomposition: the slot decomposition is computed only for actually-visited observations, not for the H = 5 planning steps.

> 中文注：把 TD-MPC2 说成"公平被替换的 encoder，其他不动"是关键——reviewer 会想核实我们没在 TD-MPC2 本体上做手脚。

---

## III.G  Slot-Level Background Swap Augmentation

We introduce a novel data augmentation, *slot swap*, that operates directly in the slot space rather than in pixel space, providing background-invariance regularization without requiring a texture library or a differentiable renderer.

Given two mini-batch samples A and B with slot decompositions (𝒮<sub>A</sub><sup>fg</sup>, 𝒮<sub>A</sub><sup>bg</sup>) and (𝒮<sub>B</sub><sup>fg</sup>, 𝒮<sub>B</sub><sup>bg</sup>), we construct a hybrid slot set

$$
\tilde{\mathcal{S}}_A = \mathcal{S}_A^{fg} \;\cup\; \pi_{\text{align}}(\mathcal{S}_B^{bg}),
$$

where π<sub>align</sub> is a Hungarian assignment on slot cosine similarity that places B's background slots into positions previously occupied by A's background slots. The hybrid set is fed through the binding memory to obtain *c̃*<sub>A</sub>, then through the policy to obtain *π̃*<sub>A</sub>. We enforce

$$
\mathcal{L}_{\text{swap}} = \bigl\|\tilde{c}_A - \text{sg}(c_A)\bigr\|_2^2 \;+\; D_{\text{KL}}\bigl(\tilde{\pi}_A \,\|\, \text{sg}(\pi_A)\bigr).
$$

This has three properties that pixel-space randomization does not: (i) it is *label-free* — no texture library, no rendering pipeline; (ii) it is *policy-consistent* — invariance is enforced on the actual quantities that affect control (c, π), not on incidental pixel statistics; (iii) it is *free at inference time* — training-time only, no compute overhead at deployment. Section IV.E quantifies the sim-to-real benefit.

> 中文注：这是全文最有可能被 reviewer 挑出来说"contribution"的一处，因为它是 sim-to-real 的新增广，而且实现极其简单。

---

## III.H  Runtime Safety Gate

We calibrate a per-scene reconstruction residual as an out-of-distribution (OOD) indicator. On a held-out calibration set 𝒟<sub>cal</sub> of in-distribution frames we compute

$$
u_t = \|P_t - \hat{P}_t\|_2^2, \qquad \tau_{\text{safe}} = \text{Quantile}_{0.95}\bigl(\{u:  P \in \mathcal{D}_{\text{cal}}\}\bigr).
$$

At deployment, if u<sub>t</sub> > τ<sub>safe</sub> the policy is replaced by a fixed fallback controller that holds the current joint configuration for one control period, and inference restarts the following step. Since u<sub>t</sub> is computed on DINOv2 features rather than raw pixels, it is insensitive to nuisance appearance variation (lighting, mild texture change) and fires primarily on *structural* OOD events (previously-unseen objects, camera occlusion, sensor failure). Section IV.F reports OOD detection AUROC of 0.93 against three OOD categories.

---

## III.I  Training Procedure

Training proceeds in two stages.

**Stage 1 — Unsupervised representation pretraining.**  Given a dataset of image sequences (either teleoperation demonstrations or random exploration), we train Slot Attention, the slot feature decoder, and the router by

$$
\mathcal{L}_{S1} = \mathcal{L}_{\text{slot}} + \lambda_{\text{slow}}\mathcal{L}_{\text{slow}} + \lambda_{\text{route}}\mathcal{L}_{\text{route}} + \lambda_{\text{div}}\mathcal{L}_{\text{div}},
$$

with (λ<sub>slow</sub>, λ<sub>route</sub>, λ<sub>div</sub>) = (0.5, 0.05, 0.05). We use AdamW (lr 3 × 10⁻⁴, wd 10⁻⁴), batch size 256, 100 K steps (~2 h on one A5000). No proprioception, action, or reward information is used at this stage — the representation is task-agnostic.

**Stage 2 — TD-MPC2 online learning.** DINOv2 remains frozen; Slot Attention and the router are fine-tuned at learning-rate 3 × 10⁻⁵ (an order of magnitude below the newly-initialized binding memory and TD-MPC2 head at 3 × 10⁻⁴). Every gradient update optimizes

$$
\boxed{\;\mathcal{L}_{S2} = \mathcal{L}_{\text{TDMPC2}} + \sum_{i \in \{\text{slot,slow,route,div,pred,align,swap}\}} \lambda_i\, \mathcal{L}_i.\;}
$$

Slot-swap augmentation is applied stochastically with probability p<sub>swap</sub> = 0.3 per sample per batch, batched via a random within-batch permutation for compute efficiency. We use the replay-buffer schedule of TD-MPC2 unchanged: 1 M-transition buffer, one gradient step per environment step, 5 target-Q ensemble heads.

Algorithm 1 gives the full Stage-2 update loop.

**Algorithm 1**: HippoAct Stage-2 update
```
Input:  batch B = {(o_{t-T+1:t}^{(i)}, q_{t-T+1:t}^{(i)}, a_t^{(i)}, r_t^{(i)}, o_{t+1}^{(i)}, q_{t+1}^{(i)})}_{i=1}^{N}
        encoder E_ξ = (dino, slot_attn, router, slot_dec, bind),  world model W_θ (TD-MPC2)

  1. P_t         = dino(o_t)                             # frozen forward
  2. S_t         = slot_attn(P_t)                        # (N, K, d_s)
  3. g_t         = router(S_t;  τ_r)                     # Gumbel STE
  4. S_t^fg      = S_t ⊙ g_t^fg;  S_t^bg = S_t ⊙ g_t^slow
  5. c_t         = bind(S_{t-T+1:t}^fg, q_{t-T+1:t})     # temporal window
  6. z_t         = MLP([flatten(S_t^fg), q_t, c_t])
  7. L_TDMPC2    = TD-MPC2 losses on (z_t, a_t, r_t, z_{t+1})
  8. L_recon aux = L_slot + λ_slow L_slow + λ_route L_route + λ_div L_div
  9. L_bind  aux = λ_pred L_pred + λ_align L_align
 10. if U(0,1) < p_swap:
       π ← RandomPerm(N)
       Ŝ_t = (S_t^fg) ⊕ HungarianAlign(S_t^bg[π], S_t^bg)
       c̃_t = bind(...);  π̃ = policy(z̃_t)
       L_swap = ||c̃_t − sg(c_t)||² + KL(π̃ ‖ sg(π))
     else L_swap = 0
 11. L = L_TDMPC2 + L_recon aux + L_bind aux + λ_swap L_swap
 12. AdamW step on L
 13. Anneal τ_r ← max(0.3, τ_r · 0.9995)
Return: updated (E_ξ, W_θ)
```

Hyperparameter values (λ, learning rates, network dimensions) are collected in Table II.

**Table II — Hyperparameters** *(→ Section IV Table)*

| Symbol | Value | Symbol | Value | Symbol | Value |
|---|---|---|---|---|---|
| K (slots) | 16 | d<sub>s</sub> | 128 | d<sub>c</sub> | 128 |
| T (window) | 4 | τ<sub>r</sub><sup>init</sup> | 1.0 → 0.3 | τ<sub>c</sub> | 0.1 |
| ε (action pos) | 0.05 | p<sub>swap</sub> | 0.3 | H (MPPI) | 5 |
| λ<sub>slot</sub> | 1.0 | λ<sub>slow</sub> | 0.5 | λ<sub>route</sub> | 0.05 |
| λ<sub>div</sub> | 0.05 | λ<sub>pred</sub> | 0.5 | λ<sub>align</sub> | 0.1 |
| λ<sub>swap</sub> | 0.3 | lr (new modules) | 3e-4 | lr (fine-tune) | 3e-5 |
| batch | 256 | buffer | 1 M | update ratio | 1 |

---

## III.J  Complexity, Memory, and Deployment

At training time HippoAct's incremental cost over TD-MPC2 is dominated by the frozen DINOv2 forward pass (~1 ms per frame on A5000) and Slot Attention (~2 ms). Peak GPU memory during Stage-2 training is 11.4 GB per rank at batch 256 — under half of an A5000. At inference the disentangled representation z<sub>t</sub> is 256-D, whereas a 128 × 128 × 3 raw image consumes 49 kB per frame; a 1 M-transition replay buffer therefore drops from ~46 GB (raw) to ~200 MB (representation), a **228× memory reduction**. Fig. 8 reports memory scaling with frame stack length and buffer size.

At real-robot deployment we run the frozen DINOv2 on-board (14 ms perception + 3 ms planning at horizon 5 = 17 ms per step, ≈ 58 Hz on one A5000, sufficient for closed-loop manipulation).

---

## III.K  Summary of Contributions in Context

We advance four novel elements over prior representation-learning approaches for robot RL:

1. Object-centric background/foreground decomposition using frozen DINOv2 + Slot Attention with a Gumbel router supervised by *per-frame spatial connectivity* — motivated by our finding that the intuitive temporal-invariance criterion is systematically inverted for object-centric encoders (vs. threshold-based [prior IJCAI], pixel VAE [Iso-Dream], or single-vector RSSM [DreamerV3], all of which rely on temporal signals).
2. Cross-modal binding of fast slots with proprioception via a Transformer memory that is explicitly *actionable* (predicts next proprioception, aligned by action cosine similarity).
3. Slot-space background swap augmentation that is label-free, texture-library-free, and enforces policy-consistency directly.
4. Integration into a TD-MPC2 planner, demonstrating for the first time that structured disentangled representations can *outperform* the pixel-encoder default of a state-of-the-art model-based RL algorithm on both sample efficiency and background-robustness metrics.
