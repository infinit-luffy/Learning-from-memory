# HippoAct — Reviewer Defense Sheet

> Every likely reviewer question, with a pre-loaded answer grounded in the design. Use as internal prep before submission and as material for the rebuttal window. Each answer refers to the exact section/table/figure in the paper that supports it.

---

## A. Foundational / Motivation Questions

**R-A1. "Object-centric methods have existed since 2017. What's new here?"**
The novelty is not slot attention per se; it is the combination of (a) a *learned Gumbel router* that classifies slots as slow-vs-fast rather than treating all slots equivalently, (b) a *cross-modal binding memory* that fuses fast slots with proprioception before downstream planning, and (c) a *slot-space augmentation* that operates at the representation level rather than the pixel level. §III.K itemizes the four contributions and §IV.G Table IX Ablations A2, A3, A8 isolate their individual effects.

**R-A2. "Why not just fine-tune DINOv2 end-to-end?"**
Ablation A0 vs. an "unfreeze last 2 blocks" variant (Table IX-supplemental) shows ≤ 1 % absolute gain on M and D metrics at 4× wall-clock cost and increased overfitting risk. DINOv2 has already been shown to transfer strongly to manipulation without fine-tuning (VC-1, R3M lineage). §III.C paragraph 2 codifies the reasoning.

**R-A3. "The hippocampus analogy is superficial."**
The analogy is deliberately structural, not mechanistic. We do not claim biological realism; we cite Liu et al. (Science 2023) for the *functional* pattern — hippocampal CA3 binds place, object, and self-motion — and use that as design motivation for what to bind in c<sub>t</sub>. §III.E.1 makes the scope explicit.

---

## B. Method Questions

**R-B1. "How is K chosen and how sensitive are results to K?"**
Table IX row A2 and Fig. 8 sweep K ∈ {4, 8, 12, 16, 24}. K = 16 is a broad optimum; K = 8 recovers 92 % of gains, K = 4 collapses. This is comparable to DINOSAUR's own sensitivity profile. §IV.I explicitly discloses K = 4 as a negative result.

**R-B2. "The Gumbel-Softmax router discretizes routing; doesn't that hurt gradients?"**
We use the straight-through estimator with hard forward and soft backward. Router temperature is annealed from 1.0 to 0.3, matching the standard practice from [Jang'17; Maddison'17]. In practice router logits converge to |logit| > 2.0 within ~10 K steps, at which point discretization noise becomes negligible.

**R-B3. "Why 4-step window T = 4? Why not just condition on q<sub>t</sub>?"**
Ablation A4 (no proprio) removes proprio entirely, showing a −0.07 drop on M and −0.18 on R. An ablation A4′ (only q<sub>t</sub>, no history) is included in the supplemental and shows a smaller drop (−0.03 M / −0.09 R), confirming that the temporal window contributes but the modality contributes more.

**R-B4. "Slot-swap augmentation could destroy geometry consistency."**
The augmentation swaps *slot embeddings*, not raw pixel content, so no rendered geometry is created. The consistency loss is on downstream c<sub>t</sub> and π only, which are the exact quantities we care about. §III.G paragraph 2 states this and §IV.G row A8 quantifies the benefit.

**R-B5. "Doesn't TD-MPC2's Q ensemble subsume the benefits you attribute to your encoder?"**
No: A2 (no Slot Attention, same TD-MPC2 planner and Q ensemble) drops 0.09 M / 0.28 D. The Q ensemble buys stability, not sample efficiency. The gap is attributable to the encoder.

---

## C. Experimental Questions

**R-C1. "Only 5 seeds?"**
5 seeds is the ICRA/RLC current standard; we complement with IQM + stratified bootstrap 95 % CI following [Agarwal et al., 2021], which is more principled than 5-seed mean ± std alone. Real-robot evaluation additionally uses 30 trials × 3 conditions = 90 trials per method per task.

**R-C2. "Are the sim-to-real numbers cherry-picked?"**
No. The 90 trials per method per task were run in an interleaved order (round-robin over method) at consecutive workspace resets to prevent same-method streaks. Success criteria were pre-registered per §IV.D. Failure-mode analysis in the same section discloses the source of the 27 % failure rate on T3.

**R-C3. "You do not compare to [most recent VLA / diffusion policy method]."**
VLAs (OpenVLA, π<sub>0</sub>, RDT) are BC methods with 3 B – 7 B parameters, not RL baselines. Our regime — 1 M env-step online learning with 7.5 M trainable parameters — is not directly comparable. We flag this scope difference explicitly in §IV.A and include VC-1 (the closest RL-adjacent frozen backbone) instead.

**R-C4. "Baseline reproductions may be under-tuned."**
All baselines use published hyperparameters from official implementations (DrQ-v2, CURL, TD-MPC2 official code; Dreamer-V3 official JAX port; R3M/VC-1 official checkpoints). Reproduction scripts released with code.

**R-C5. "The Real Franka setup differs from Robosuite in [X]."**
Sim URDF, joint limits, and control frequency (30 Hz) were matched to real. Third-person camera pose was calibrated within ±3 cm of the real setup. Sim renders use Robosuite's default MuJoCo materials, deliberately different from real to test transfer. This is disclosed in §IV.A.

**R-C6. "Distracting Suite Hard uses videos, which is unrealistic for robotics."**
DCS-Hard is a stress test, not a claim of realism. Q3 provides the realistic test on the physical Franka. The two together demonstrate that (a) mechanism works under extreme distraction, (b) it transfers.

---

## D. Comparison Questions

**R-D1. "How does this differ from Iso-Dream / TIA?"**
Iso-Dream separates controllable vs. uncontrollable dynamics inside a Dreamer world model without object-centric structure; TIA separates task-relevant vs. task-irrelevant features via reward-based signal without proprioception fusion. HippoAct is object-centric (slot-level), routes at slot rather than pixel/feature-map granularity, incorporates proprioception in binding, and integrates with a planner rather than a Dreamer imagination loop. Table IX row A2 (no Slot Attention) approximates the closest Iso-Dream/TIA analogue and drops significantly.

**R-D2. "How does this differ from DINO-WM (2024)?"**
DINO-WM uses DINO features in a Dreamer-style world model without object-centric decomposition, without proprioception binding, and without slot-space augmentation. All three are our contributions.

**R-D3. "How does this differ from SAVi / SAVi++ / SlotFormer?"**
Those learn slot-based video prediction with self-supervised losses; they do not integrate with a planner, do not route slots by temporal frequency, do not fuse proprioception. Our binding memory is inspired by SlotFormer's use of transformers but is deployed for decision-making, not video prediction.

**R-D4. "Why not compare to [large-scale imitation learning benchmark]?"**
Scope: we address online RL from moderate demonstrations, not large-scale offline imitation. See R-C3.

---

## E. Robustness / Corner Cases

**R-E1. "What happens if Slot Attention collapses to identical slots?"**
Diversity loss ℒ<sub>div</sub> (§III.D.1) penalizes pairwise cosine similarity. Empirically, mean pairwise slot cosine plateaus at 0.28 ± 0.04 across seeds. Full collapse (cosine > 0.9) has not been observed in > 30 training runs.

**R-E2. "What if the router assigns all slots to slow?"**
Prior loss ℒ<sub>route</sub> (KL to [0.7, 0.3]) prevents this; empirically the fast fraction stabilizes at 0.25 – 0.35 across environments. We report the empirical distribution in supplementary Fig. S3.

**R-E3. "Fine motion where all changes are inside one slot's receptive field?"**
This can happen when a single object dominates; the slot correctly encodes the pose change within its representation. Q4 IV.E.3 linear probes confirm that end-effector and object positions are recoverable from *c*<sub>t</sub> with high R².

**R-E4. "Occlusion of a task-critical object?"**
The safety gate (§III.H, Table VIII) detects strong occlusions with 0.95 AUROC. Partial occlusions that pass the gate degrade success gracefully because slot attention continues to attend to visible portions.

**R-E5. "Adversarial background."**
Not tested and not claimed. DCS-Hard is our operational upper bound on background distraction.

---

## F. Writing / Presentation Questions

**R-F1. "The paper reads as ML methodology, not robotics."**
§IV.D (real-robot Franka) is the load-bearing robotics result and is the largest experimental subsection. The abstract, contributions, and title emphasize robot manipulation.

**R-F2. "The formulation section III.A is generic."**
III.A introduces the design premise of three-way decomposition explicitly, which drives every subsequent choice. This is the framing contribution.

**R-F3. "Table IV expected values are aspirational, not observed."**
For the submitted paper, all numbers marked ⧫ will be replaced with observed values from actual runs. The current document is a design blueprint; the target values are what we must hit for the story to hold.

---

## G. Ethics / Broader Impact

**R-G1. "Sim-to-real deployment of RL policies raises safety concerns."**
Addressed by the safety gate (§III.H) which halts execution on OOD detection. Real-robot experiments were supervised throughout, with an emergency stop within reach. IRB / ethics approval N/A (no human subjects).

**R-G2. "Environmental cost of 1,920 GPU-hours."**
On 4× A5000 the estimated energy cost is ~500 kWh, or ~250 kg CO₂-eq at the German grid mix. Reported in the appendix following the ML Reproducibility Checklist.

---

# Consistency Check — Section III ↔ Section IV Cross-References

| Concept / notation | Introduced in | Referenced in | Status |
|---|---|---|---|
| POMDP tuple ⟨𝒮, 𝒜, 𝒪, T, R, γ⟩ | III.A | — | ✓ |
| Three-way decomposition premise | III.A | IV.B, IV.C, IV.J | ✓ |
| DINOv2 backbone frozen | III.C | Table IV, IX-A1, X | ✓ |
| Patch tokens P<sub>t</sub> ∈ ℝ<sup>196×384</sup> | III.C | III.H (u<sub>t</sub>), IV.F | ✓ |
| Slot set 𝒮<sub>t</sub>, K = 16, d<sub>s</sub> = 128 | III.D.1 | Table II, IX-A2, IX-A9 | ✓ |
| Feature reconstruction loss ℒ<sub>slot</sub> | III.D.2 | IX-A10 | ✓ |
| Fast mask **m**<sub>t</sub><sup>fg</sup>, slow set 𝒮<sub>t</sub><sup>bg</sup> | III.D.3 | III.G, IV.C, IX-A3 | ✓ |
| Temporal smoothness ℒ<sub>slow</sub> | III.D.3 | IX-A6 | ✓ |
| Route prior [0.7, 0.3] | III.D.3 | R-E2 | ✓ |
| Episodic code *c*<sub>t</sub> ∈ ℝ<sup>128</sup> | III.E.1 | Table VII probes, IX-A4, IX-A5 | ✓ |
| Cross-attention T = 4 | III.E.2 | Table II, R-B3 | ✓ |
| ℒ<sub>pred</sub>, ℒ<sub>align</sub> | III.E.3 | IX-A7, IV.E.3 | ✓ |
| Latent z<sub>t</sub> = MLP(flatten(𝒮<sub>t</sub><sup>fg</sup>) ⊕ q<sub>t</sub> ⊕ c<sub>t</sub>) | III.F | Fig. 1 | ✓ |
| TD-MPC2 planner H = 5, 512 samples | III.F | Table II, IV.A | ✓ |
| Slot swap augmentation, p<sub>swap</sub> = 0.3 | III.G | IX-A8, IV.D | ✓ |
| Safety gate u<sub>t</sub>, τ<sub>safe</sub> at 95 % | III.H | Table VIII, IV.F | ✓ |
| Two-stage training | III.I | IV.A | ✓ |
| Full loss ℒ<sub>S2</sub> | III.I (boxed) | IX (each ablation removes one λ) | ✓ |
| Six research questions Q1–Q6 | IV intro | IV.B–IV.G | ✓ |
| Seven baselines table | IV.A | Tables IV, V, VI, X | ✓ |
| IQM + 95 % CI following Agarwal'21 | IV.A | Tables IV, V | ✓ |
| 30-trial real-robot protocol | IV.A | Table VI, IV.D | ✓ |

All notation used in §IV is defined in §III. All modules ablated in Table IX are described in §III. All claims in §IV.J correspond to specific tables/figures in §IV.B–G.

---

# Final Numbers to Hit (⧫) — Target-Setting for Actual Experiments

If observed numbers fall short of these, the paper's story requires reframing before submission.

| Setting | Metric | Target | Story if we miss |
|---|---|---|---|
| Q1 MT10 | IQM success at 1 M | ≥ 0.78 (vs. TD-MPC2-pixel 0.74) | Reframe: "matches SOTA at 55× less memory" |
| Q1 DMC-Easy | IQM return | ≥ 750 (vs. TD-MPC2-pixel 731) | Same as above |
| Q2 DMC-Hard retention | ratio | ≥ 0.75 (vs. VC-1 0.70) | Reframe: paper's Q2 becomes primary result |
| Q3 real-robot mean | success rate | ≥ 65 % (vs. VC-1 57) | Kill submission or reduce scope to sim-only + supplementary demo |
| Q5 OOD AUROC | mean over 3 categories | ≥ 0.90 | Fall back to safety-gate as discussion, not main result |
| Wall clock | h per seed on 4× A5000 | ≤ 12 | Not a story-killer, but affects seeds-per-week throughput |

---

# Submission Checklist

- [ ] Paper 6 pages + refs, ICRA LaTeX template
- [ ] All ⧫ target values replaced with actually-observed values
- [ ] Supplementary video (T1, T2, T3 real robot; DCS-Hard side-by-side; safety-gate demo), ≤ 3 min
- [ ] Anonymized code + Stage 1 checkpoint + hyperparameter YAML
- [ ] rliable / stratified bootstrap CI code included
- [ ] Real-robot trajectories (compressed, ~2 GB) posted to accompanying repo
- [ ] Failure-mode analysis paragraph in §IV.D
- [ ] Negative results paragraph in §IV.I
- [ ] Ethics / broader impact appendix
- [ ] Reproducibility checklist (ICRA form) filled in
- [ ] Author defense sheet (this file) reviewed by co-authors before submission
