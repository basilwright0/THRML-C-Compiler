# Self-Timed Thermodynamic Pipelines: An Autonomous Stochastic Clock for Host-Free Chained EBM Execution on p-bit Hardware

**Status:** working outline (draft v1, 2026-05-17)

**Thesis (one sentence, defend verbatim):** An autonomous stochastic clock — realized as a
p-bit subsystem with no global clock — enables closed-loop, host-free, single-fabric chaining
of EBM stages, converting the open-loop host-driven DTM scheduler into a data-responsive
pipeline that is simultaneously area-efficient and I/O-efficient.

**Target venue:** *npj Unconventional Computing* (primary) or a computer-architecture venue;
arXiv preprint first to stake priority.

**How to use this outline:**
- Section 4 is the paper. Section 3 is the enabling mechanism; Section 6.1 is the
  make-or-break result.
- If 6.1 shows host overhead is negligible in realistic regimes, pivot the headline to
  6.2–6.3 (adaptive compute + robustness). Build experiments so either outcome is publishable.
- The explicit non-claims box (Section 1) and Section 7.2 are not optional — they pre-empt the
  identified rejections (DTM anticipation, adaptive-MCMC overlap, generic dataflow).

---

## Abstract
- Problem: p-bit TSUs are stateless samplers; expressivity needs chained EBMs (DTMs), but
  DTM's step schedule is open-loop and host-driven.
- Contribution: an autonomous on-chip stochastic clock + the host-free single-fabric
  architecture it enables.
- Headline result placeholder: *X× lower per-sample energy / Y% I/O overhead eliminated at
  Z× area of multiplexed DTM, with graceful degradation under timing jitter.*

## 1. Introduction
- Why multi-stage (DTM) pipelines are necessary on stateless samplers.
- The unexamined cost: inter-stage orchestration (advance decision, weight reload, re-clamp)
  is host-driven and open-loop.
- Precise thesis.
- **Explicit non-claims box.** We do NOT claim: stateful p-bit pipelines are new;
  clamp-based state transfer is new; adaptive stopping is new generically.
- Contributions list: (i) the autonomous stochastic clock mechanism; (ii) the host-free
  single-fabric architecture; (iii) a re-derived cost model; (iv) a 3-way evaluation;
  (v) honest regime characterization (strong vs marginal advantage).

## 2. Background & Related Work (positioned as differentiation, not survey)
- 2.1 p-bit sampling & block/chromatic Gibbs; the TSU/THRML model.
- 2.2 Camsari invertible logic — combinational, stateless; the sequencing gap it leaves open.
- 2.3 **DTM step scheduler (primary baseline)** — open-loop, host-driven, fixed K_mix, two
  deployment modes (temporally-multiplexed vs spatially-unrolled).
- 2.4 MCMC scheduling: scan-order, lifted/non-reversible, adaptive/early-stopping — *novelty
  is the autonomous on-chip realization, not adaptive stopping per se*.
- 2.5 Host-round-trip avoidance (dataflow/PIM/fusion) — generic principle; ours is
  sampler-specific.
- 2.6 Thermodynamic cost of timekeeping (Barato–Seifert, TUR, biochemical Markov clocks) —
  grounding for 3.6.

## 3. The Autonomous Stochastic Clock (mechanism)
- 3.1 Requirements: sequencing + latching + stage isolation; why a clock alone is
  insufficient.
- 3.2 Clock as a p-bit subsystem (bio-inspired oscillator motif; self-timed; no global clock).
- 3.3 Closed-loop advance: stochastic settling/convergence detection drives stage transition.
- 3.4 Transition protocol: clamp -> settle -> latch -> release.
- 3.5 Stage isolation / directivity to prevent back-diffusion into completed stages.
- 3.6 Clock cost/precision analysis vs the Barato–Seifert bound.

## 4. Host-Free Locally-Chained Architecture (the headline)
- 4.1 Single-fabric temporal reuse with **on-chip** weight reprogramming.
- 4.2 Clock-sequenced reload/re-clamp with no host in the loop.
- 4.3 Comparison table vs both DTM modes on {control authority, I/O, area}.
- 4.4 Cost model: re-derive against DTM's E = T·K_mix·L²·E_cell — which terms are eliminated;
  explicit area model for the ~T× tradeoff (stated, not hidden).

## 5. Methodology
- 5.1 Platform: THRML/JAX, p-bit model, hardware cost model + parameters.
- 5.2 Baselines: (1) temporally-multiplexed host-driven DTM, (2) spatially-unrolled DTM,
  (3) ours.
- 5.3 Workloads: one DTM-favorable generative task (fair to the baseline) + one
  combinatorial/QUBO pipeline (generality).
- 5.4 Metrics: per-sample energy & latency **decomposed into sampling vs orchestration**,
  sample quality, sweeps consumed, pbit area, robustness under injected stage-timing jitter.
- 5.5 Ablations: fixed vs adaptive advancement; clock autonomy on/off; stage-isolation
  on/off.

## 6. Results
- 6.1 3-way energy/latency decomposition; **crossover T/L where host overhead becomes
  non-negligible** (defines the strong-advantage regime).
- 6.2 Adaptive compute: total sweeps vs fixed K_mix at equal sample quality.
- 6.3 Robustness: quality vs timing-jitter, ours vs host-driven.
- 6.4 Area–I/O tradeoff curve across T (ours vs spatial-unroll).
- 6.5 Empirical clock cost/precision vs theoretical bound.

## 7. Discussion
- 7.1 Honest regime characterization: where the advantage is strong vs marginal.
- 7.2 Threats to validity: published-state framing (Extropic may have unpublished timers);
  adaptive-MCMC overlap; simulation != silicon.
- 7.3 Generalization beyond denoising chains (arbitrary inter-stage logic) — preliminary
  evidence.

## 8. Limitations & Future Work
- Area/depth cap; silicon validation; clock subsystem overhead; workload breadth.

## 9. Conclusion

## Key references
- Extropic, "An efficient probabilistic hardware architecture for diffusion-like models,"
  arXiv:2510.23972.
- THRML (Extropic) — docs.thrml.ai; github.com/extropic-ai/thrml.
- Camsari, Faria, Sutton, Datta, "Stochastic p-Bits for Invertible Logic," PRX 7, 031014
  (2017).
- Lucas, "Ising formulations of many NP problems" (2014).
- Lifted / non-reversible MCMC; adaptive Gibbs samplers.
- Barato & Seifert, "Cost and Precision of Brownian Clocks," PRX 6, 041053 (2016);
  "Measuring the Thermodynamic Cost of Timekeeping," PRX 11, 021029 (2021).
- "Robust oscillations in multi-cyclic Markov state models of biochemical clocks,"
  J. Chem. Phys. 152, 055101 (2020).
