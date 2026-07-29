# `hippoact` branch

This branch is the ICRA-targeted redesign of the original IJCAI "Learning from
Memory" work. The original Atari-focused code (`train.py`, `atari_wrappers.py`,
`models/`) is retained on this branch unchanged for reference.

## Layout

- `hippoact/` — new implementation. Object-centric slow/fast decomposition +
  cross-modal binding memory + slot-space background swap augmentation, on a
  frozen DINOv2 backbone. See `hippoact/README.md` for server setup and how
  to run Stage-1 pretraining.
- `docs/` — paper-ready design documents (Method §III, Experiments §IV,
  reviewer defense sheet, testing strategy, real-robot task specs, baseline
  recommendation, and the top-level ICRA redesign proposal).

## Status

- Phase 1 (encoder + Stage-1 pretraining + tests) — done.
- Phase 2 (TD-MPC2 integration + DMC/Distracting-Suite env wrappers + real
  robot deployment) — pending choice of arm SDK.

Start reading here: `docs/paper_narrative_organization.md` for the four P1-P4
pain-point mapping, then `hippoact/README.md` to run code.
