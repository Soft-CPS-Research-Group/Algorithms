# Research notes

## Current campaign status — 2026-09-25

The author has retired the previous experimental line. Its campaign plans,
queue of follow-up variants, temporal splits, budgets and promotion rules do
not define the next campaign. At the author's subsequent request, the corrected Portugal dataset evaluation
contract is now recorded in [Portugal evaluation v2](../docs/portugal_evaluation_protocol_v2.md).
It prepares paired scorecards and confirmation inputs without reopening the old
training campaign or launching jobs.

The implementation, dataset contracts, collected evidence and lessons remain
available. Retiring a research campaign does not disable software regression
tests or discard their coverage. It also does not imply that remote jobs were
canceled: no remote execution state was changed during this documentation
cleanup.

The previous untracked campaign-planning package and Transformer review were
moved to an external local archive. Two result-table drafts were retained as
historical illustrations, with explicit warnings that their performance values
are hypothetical. Their former `THESIS-EXP-V1` plan is no longer active.

## Retained context

- [Annual REC dataset contract and variants](datasets/annual-rec-suite.md).
- [Observed provisional TI-MARL results and lessons](ti-marl/provisional-results-2026-09.md).
- [Historical proposed result tables](thesis-experiments/proposed-thesis-results-for-supervisors.md)
  and their [short summary](thesis-experiments/proposed-thesis-results-supervisor-summary.md):
  illustrative performance values, not measured results or a future test plan.
- [TI-MARL design decisions](ti-marl/decision-log.md) and the existing compiler,
  runtime, interface and verification notes: implementation knowledge retained.

The old [experiment plan](ti-marl/experiment-plan.md) and
[protocol](ti-marl/experiment-protocol-v1.md) remain historical references for
interpreting existing artefacts. They must not be resumed automatically.
