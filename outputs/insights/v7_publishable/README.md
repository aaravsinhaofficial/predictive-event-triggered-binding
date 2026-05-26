# V7 Publishable Insights Results

This folder contains the final distilled results for the leakage-controlled V7
experiment. The appropriate framing is an Insights-style failure-analysis paper,
not a clean positive ETB-superiority paper.

## Files

- `report.md`: generated summary report.
- `all_results.csv`: all evaluated runs and metrics.
- `etb_vs_baselines.csv`: ETB margins against baselines.
- `rt_supervision_comparison.csv`: main ETB versus RT-supervised upper-bound ETB.
- `naturalstories_uncertainty.csv`: bootstrap and permutation diagnostics.
- `figures/`: PNG/PDF figures for the main paper story.

## Bottom Line

The results do not support the claim that ETB is broadly better than strong
dynamic baselines. They do support a methodological warning:

> Apparent reading-time alignment from neural gates is fragile. It can be
> inflated by RT supervision, rivaled by shallow punctuation/boundary heuristics,
> and decoupled from broad syntactic or filler-gap improvements.

## Clean Main Results

Mean clean main results across available seeds:

| Variant | BLiMP | SyntaxGym | FillerGap | Natural Stories flex delta R2 | Natural Stories story-CV flex |
|---|---:|---:|---:|---:|---:|
| ETB | 0.916 | 0.772 | 0.992 | 0.0046 | 0.0089 |
| generic_dynamic | 0.935 | 0.834 | 0.997 | 0.0034 | 0.0036 |
| punctuation_only | 0.934 | 0.822 | 0.998 | 0.0099 | 0.0073 |

Interpretation:

- ETB has a real but modest Natural Stories gate signal in the clean setup.
- ETB is not better overall: it is weaker than generic dynamic baselines on
  BLiMP, SyntaxGym, and FillerGap.
- `punctuation_only` is the strongest Natural Stories baseline, which is a major
  confound for cognitive-memory interpretations of gate/RT alignment.

## Seed-Level Pattern

ETB is seed-sensitive:

- Seed 23 is weak: ETB loses to `generic_dynamic` on Natural Stories flexible
  delta R2 and is much worse on syntax/filler metrics.
- Seeds 29 and 31 are stronger: ETB beats `generic_dynamic` on Natural Stories
  flexible delta R2 and story-CV flexible delta R2.
- Even in stronger seeds, ETB still loses or ties on BLiMP/SyntaxGym/FillerGap.

This supports an instability/failure-mode story rather than a clean positive
mechanism story.

## RT-Supervised Upper Bound

RT-supervised ETB increases Natural Stories delta R2 but does not consistently
preserve syntax/filler performance.

Average RT-supervised minus main ETB:

| Metric | Mean Change |
|---|---:|
| BLiMP | -0.0074 |
| SyntaxGym | -0.0405 |
| FillerGap | -0.0013 |
| Natural Stories flexible delta R2 | +0.0065 |
| Natural Stories story-CV flexible delta R2 | +0.0360 |

Interpretation:

- RT supervision reliably inflates reading-time alignment.
- This is useful as an upper bound or leakage stress test.
- It is not evidence for the main unsupervised ETB hypothesis.

## Auxiliary Ablation

The seed-23 `no_aux` / `no_contrastive` ablations collapse:

- BLiMP drops to about 0.600.
- SyntaxGym drops to about 0.299.
- FillerGap drops to about 0.473.
- Natural Stories flexible delta R2 drops to near zero.

Interpretation:

- ETB's syntax/filler behavior is largely carried by the auxiliary
  contrastive/supervised training signal.
- It should not be described as fully spontaneous syntactic emergence.

## Best Paper Framing

The strongest paper claim is:

> We find that event-triggered gates can show modest reading-time alignment under
> leakage-controlled evaluation, but this alignment is unstable, sensitive to
> auxiliary training, and matched or exceeded by simple boundary-based baselines.
> RT-supervised gates produce larger effects, demonstrating how easily
> psycholinguistic alignment claims can be inflated.

Avoid claiming:

> ETB explains human reading times beyond surprisal.

Prefer:

> RT-alignment claims for neural gates require leakage checks, flexible surprisal
> controls, and shallow boundary baselines.

## Recommended Figures

- `figures/clean_naturalstories_delta_r2.png`: clean Natural Stories gate signal.
- `figures/etb_minus_strong_baselines.png`: ETB margins against strong baselines.
- `figures/rt_supervision_upper_bound.png`: RT-supervision inflation.
- `figures/auxiliary_ablation_collapse.png`: auxiliary dependence.
- `figures/naturalstories_uncertainty.png`: bootstrap intervals.

## Submission Read

This is a credible negative-results / methodological-warning package for
Insights. It is not a positive ETB-superiority package for a main-conference
result unless the claim is substantially weakened and centered on diagnostic
lessons.
