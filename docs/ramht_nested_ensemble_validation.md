# Strict Nested Validation of RAMHT + XGBoost

## Objective

This experiment removes the test-selection bias in the earlier fixed-weight
RAMHT + XGBoost ensemble. For every outer split and label, both models are
trained without access to the outer-test observations. The RAMHT blend weight
is selected only on the corresponding outer-validation observations and then
frozen before evaluation on outer-test.

The evaluation contains 26 outer splits per label:

- 3 repeated-random splits
- 23 chromosome-holdout splits

## Procedure

For each label and outer split:

1. Train RAMHT and XGBoost using outer-train only.
2. Generate predictions for outer-validation and outer-test.
3. On outer-validation, scan RAMHT weights from 0.0 to 1.0 in steps of 0.1.
4. Select the weight with the highest validation Pearson correlation. Ties
   prefer the lower RAMHT weight.
5. Apply the frozen weight once to outer-test.
6. Compare the blend and XGBoost on the same outer-test observations.

This produces 104 paired test comparisons: 26 splits for each of 4 labels.

## Main Pearson Results

Results below aggregate all 26 outer-test splits for each label. The confidence
interval is a paired bootstrap 95% CI over split-level Pearson differences.

| Label | XGBoost mean | Nested blend mean | Mean delta | Wins / losses / ties | 95% CI | Wilcoxon p | Sign-permutation p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gene late | 0.49915 | 0.50074 | +0.00159 | 15 / 9 / 2 | [-0.00345, 0.00575] | 0.1615 | 0.5418 |
| Gene total | 0.54074 | 0.55687 | **+0.01613** | **23 / 3 / 0** | **[0.01042, 0.02200]** | **9.15e-06** | **2.00e-05** |
| Exon late | 0.55613 | 0.56003 | **+0.00391** | **19 / 6 / 1** | **[0.00134, 0.00661]** | **0.00493** | **0.00706** |
| Exon total | 0.77295 | 0.77457 | +0.00162 | 17 / 8 / 1 | [-0.00005, 0.00316] | 0.04797 | 0.06161 |

The strongest result is gene total: the improvement is stable across splits,
its bootstrap interval excludes zero, and both paired tests are significant.
Exon late also shows a smaller but statistically supported improvement. Gene
late does not show convincing evidence of a Pearson improvement. Exon total is
borderline: the Wilcoxon result is below 0.05, but the bootstrap interval and
permutation test do not exclude the null.

## Selected Weights

Validation-selected RAMHT weights vary across splits, showing why a strict
per-split selection procedure is preferable to a globally test-selected weight.

| Label | Mean RAMHT weight | Median | Min | Max |
|---|---:|---:|---:|---:|
| Gene late | 0.258 | 0.25 | 0.0 | 0.6 |
| Gene total | 0.431 | 0.40 | 0.1 | 0.7 |
| Exon late | 0.258 | 0.25 | 0.0 | 0.5 |
| Exon total | 0.269 | 0.30 | 0.0 | 0.4 |

Gene total consistently assigns more weight to RAMHT than the other labels,
which agrees with its larger outer-test gain.

## Residual Correlation

Mean outer-test residual correlations between RAMHT and XGBoost are high:

| Label | Mean residual correlation |
|---|---:|
| Gene late | 0.946 |
| Gene total | 0.934 |
| Exon late | 0.939 |
| Exon total | 0.928 |

Therefore, RAMHT and XGBoost are not broadly independent error models. The
ensemble gain is best interpreted as a modest correction from complementary
sequence information, especially for gene total, rather than as evidence of
strongly orthogonal representations.

## Conclusion

Strict nested validation confirms that RAMHT contributes useful complementary
signal for **gene total** and **exon late**. The evidence is strongest for gene
total, where the blend wins 23 of 26 splits and improves mean Pearson by
0.01613 with a confidence interval fully above zero.

The earlier fixed 0.2-weight result should not be presented as an unbiased
performance estimate. The nested results are the appropriate primary numbers.
For gene late and exon total, the evidence is insufficient or borderline, so
claims of universal improvement across all labels should be avoided.

## Reproduction

Primary outputs:

- `data/processed/ramht_nested_outer_predictions.tsv`
- `data/processed/nested_ensemble_xgboost_predictions.tsv`
- `data/processed/nested_ramht_xgboost_weight_selections.tsv`
- `data/processed/nested_ramht_xgboost_metrics.tsv`
- `data/processed/nested_ramht_xgboost_summary.tsv`
- `data/processed/nested_ramht_xgboost_statistics.tsv`

Evaluation command:

```bash
PYTHONPATH=src python scripts/evaluate_nested_ramht_xgboost.py \
  --ramht-predictions data/processed/ramht_nested_outer_predictions.tsv \
  --xgboost-predictions data/processed/nested_ensemble_xgboost_predictions.tsv \
  --out-prefix data/processed/nested_ramht_xgboost
```
