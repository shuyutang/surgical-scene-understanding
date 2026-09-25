# Phase 1 segmentation, test evaluation

Checkpoint `runs/seg_unet_r34_20260924-230723/best.pt` (epoch 55, softmax temperature 1.560 fit on dev). Internal: 1135 frames / 10 patients. External (EndoVis18, zero-shot): 3232 frames / 19 sequences. 95% CIs: patient/sequence-clustered bootstrap.

## Pre-specified criteria

| ID | Endpoint | Estimate [95% CI] | Criterion | Result |
|---|---|---|---|---|
| P1 | Instrument Dice (internal) | 0.928 [0.909, 0.941] | lower bound >= 0.85 | PASS |
| P2 | Tip Dice (internal) | 0.829 [0.794, 0.856] | lower bound >= 0.7 | PASS |
| S1 | Tip boundary F 3px (internal) | 0.683 [0.644, 0.714] | lower bound >= 0.6 | PASS |
| S2a | Liver Dice (internal) | 0.850 [0.817, 0.875] | lower bound >= 0.6 | PASS |
| S2b | Stomach Dice (internal) | 0.814 [0.735, 0.866] | lower bound >= 0.6 | PASS |
| S3 | Pixel ECE, 32 classes (internal) | 0.021 [0.007, 0.046] | upper bound <= 0.05 | PASS |
| S4 | Instrument Dice zero-shot (external) | 0.778 [0.748, 0.810] | lower bound >= 0.7 | PASS |

## Robustness subgroups (R1: within 0.10 of overall instrument Dice)

| Subgroup | Frames | Patients | Instrument Dice [95% CI] | Δ vs overall | R1 |
|---|---|---|---|---|---|
| blurry | 288 | 10 | 0.877 [0.843, 0.903] | -0.051 | PASS |
| glare | 274 | 10 | 0.913 [0.896, 0.928] | -0.015 | PASS |
| hazy | 206 | 10 | 0.911 [0.883, 0.931] | -0.017 | PASS |
| dark | 241 | 10 | 0.936 [0.923, 0.947] | +0.008 | PASS |

## Harmonized parts (exploratory)

| Set | Tip | Wrist | Shaft |
|---|---|---|---|
| internal | 0.829 [0.794, 0.856] | 0.907 [0.898, 0.915] | 0.879 [0.844, 0.902] |
| external | 0.571 [0.522, 0.610] | 0.716 [0.667, 0.761] | 0.637 [0.586, 0.682] |

## Anatomy Dice (internal)

| Structure | Dice [95% CI] |
|---|---|
| Liver | 0.850 [0.817, 0.875] |
| Stomach | 0.814 [0.735, 0.866] |
| Pancreas | 0.719 [0.623, 0.795] |
| Spleen | 0.505 [0.312, 0.610] |
| Gallbladder | 0.663 [0.468, 0.862] |

## All classes (internal, pooled; mIoU over present classes = 0.699)

| Class | Dice | IoU | Boundary F | HD95 (px @640×512) | GT pixels |
|---|---|---|---|---|---|
| Background | 0.865 | 0.762 | 0.830 | 35.0 | 29,782,337 |
| HarmonicAce_Head | 0.764 | 0.619 | 0.731 | 12.7 | 1,723,683 |
| HarmonicAce_Body | 0.914 | 0.842 | 0.787 | 7.0 | 5,686,811 |
| MarylandBipolarForceps_Head | 0.888 | 0.799 | 0.725 | 11.0 | 3,580,627 |
| MarylandBipolarForceps_Wrist | 0.890 | 0.802 | 0.726 | 8.8 | 2,507,509 |
| MarylandBipolarForceps_Body | 0.824 | 0.700 | 0.627 | 13.1 | 2,180,615 |
| CadiereForceps_Head | 0.851 | 0.741 | 0.734 | 10.6 | 1,793,399 |
| CadiereForceps_Wrist | 0.845 | 0.732 | 0.686 | 10.0 | 1,132,944 |
| CadiereForceps_Body | 0.816 | 0.689 | 0.672 | 9.5 | 2,549,659 |
| CurvedAtraumaticGrasper_Head | 0.480 | 0.316 | 0.365 | 45.0 | 1,437,793 |
| CurvedAtraumaticGrasper_Body | 0.771 | 0.628 | 0.560 | 26.4 | 4,006,743 |
| Stapler_Head | 0.837 | 0.720 | 0.602 | 22.0 | 3,796,097 |
| Stapler_Body | 0.845 | 0.731 | 0.572 | 30.4 | 2,480,669 |
| MediumLargeClipApplier_Head | 0.862 | 0.757 | 0.617 | 13.6 | 1,231,314 |
| MediumLargeClipApplier_Wrist | 0.862 | 0.757 | 0.585 | 14.4 | 1,047,531 |
| MediumLargeClipApplier_Body | 0.657 | 0.490 | 0.462 | 22.0 | 618,687 |
| SmallClipApplier_Head | 0.909 | 0.833 | 0.820 | 7.0 | 449,282 |
| SmallClipApplier_Wrist | 0.942 | 0.890 | 0.762 | 7.1 | 1,092,914 |
| SmallClipApplier_Body | 0.833 | 0.714 | 0.687 | 7.1 | 691,292 |
| SuctionIrrigation | 0.755 | 0.606 | 0.611 | 6.0 | 1,202,390 |
| Needle | 0.642 | 0.473 | 0.701 | 29.6 | 216,315 |
| Endotip | 0.959 | 0.921 | 0.915 | 2.2 | 1,489,501 |
| Specimenbag | 0.894 | 0.809 | 0.541 | 52.6 | 11,044,145 |
| DrainTube | 0.879 | 0.784 | 0.767 | 8.9 | 1,565,530 |
| Liver | 0.850 | 0.739 | 0.621 | 56.1 | 34,472,443 |
| Stomach | 0.814 | 0.687 | 0.543 | 44.6 | 33,370,555 |
| Pancreas | 0.719 | 0.561 | 0.423 | 53.1 | 14,428,041 |
| Spleen | 0.505 | 0.338 | 0.405 | 30.6 | 1,092,921 |
| Gallbladder | 0.663 | 0.496 | 0.487 | 32.5 | 9,781,173 |
| Gauze | 0.953 | 0.910 | 0.801 | 13.0 | 39,917,650 |
| TheOther_Instruments | 0.853 | 0.744 | 0.560 | 39.1 | 11,534,644 |
| TheOther_Tissues | 0.880 | 0.786 | 0.604 | 76.4 | 144,011,586 |
