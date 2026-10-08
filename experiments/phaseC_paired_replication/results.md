# Frozen COCO paired-data results

[ours] Measurements on 1,000 original-release COCO2017 validation image groups, five captions each. Same encoder; independently sampled evaluation data. T2I uses query-prefixed captions; I2T uses document-prefixed captions.

## Recorded paired retrieval

| representation | direction | queries | gallery | Hit@1 % | Hit@5 % | Hit@10 % | set recall@10 % | overlap@10 | changed identities | boundary ties@1/5/10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native_768 | t2i | 5000 | 1000 | 73.860 | 93.500 | 97.160 | 97.160 | 1.00000 | 0 | {'1': 0, '10': 0, '5': 0} |
| native_768 | i2t | 1000 | 5000 | 84.300 | 94.800 | 98.400 | 80.240 | 1.00000 | 0 | {'1': 1, '10': 4, '5': 3} |
| mrl_256 | t2i | 5000 | 1000 | 72.140 | 92.400 | 96.960 | 96.960 | 0.83630 | 4410 | {'1': 0, '10': 0, '5': 0} |
| mrl_256 | i2t | 1000 | 5000 | 83.600 | 95.000 | 98.100 | 78.500 | 0.83750 | 879 | {'1': 1, '10': 9, '5': 5} |
| mrl_128 | t2i | 5000 | 1000 | 62.500 | 87.060 | 93.820 | 93.820 | 0.64778 | 4939 | {'1': 0, '10': 0, '5': 0} |
| mrl_128 | i2t | 1000 | 5000 | 73.700 | 93.200 | 95.900 | 68.900 | 0.65240 | 987 | {'1': 1, '10': 1, '5': 5} |

[ours] Primary endpoint is T2I Hit@10. I2T Hit@10 and positive-set recall@10 are different multiple-positive endpoints.

## Paired deltas and query changes

| representation | direction | Δ Hit@10 pp | conditional 95% interval | gained queries | lost queries | unchanged | Δ set recall@10 pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mrl_256 | t2i | -0.200 | [-0.540, +0.140] | 28 | 38 | 4934 | -0.200 |
| mrl_256 | i2t | -0.300 | [-1.100, +0.500] | 6 | 9 | 985 | -1.740 |
| mrl_128 | t2i | -3.340 | [-4.060, -2.620] | 27 | 194 | 4779 | -3.340 |
| mrl_128 | i2t | -2.500 | [-3.600, -1.400] | 4 | 29 | 967 | -11.340 |

[ours] Intervals resample 1,000 parent-image units, carrying five captions and paired conditions together, conditional on the fixed gallery. Marginal percentile intervals; no equivalence test or encoder uncertainty.

## Coarse category image-only comparator

| representation | eligible categories | macro P@10 % | Δ native pp | macro chance % |
| --- | --- | --- | --- | --- |
| native_768 | 75 | 48.842 | +0.000 | 3.873 |
| mrl_256 | 75 | 48.125 | -0.717 | 3.873 |
| mrl_128 | 75 | 44.880 | -3.963 | 3.873 |

[ours] Eligibility was frozen at >=10 member images. Each category queries its members against 999 other images; multi-category membership is preserved. This is coarse image-only category utility, not paired-caption relevance.

| category | support | macro eligible | chance % | native % | 256 % | 128 % |
| --- | --- | --- | --- | --- | --- | --- |
| person | 548 | True | 54.755 | 82.445 | 82.737 | 81.734 |
| traffic light | 36 | True | 3.504 | 37.778 | 38.889 | 33.889 |
| fire hydrant | 19 | True | 1.802 | 50.000 | 43.158 | 47.895 |
| stop sign | 16 | True | 1.502 | 33.750 | 35.000 | 29.375 |
| parking meter | 6 | False | 0.501 | 18.333 | 18.333 | 13.333 |
| bench | 43 | True | 4.204 | 32.558 | 31.860 | 28.605 |
| bird | 28 | True | 2.703 | 42.857 | 43.571 | 41.071 |
| cat | 35 | True | 3.403 | 83.429 | 84.000 | 78.571 |
| dog | 40 | True | 3.904 | 53.500 | 50.500 | 39.750 |
| horse | 20 | True | 1.902 | 42.500 | 44.000 | 37.500 |
| bicycle | 28 | True | 2.703 | 21.786 | 20.714 | 20.000 |
| sheep | 13 | True | 1.201 | 63.846 | 68.462 | 62.308 |
| cow | 21 | True | 2.002 | 45.238 | 46.190 | 41.429 |
| elephant | 17 | True | 1.602 | 90.000 | 88.235 | 82.941 |
| bear | 17 | True | 1.602 | 82.941 | 83.529 | 81.765 |
| zebra | 15 | True | 1.401 | 91.333 | 90.000 | 86.667 |
| giraffe | 20 | True | 1.902 | 98.000 | 98.000 | 98.000 |
| backpack | 38 | True | 3.704 | 16.579 | 15.789 | 13.684 |
| umbrella | 29 | True | 2.803 | 15.862 | 16.207 | 15.862 |
| car | 97 | True | 9.610 | 32.268 | 33.299 | 31.134 |
| handbag | 57 | True | 5.606 | 20.877 | 21.228 | 24.386 |
| tie | 25 | True | 2.402 | 31.600 | 31.200 | 26.000 |
| suitcase | 16 | True | 1.502 | 15.000 | 11.250 | 10.625 |
| frisbee | 16 | True | 1.502 | 51.875 | 44.375 | 41.250 |
| skis | 23 | True | 2.202 | 84.783 | 83.913 | 84.348 |
| snowboard | 13 | True | 1.201 | 36.154 | 36.154 | 33.077 |
| sports ball | 31 | True | 3.003 | 33.226 | 34.194 | 32.258 |
| kite | 24 | True | 2.302 | 75.417 | 75.417 | 65.833 |
| baseball bat | 23 | True | 2.202 | 78.261 | 76.957 | 71.304 |
| motorcycle | 32 | True | 3.103 | 48.125 | 47.812 | 42.812 |
| baseball glove | 17 | True | 1.602 | 71.176 | 68.824 | 70.000 |
| skateboard | 27 | True | 2.603 | 92.593 | 93.333 | 91.481 |
| surfboard | 30 | True | 2.903 | 88.333 | 88.333 | 87.667 |
| tennis racket | 39 | True | 3.804 | 91.538 | 91.026 | 89.744 |
| bottle | 79 | True | 7.808 | 33.038 | 30.506 | 32.911 |
| wine glass | 27 | True | 2.603 | 20.000 | 21.111 | 20.370 |
| cup | 92 | True | 9.109 | 33.478 | 32.065 | 33.152 |
| fork | 31 | True | 3.003 | 28.710 | 27.419 | 25.484 |
| knife | 38 | True | 3.704 | 25.526 | 22.895 | 19.474 |
| airplane | 19 | True | 1.802 | 77.895 | 78.947 | 73.684 |
| spoon | 27 | True | 2.603 | 19.259 | 18.148 | 14.815 |
| bowl | 69 | True | 6.807 | 32.464 | 30.725 | 30.870 |
| banana | 22 | True | 2.102 | 37.727 | 37.727 | 34.091 |
| apple | 17 | True | 1.602 | 23.529 | 22.353 | 21.176 |
| sandwich | 17 | True | 1.602 | 63.529 | 61.176 | 55.294 |
| orange | 18 | True | 1.702 | 28.889 | 27.222 | 30.000 |
| broccoli | 10 | True | 0.901 | 42.000 | 42.000 | 42.000 |
| carrot | 16 | True | 1.502 | 45.625 | 46.250 | 44.375 |
| hot dog | 12 | True | 1.101 | 27.500 | 30.833 | 28.333 |
| pizza | 29 | True | 2.803 | 64.138 | 62.759 | 58.966 |
| bus | 34 | True | 3.303 | 55.882 | 52.647 | 47.941 |
| donut | 11 | True | 1.001 | 40.909 | 35.455 | 27.273 |
| cake | 32 | True | 3.103 | 50.625 | 51.562 | 40.000 |
| chair | 133 | True | 13.213 | 39.474 | 39.023 | 37.594 |
| couch | 42 | True | 4.104 | 38.333 | 38.095 | 35.952 |
| potted plant | 31 | True | 3.003 | 15.161 | 13.226 | 11.290 |
| bed | 30 | True | 2.903 | 37.667 | 37.667 | 31.000 |
| dining table | 106 | True | 10.511 | 48.585 | 47.075 | 44.623 |
| train | 35 | True | 3.403 | 83.714 | 83.143 | 78.000 |
| toilet | 24 | True | 2.302 | 56.250 | 54.583 | 52.917 |
| tv | 49 | True | 4.805 | 49.184 | 50.408 | 47.347 |
| laptop | 45 | True | 4.404 | 62.444 | 61.778 | 59.778 |
| mouse | 22 | True | 2.102 | 88.636 | 88.182 | 83.636 |
| remote | 32 | True | 3.103 | 42.188 | 41.250 | 38.125 |
| keyboard | 30 | True | 2.903 | 71.667 | 73.333 | 68.000 |
| cell phone | 36 | True | 3.504 | 25.556 | 25.556 | 23.611 |
| microwave | 13 | True | 1.201 | 41.538 | 37.692 | 33.077 |
| oven | 25 | True | 2.402 | 50.800 | 50.400 | 40.400 |
| truck | 48 | True | 4.705 | 21.458 | 22.500 | 20.000 |
| toaster | 0 | False | unscorable | unscorable | unscorable | unscorable |
| sink | 41 | True | 4.004 | 60.000 | 59.024 | 54.878 |
| refrigerator | 27 | True | 2.603 | 42.593 | 39.630 | 38.519 |
| book | 49 | True | 4.805 | 29.592 | 29.184 | 29.592 |
| clock | 39 | True | 3.804 | 45.641 | 43.846 | 38.718 |
| vase | 29 | True | 2.803 | 30.345 | 29.310 | 26.552 |
| scissors | 5 | False | 0.400 | 4.000 | 4.000 | 0.000 |
| teddy bear | 17 | True | 1.602 | 56.471 | 51.176 | 38.235 |
| hair drier | 0 | False | unscorable | unscorable | unscorable | unscorable |
| boat | 21 | True | 2.002 | 39.524 | 43.333 | 30.952 |
| toothbrush | 5 | False | 0.400 | 14.000 | 12.000 | 14.000 |

## Marginal geometry (descriptive)

| representation | role | variance | mean cosine | RankMe | participation ratio | RankMe/D | PR/D |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native_768 | document | 0.326134 | 0.673866 | 324.522053 | 89.589961 | 0.422555 | 0.116654 |
| native_768 | image | 0.395358 | 0.604642 | 325.579195 | 90.690647 | 0.423931 | 0.118087 |
| native_768 | query | 0.417176 | 0.582824 | 327.639487 | 87.242077 | 0.426614 | 0.113596 |
| mrl_256 | document | 0.314944 | 0.685056 | 172.657359 | 71.093243 | 0.674443 | 0.277708 |
| mrl_256 | image | 0.374912 | 0.625088 | 178.144232 | 72.820347 | 0.695876 | 0.284454 |
| mrl_256 | query | 0.402054 | 0.597946 | 177.462712 | 69.036574 | 0.693214 | 0.269674 |
| mrl_128 | document | 0.210212 | 0.789788 | 83.026043 | 56.190069 | 0.648641 | 0.438985 |
| mrl_128 | image | 0.250436 | 0.749564 | 86.732217 | 56.227687 | 0.677595 | 0.439279 |
| mrl_128 | query | 0.281963 | 0.718037 | 88.491266 | 55.404506 | 0.691338 | 0.432848 |

[interpretation] Separate marginal geometry is not paired alignment. Three dimensions do not validate a general utility predictor.

## Native relevance permutation and random-order floors

| direction | scrambled Hit@1 % | Hit@5 % | Hit@10 % | set recall@10 % | random-order Hit@10 % |
| --- | --- | --- | --- | --- | --- |
| t2i | 0.020 | 0.520 | 1.220 | 1.220 | 1.000 |
| i2t | 0.000 | 0.100 | 0.200 | 0.040 | 0.996 |

[ours] One fixed one-parent circular shift changes recorded relevance only, preserving vectors, rankings and geometry. Floors are combinatorial gallery/positive-count references, not realistic irrelevant-caption guarantees.
