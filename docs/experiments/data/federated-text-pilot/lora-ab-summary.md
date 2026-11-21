| Condition | Image | Method | N | Candidate recall | FullCoverage | Informative AP |
|---|---|---|---:|---:|---:|---:|
| fedsgd-b1 | gt | baseline | 30 | 0.9943 | 26/30 | 0.9922 |
| fedsgd-b1 | gt | ab_consistency | 30 | 0.9934 | 26/30 | 0.9890 |
| fedsgd-b1 | gt | residual_left | 30 | 0.9943 | 26/30 | 0.9922 |
| fedsgd-b1 | gt | residual_left_gated | 30 | 0.9943 | 26/30 | 0.9922 |
| noise | gt | baseline | 30 | 0.1738 | 0/30 | 0.1100 |
| noise | gt | ab_consistency | 30 | 0.1701 | 0/30 | 0.1103 |
| noise | gt | residual_left | 30 | 0.1691 | 0/30 | 0.1108 |
| noise | gt | residual_left_gated | 30 | 0.1701 | 0/30 | 0.1036 |
| fedsgd-b1 | random | baseline | 30 | 0.9906 | 26/30 | 0.9871 |
| fedsgd-b1 | random | ab_consistency | 30 | 0.9887 | 26/30 | 0.9827 |
| fedsgd-b1 | random | residual_left | 30 | 0.9906 | 26/30 | 0.9866 |
| fedsgd-b1 | random | residual_left_gated | 30 | 0.9906 | 26/30 | 0.9866 |
| noise | random | baseline | 30 | 0.1792 | 0/30 | 0.1077 |
| noise | random | ab_consistency | 30 | 0.1841 | 0/30 | 0.1083 |
| noise | random | residual_left | 30 | 0.1776 | 0/30 | 0.1122 |
| noise | random | residual_left_gated | 30 | 0.1566 | 0/30 | 0.0980 |
