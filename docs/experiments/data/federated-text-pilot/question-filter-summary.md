# Question-filter Stage-1 results

Development diagnostics; not sequence recovery.

| Defense | Method | Image | Q recall | Q-exclusive recall | A recall | Joint recall | Joint full |
|---|---|---|---:|---:|---:|---:|---:|
| noise | Residual | random | 0.0998 | 0.0991 | 0.3981 | 0.1792 | 0/30 |
| noise | B anchor | random | 0.1075 | 0.1005 | 0.8273 | 0.3143 | 0/30 |
| noise | Blind question prior | random | 0.4659 | 0.4620 | 0.8273 | 0.5664 | 0/30 |
| noise | Answer-conditioned prior | random | 0.4275 | 0.4207 | 0.8310 | 0.5441 | 0/30 |
| noise | K/V energy | random | 0.1324 | 0.1261 | 0.7111 | 0.2901 | 0/30 |
| noise | B forward probe | random | 0.1456 | 0.1401 | 0.6981 | 0.3016 | 0/30 |
| noise | B input probe | random | 0.0119 | 0.0056 | 0.6911 | 0.1991 | 0/30 |
| noise | Residual | gt | 0.0957 | 0.0967 | 0.3462 | 0.1738 | 0/30 |
| noise | B anchor | gt | 0.0979 | 0.0926 | 0.8216 | 0.3087 | 0/30 |
| noise | Blind question prior | gt | 0.4548 | 0.4528 | 0.8216 | 0.5589 | 0/30 |
| noise | Answer-conditioned prior | gt | 0.4185 | 0.4115 | 0.8269 | 0.5376 | 0/30 |
| noise | K/V energy | gt | 0.1121 | 0.1070 | 0.7061 | 0.2757 | 0/30 |
| noise | B forward probe | gt | 0.1456 | 0.1401 | 0.6981 | 0.3016 | 0/30 |
| noise | B input probe | gt | 0.0119 | 0.0056 | 0.6911 | 0.1991 | 0/30 |
| sparsify | Residual | random | 0.5389 | 0.5368 | 0.5752 | 0.5689 | 0/30 |
| sparsify | B anchor | random | 0.5322 | 0.5263 | 0.8690 | 0.6464 | 1/30 |
| sparsify | Blind question prior | random | 0.5903 | 0.5850 | 0.8329 | 0.6697 | 1/30 |
| sparsify | Answer-conditioned prior | random | 0.6072 | 0.6019 | 0.8551 | 0.6884 | 1/30 |
| sparsify | K/V energy | random | 0.4077 | 0.4008 | 0.7762 | 0.5141 | 0/30 |
| sparsify | Masked ridge | random | 0.0214 | 0.0151 | 0.6486 | 0.1907 | 0/30 |
| sparsify | B forward probe | random | 0.1502 | 0.1449 | 0.6556 | 0.2878 | 0/30 |
| sparsify | B input probe | random | 0.0119 | 0.0056 | 0.6486 | 0.1823 | 0/30 |
| sparsify | Residual | gt | 0.5423 | 0.5403 | 0.5760 | 0.5784 | 0/30 |
| sparsify | B anchor | gt | 0.5423 | 0.5363 | 0.8729 | 0.6560 | 1/30 |
| sparsify | Blind question prior | gt | 0.6039 | 0.5986 | 0.8598 | 0.6882 | 1/30 |
| sparsify | Answer-conditioned prior | gt | 0.6229 | 0.6176 | 0.8800 | 0.7070 | 1/30 |
| sparsify | K/V energy | gt | 0.4197 | 0.4128 | 0.7601 | 0.5126 | 0/30 |
| sparsify | Masked ridge | gt | 0.0269 | 0.0206 | 0.6486 | 0.1954 | 0/30 |
| sparsify | B forward probe | gt | 0.1502 | 0.1449 | 0.6556 | 0.2878 | 0/30 |
| sparsify | B input probe | gt | 0.0119 | 0.0056 | 0.6486 | 0.1823 | 0/30 |
