# Cross-dataset evaluation - local models (valid, prob-sum late merge, leakage-clean)

## RoBERTa

### RoBERTa - slot-f1 (valid, prob-sum late merge, leakage-clean)
| train \ eval | DNRTI | ATTACKER | APTNER | CYNER |
|---|---|---|---|---|
| **DNRTI** | **0.63** | 0.30 | 0.21 | 0.16 |
| **ATTACKER** | 0.34 | **0.61** | 0.26 | 0.37 |
| **APTNER** | 0.39 | 0.32 | **0.59** | 0.55 |
| **CYNER** | 0.25 | 0.24 | 0.32 | **0.73** |

### RoBERTa - precision (valid, prob-sum late merge, leakage-clean)
| train \ eval | DNRTI | ATTACKER | APTNER | CYNER |
|---|---|---|---|---|
| **DNRTI** | **0.63** | 0.29 | 0.15 | 0.13 |
| **ATTACKER** | 0.38 | **0.61** | 0.20 | 0.31 |
| **APTNER** | 0.48 | 0.40 | **0.51** | 0.58 |
| **CYNER** | 0.30 | 0.29 | 0.28 | **0.76** |

### RoBERTa - recall (valid, prob-sum late merge, leakage-clean)
| train \ eval | DNRTI | ATTACKER | APTNER | CYNER |
|---|---|---|---|---|
| **DNRTI** | **0.64** | 0.32 | 0.38 | 0.21 |
| **ATTACKER** | 0.32 | **0.62** | 0.40 | 0.45 |
| **APTNER** | 0.33 | 0.27 | **0.69** | 0.52 |
| **CYNER** | 0.22 | 0.21 | 0.37 | **0.71** |

### RoBERTa - ul_slot-f1 (valid, prob-sum late merge, leakage-clean)
| train \ eval | DNRTI | ATTACKER | APTNER | CYNER |
|---|---|---|---|---|
| **DNRTI** | **0.71** | 0.43 | 0.37 | 0.44 |
| **ATTACKER** | 0.55 | **0.67** | 0.36 | 0.48 |
| **APTNER** | 0.55 | 0.45 | **0.66** | 0.60 |
| **CYNER** | 0.57 | 0.42 | 0.58 | **0.82** |

### RoBERTa - l_slot-f1 (valid, prob-sum late merge, leakage-clean)
| train \ eval | DNRTI | ATTACKER | APTNER | CYNER |
|---|---|---|---|---|
| **DNRTI** | **0.69** | 0.40 | 0.27 | 0.20 |
| **ATTACKER** | 0.44 | **0.69** | 0.39 | 0.48 |
| **APTNER** | 0.44 | 0.41 | **0.67** | 0.59 |
| **CYNER** | 0.29 | 0.38 | 0.35 | **0.78** |
