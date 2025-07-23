# llama_cpp - Advanced Label Analysis Report

## 1. Label Anomalies
- Zombie labels (unused in last 6 months): 26
- Orphaned labels (only on closed issues): 25
- Solo labels (never co-occur): 0
- Significant usage spikes detected: 17

## 2. Label Quality Metrics
- Average labeling delay: 113.3 hours
- Issues without labels: 14.8%
- Single label issues: 37.1%
- Multi-label issues: 48.1%
- Average labels per issue: 1.45

## 3. Performance Bottlenecks
- High stall rate labels: 0
- Comment explosion labels: 13
- Problematic label combinations: 14
- Worst stall label: help wanted
- Worst comment explosion label: model

## 4. Label Optimization Opportunities
- Redundant label pairs identified: 4
- Labels needed for 50% coverage: 1
- Labels needed for 80% coverage: 2
- Labels needed for 95% coverage: 6
- Total labels in system: 61

## 5. Complete Distribution Statistics
- Total unique labels: 61
- Gini coefficient: 0.869
- Entropy: 2.148
- Power law exponent: 2.0301778316282917
- Mean label usage: 130.4
- Median label usage: 11

## 6. Label System Health
- Entropy trend: -0.022 (decreasing complexity)
- Current entropy: 1.028239168000227
- Dying labels: 1
- Growing labels: 1
- Team consistency score: 0.10

## Key Recommendations

1. **Label Cleanup**: Remove 26 zombie labels that haven't been used recently
2. **Redundancy Reduction**: Consider merging 4 redundant label pairs
3. **Process Improvement**: Focus on 0 labels with high stall rates
4. **Simplification**: Current system uses 61 labels but only 2 are needed for 80% coverage

## Files Generated
- Complete label distribution: llama_cpp_complete_label_distribution.csv
- Visualizations: *_label_*.png files
