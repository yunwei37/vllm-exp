# vllm - Advanced Label Analysis Report

## 1. Label Anomalies
- Zombie labels (unused in last 6 months): 3
- Orphaned labels (only on closed issues): 10
- Solo labels (never co-occur): 0
- Significant usage spikes detected: 13

## 2. Label Quality Metrics
- Average labeling delay: 132.3 hours
- Issues without labels: 16.5%
- Single label issues: 49.6%
- Multi-label issues: 33.9%
- Average labels per issue: 1.19

## 3. Performance Bottlenecks
- High stall rate labels: 3
- Comment explosion labels: 5
- Problematic label combinations: 23
- Worst stall label: unstale
- Worst comment explosion label: keep-open

## 4. Label Optimization Opportunities
- Redundant label pairs identified: 0
- Labels needed for 50% coverage: 2
- Labels needed for 80% coverage: 4
- Labels needed for 95% coverage: 8
- Total labels in system: 39

## 5. Complete Distribution Statistics
- Total unique labels: 39
- Gini coefficient: 0.843
- Entropy: 2.040
- Power law exponent: 2.520337178555285
- Mean label usage: 294.8
- Median label usage: 28

## 6. Label System Health
- Entropy trend: 0.007 (increasing complexity)
- Current entropy: 1.754354358565613
- Dying labels: 4
- Growing labels: 5
- Team consistency score: 0.14

## Key Recommendations

1. **Label Cleanup**: Remove 3 zombie labels that haven't been used recently
2. **Redundancy Reduction**: Consider merging 0 redundant label pairs
3. **Process Improvement**: Focus on 3 labels with high stall rates
4. **Simplification**: Current system uses 39 labels but only 4 are needed for 80% coverage

## Files Generated
- Complete label distribution: vllm_complete_label_distribution.csv
- Visualizations: *_label_*.png files
