# sglang - Advanced Label Analysis Report

## 1. Label Anomalies
- Zombie labels (unused in last 6 months): 1
- Orphaned labels (only on closed issues): 9
- Solo labels (never co-occur): 0
- Significant usage spikes detected: 6

## 2. Label Quality Metrics
- Average labeling delay: 217.3 hours
- Issues without labels: 51.5%
- Single label issues: 33.1%
- Multi-label issues: 15.4%
- Average labels per issue: 0.71

## 3. Performance Bottlenecks
- High stall rate labels: 2
- Comment explosion labels: 8
- Problematic label combinations: 7
- Worst stall label: speculative-decoding
- Worst comment explosion label: collaboration

## 4. Label Optimization Opportunities
- Redundant label pairs identified: 0
- Labels needed for 50% coverage: 1
- Labels needed for 80% coverage: 3
- Labels needed for 95% coverage: 10
- Total labels in system: 38

## 5. Complete Distribution Statistics
- Total unique labels: 38
- Gini coefficient: 0.774
- Entropy: 2.283
- Power law exponent: 1.9377804063887252
- Mean label usage: 47.9
- Median label usage: 14

## 6. Label System Health
- Entropy trend: 0.114 (increasing complexity)
- Current entropy: 2.1749565953207304
- Dying labels: 5
- Growing labels: 3
- Team consistency score: 0.13

## Key Recommendations

1. **Label Cleanup**: Remove 1 zombie labels that haven't been used recently
2. **Redundancy Reduction**: Consider merging 0 redundant label pairs
3. **Process Improvement**: Focus on 2 labels with high stall rates
4. **Simplification**: Current system uses 38 labels but only 3 are needed for 80% coverage

## Files Generated
- Complete label distribution: sglang_complete_label_distribution.csv
- Visualizations: *_label_*.png files
