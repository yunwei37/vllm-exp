# LLM-Based Issue Analysis for LLM Serving Frameworks

## Overview
This document outlines LLM-based analysis methods for understanding issue patterns in vLLM, SGLang, and llama.cpp. The analysis focuses on extracting semantic insights from issue content using language models.

```
'/root/yunwei37/vllm-exp/bug-study/analysis_llm/analysis_results/vllm/long_tail' start from this dir, read the sampling and summary and answer rq and discover patterns based on the sampled commits in '/root/yunwei37/vllm-exp/bug-study/analysis_llm/markdown_samples'. start from bug and performance, make sure you cover all. follow the guideline and make sure you read everything in the target dir. use /root/yunwei37/vllm-exp/bug-study/analysis_llm/check_analysis_completion.py to check your progress. the result should be like /root/yunwei37/vllm-exp/bug-study/analysis_llm/analysis_results/vllm/label_based/bug/findings.md
```

## Label-Based Sampling Strategy

### Sampling Method
Each label forms a distinct sampling group with standardized sample sizes:
- **Major labels** (≥30 issues): Random sample of 30 issues
- **Minor labels** (<30 issues): All available issues
- **Merged group**: Combine all labels with <30 issues into one analysis group

### Label Groups for vLLM

#### Major Label Groups (30 samples each)
1. **bug** (3,987 total issues)
2. **stale** (2,868 total issues)
3. **usage** (1,285 total issues)
4. **feature request** (1,092 total issues)
5. **installation** (342 total issues)
6. **performance** (292 total issues)
7. **new-model** (244 total issues)
8. **misc** (230 total issues)
9. **RFC** (189 total issues)
10. **unstale** (158 total issues)
11. **documentation** (157 total issues)
12. **good first issue** (112 total issues)
13. **ray** (96 total issues)
14. **rocm** (75 total issues)

#### Minor Label Groups (all available issues)
15. **help wanted** (57 total issues)
16. **structured-output** (56 total issues)
17. **ci-failure** (36 total issues)
18. **tool-calling** (34 total issues)
19. **v1** (28 total issues)
20. **torch.compile** (28 total issues)
21. **distributed** (24 total issues)
22. **keep-open** (20 total issues)
23. **tracking** (17 total issues)
24. **vulkan** (13 total issues)
25. **windows** (13 total issues)
26. **roadmap** (7 total issues)
27. **neuron** (6 total issues)
28. **CPU** (6 total issues)
29. **quantization** (5 total issues)

### Label Groups for SGLang

#### Major Label Groups (30 samples each)
1. **inactive** (760 total issues)
2. **high priority** (178 total issues)
3. **good first issue** (169 total issues)
4. **help wanted** (156 total issues)
5. **bug** (94 total issues)
6. **deepseek** (57 total issues)
7. **enhancement** (46 total issues)
8. **documentation** (34 total issues)
9. **feature** (33 total issues)

#### Minor Label Groups (all available issues)
10. **amd** (29 total issues)
11. **collaboration** (28 total issues)
12. **await-response** (28 total issues)
13. **lora** (24 total issues)
14. **speculative-decoding** (24 total issues)
15. **MLLM** (23 total issues)
16. **performance** (20 total issues)
17. **new-model** (19 total issues)
18. **quant** (18 total issues)
19. **flashinfer** (15 total issues)
20. **router** (12 total issues)
21. Other labels with <12 issues

### Label Groups for llama.cpp

#### Major Label Groups (30 samples each)
1. **stale** (2,496 total issues)
2. **bug-unconfirmed** (2,458 total issues)
3. **enhancement** (922 total issues)
4. **bug** (292 total issues)
5. **medium severity** (278 total issues)
6. **low severity** (171 total issues)
7. **good first issue** (159 total issues)
8. **high severity** (151 total issues)
9. **help wanted** (101 total issues)
10. **model** (95 total issues)
11. **critical severity** (84 total issues)
12. **build** (60 total issues)
13. **server/webui** (56 total issues)
14. **performance** (53 total issues)
15. **need more info** (51 total issues)
16. **duplicate** (47 total issues)
17. **research 🔬** (37 total issues)
18. **high priority** (37 total issues)
19. **question** (34 total issues)
20. **hardware** (34 total issues)

#### Minor Label Groups (all available issues)
21. **mac** (25 total issues)
22. **GGML** (20 total issues)
23. **refactoring** (18 total issues)
24. Other labels with <18 issues

## Research Questions

### For Each Major Label Group

#### RQ1: Content Consistency Analysis
- **Question**: What types of issues are actually captured under this label?
- **Method**: LLM categorization of issue content within label
- **Analysis**: Identify subcategories and mislabeled issues

#### RQ2: Problem Pattern Extraction
- **Question**: What are the common technical problems in this label category?
- **Method**: Extract problem descriptions, error messages, and symptoms
- **Analysis**: Create problem taxonomy for each label

#### RQ3: Resolution Pattern Analysis
- **Question**: How are issues in this category typically resolved?
- **Method**: Analyze closed issues for solution patterns
- **Analysis**: Extract fix strategies, workarounds, and resolution times

#### RQ4: Information Quality Assessment
- **Question**: What information is typically missing or well-provided?
- **Method**: Assess completeness of bug reports
- **Analysis**: Identify information gaps per label category

### For Minor Label Groups

#### RQ5: Label Distinction Analysis
- **Question**: Do minor labels represent distinct issue categories?
- **Method**: Compare issue content across minor labels
- **Analysis**: Recommend label consolidation or promotion

#### RQ6: Label Lifecycle Assessment
- **Question**: Why do these labels have low usage?
- **Method**: Analyze temporal patterns and content relevance
- **Analysis**: Identify deprecated or emerging categories

### Cross-Label Analysis

#### RQ7: Semantic Overlap Detection
- **Question**: Which labels have overlapping content?
- **Method**: Compare issue similarity across different labels
- **Analysis**: Identify redundant labels and suggest mergers

#### RQ8: Unlabeled Issue Categorization
- **Question**: What categories exist in unlabeled issues?
- **Method**: Cluster unlabeled issues by content
- **Analysis**: Suggest new labels or assign existing ones

## Additional Sampling Methods (Non-NLP Based)

### 1. Long-Tail Label Sampling
**Method**: Focus on rarely-used labels to understand edge cases
- **Head labels**: Top 5 labels by frequency (sample 10 each)
- **Body labels**: Middle 50% of labels (sample 20 each)
- **Tail labels**: Bottom 25% of labels (take all issues)

**Research Questions**:
- Why are tail labels rarely used?
- Do tail labels represent important edge cases?
- Should tail labels be consolidated or removed?

### 2. Temporal-Based Sampling
**Method**: Sample issues from different time periods
- **Recent burst**: Last 30 days (30 issues)
- **Steady state**: 6 months ago (30 issues)
- **Historical**: 1+ year ago (30 issues)
- **First month**: Framework's first 30 days (all issues)

**Research Questions**:
- How have issue types evolved over time?
- What new problems emerged recently?
- Which problems have persisted throughout the project?

### 3. Resolution Time-Based Sampling
**Method**: Group by time to close
- **Lightning fast**: Closed < 1 hour (30 issues)
- **Quick**: Closed 1-24 hours (30 issues)
- **Normal**: Closed 1-7 days (30 issues)
- **Slow**: Closed 7-30 days (30 issues)
- **Very slow**: Closed > 30 days (30 issues)
- **Never closed**: Open > 180 days (30 issues)

**Research Questions**:
- What distinguishes quick vs slow resolution?
- Are quickly closed issues actually fixed or dismissed?
- Why do some issues remain open indefinitely?

### 4. Complexity-Based Sampling
**Method**: Use quantitative metrics as proxy for complexity
- **Zero comments**: No discussion (30 issues)
- **Low discussion**: 1-5 comments (30 issues)
- **Medium discussion**: 6-20 comments (30 issues)
- **High discussion**: 21-50 comments (30 issues)
- **Very high discussion**: >50 comments (30 issues)

**Research Questions**:
- What causes extensive discussion?
- Are zero-comment issues low quality or self-evident?
- Does discussion length correlate with issue difficulty?

### 5. Author-Based Sampling
**Method**: Group by author characteristics
- **First-time contributors**: Author's first issue (30 issues)
- **Regular contributors**: Authors with 5-20 issues (30 issues)
- **Power users**: Authors with >20 issues (30 issues)
- **Team members**: Author association = MEMBER (30 issues)
- **External users**: Author association = NONE (30 issues)

**Research Questions**:
- How does issue quality vary by author experience?
- What problems do new users face vs experienced users?
- Do team members report different types of issues?

### 6. Reaction-Based Sampling
**Method**: Use GitHub reactions as importance proxy
- **High impact**: >10 total reactions (30 issues)
- **Moderate impact**: 3-10 reactions (30 issues)
- **Low impact**: 1-2 reactions (30 issues)
- **No engagement**: 0 reactions (30 issues)

**Research Questions**:
- What makes issues resonate with the community?
- Are highly-reacted issues actually more important?
- Why do some issues get no engagement?

### 7. Cross-Reference Sampling
**Method**: Issues that reference other issues/PRs
- **References PRs**: Issues linking to pull requests (30 issues)
- **Referenced by many**: Issues referenced by >3 other issues (all)
- **Duplicate chains**: Issues marked as duplicates (30 issues)
- **Related clusters**: Issues referencing each other (analyze clusters)

**Research Questions**:
- What patterns exist in related issue clusters?
- How accurate are duplicate markings?
- What issues spawn the most related work?

### 8. State Transition Sampling
**Method**: Focus on issues with state changes
- **Reopened issues**: Closed then reopened (all issues)
- **Stale-to-active**: Had stale label removed (30 issues)
- **Label churners**: >3 label changes (30 issues)
- **Status changers**: Multiple state_reason changes (all issues)

**Research Questions**:
- Why do issues get reopened?
- What causes label/state churn?
- Are churning issues poorly defined?

### 9. Metadata Anomaly Sampling
**Method**: Statistical outliers in metadata
- **Update anomalies**: Updated >50 times (all issues)
- **Time anomalies**: Created and closed within 1 minute (30 issues)
- **Format anomalies**: Unusually short (<10 words) or long (>2000 words) (30 each)
- **Engagement anomalies**: High views but no comments (30 issues)

**Research Questions**:
- What causes extreme metadata patterns?
- Are anomalous issues special cases or noise?
- Do format anomalies indicate quality issues?

### 10. Comparative Sampling
**Method**: Same criteria across frameworks
- **Common labels**: Issues with labels existing in all 3 frameworks (30 per label per framework)
- **Framework-unique labels**: Labels only in one framework (30 per label)
- **Similar timeframe**: Same week across frameworks (all issues from selected weeks)

**Research Questions**:
- How do frameworks handle similar problems differently?
- What problems are unique to each framework?
- Do frameworks influence each other over time?

## Implementation Structure

```python
# sample_generator.py
def generate_label_samples(framework_name, data_path):
    """Generate stratified samples for each label"""
    # Load data
    # For each label:
    #   - If count >= 30: random sample 30
    #   - If count < 30: take all
    # Save samples per label

# llm_analyzer.py
def analyze_label_group(label_name, samples):
    """Run LLM analysis on a label group"""
    # For each sample:
    #   - Extract problem type
    #   - Assess information quality
    #   - Identify patterns
    # Generate label report

# pattern_synthesizer.py
def synthesize_patterns(all_label_analyses):
    """Synthesize patterns across all labels"""
    # Cross-label comparison
    # Identify common themes
    # Generate recommendations
```

## Expected Outputs

### Per-Label Reports
- Issue type distribution within label
- Common problem patterns
- Information quality metrics
- Resolution pattern analysis

### Cross-Label Analysis
- Label overlap matrix
- Suggested label consolidations
- New label recommendations
- Labeling consistency metrics

### Framework Comparison
- Label usage patterns across frameworks
- Common vs unique problem categories
- Framework-specific issue patterns