# Label-Based Analysis - deepseek

## Research Questions

- RQ1: What types of issues are actually captured under this label?
- RQ2: What are the common technical problems in this label category?
- RQ3: How are issues in this category typically resolved?
- RQ4: What information is typically missing or well-provided?

## Sample Information

- **Framework**: sglang
- **Method Type**: label_based
- **Sample Name**: deepseek
- **Sample Size**: 30
- **Sampling Criteria**: Issues with label 'deepseek'

## LLM Analysis Prompt Instructions

When analyzing these issues, please follow these guidelines:

### 1. Citation Requirements
- **ALWAYS cite the original issue** when discussing specific examples
- Use format: `Issue #[number]([html_url])` 
- Example: `Issue #1234(https://github.com/vllm-project/vllm/issues/1234)`
- When summarizing patterns, cite at least 3 representative issues

### 2. Analysis Structure
For each research question:
1. **Overall Pattern**: Describe the general pattern observed
2. **Specific Examples**: Cite 3-5 specific issues that exemplify this pattern
3. **Variations**: Note any significant variations or outliers (with citations)
4. **Quantification**: Provide rough percentages or counts where relevant

### 3. Issue Reference Format
When referencing issues in your analysis:
```
As seen in Issue #1234(link), the user reports [specific detail]...
Similar patterns appear in Issue #5678(link) and Issue #9012(link).
```

### 4. Summary Requirements
- Group similar issues together but maintain traceability
- For each finding, provide issue numbers as evidence
- Create categories/taxonomies with example issues for each category

### 5. Quality Checks
Before finalizing analysis:
- Ensure every claim is backed by specific issue citations
- Verify issue numbers and links are correct
- Check that patterns are supported by multiple examples

### Example Analysis Format:
```
RQ1: What types of issues are actually captured under this label?

**Finding 1: Memory-related errors (40% of samples)**
- Issue #1234(link): OOM during model loading
- Issue #5678(link): GPU memory fragmentation
- Issue #9012(link): Memory leak in attention mechanism

**Finding 2: Configuration problems (30% of samples)**
- Issue #2345(link): Incorrect tensor_parallel_size
- Issue #6789(link): Incompatible model format
[etc.]
```

## Analysis Status

⏳ **Status**: Not Started
📅 **Created**: 2025-07-23 11:26:32

## Placeholder for Analysis Results

This directory will contain:
1. `analysis_results.json` - Structured analysis output
2. `findings.md` - Human-readable findings and insights
3. `patterns.json` - Extracted patterns and categories
4. `recommendations.md` - Actionable recommendations

---
*This is a placeholder document. Analysis has not been conducted yet.*
