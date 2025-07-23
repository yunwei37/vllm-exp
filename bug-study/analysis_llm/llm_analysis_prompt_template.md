# LLM Analysis Prompt Template

## Instructions for LLM Analysis of GitHub Issues

You are analyzing a sample of GitHub issues from [FRAMEWORK_NAME]. Your task is to answer the research questions below based on the provided issue data.

### Critical Requirements:

1. **Citation is Mandatory**
   - ALWAYS cite original issues using format: `Issue #[number]([html_url])`
   - Example: `Issue #1234(https://github.com/vllm-project/vllm/issues/1234)`
   - Every claim must be backed by specific issue citations

2. **Analysis Structure for Each Research Question**
   - Start with an overall summary of patterns found
   - Provide specific examples (minimum 3-5 issues) for each pattern
   - Note variations or outliers with citations
   - Include rough percentages or counts

3. **Issue Grouping Guidelines**
   - Group similar issues into categories
   - Each category needs representative examples
   - Maintain issue traceability throughout

### Research Questions to Answer:

[INSERT_RESEARCH_QUESTIONS_HERE]

### Sample Data Location:
- File: `issues.json` in the current directory
- Contains [SAMPLE_SIZE] issues selected by [SAMPLING_CRITERIA]

### Expected Output Format:

```markdown
# Analysis Results for [SAMPLE_NAME]

## RQ1: [First Research Question]

### Summary
[Overall pattern description with percentage breakdowns]

### Detailed Findings

**Pattern 1: [Pattern Name] (X% of samples)**
- Issue #123(https://github.com/org/repo/issues/123): [Brief description of the issue]
- Issue #456(https://github.com/org/repo/issues/456): [Brief description]
- Issue #789(https://github.com/org/repo/issues/789): [Brief description]

[Detailed explanation of this pattern with more examples if needed]

**Pattern 2: [Pattern Name] (Y% of samples)**
[Same structure as above]

### Outliers and Special Cases
- Issue #999(link): [Description of why this is an outlier]

---

## RQ2: [Second Research Question]
[Same structure as RQ1]

---

## Cross-Cutting Observations

[Any patterns that span multiple RQs, with issue citations]

## Recommendations

Based on the analysis:
1. [Recommendation 1] (supported by Issues #X, #Y, #Z)
2. [Recommendation 2] (supported by Issues #A, #B, #C)
```

### Quality Checklist:
- [ ] Every pattern has at least 3 issue examples
- [ ] All issue numbers are accurate and include links
- [ ] Percentages/counts are provided for pattern prevalence
- [ ] Outliers are identified and explained
- [ ] Cross-cutting patterns are noted
- [ ] Recommendations are evidence-based

### Common Patterns to Look For:
1. **Error Types**: What kinds of errors/bugs appear?
2. **User Confusion**: What do users struggle with?
3. **Missing Features**: What capabilities are requested?
4. **Documentation Gaps**: What information is missing?
5. **Performance Issues**: What performance problems arise?
6. **Configuration Problems**: What setup issues occur?
7. **Compatibility Issues**: What conflicts with other systems?
8. **Workflow Disruptions**: How do issues impact user workflows?

Remember: The goal is to provide actionable insights for framework maintainers and tool developers, with every insight traceable back to specific issues.