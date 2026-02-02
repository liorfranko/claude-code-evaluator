"""Prompt templates for LLM-based scoring.

This module contains prompt templates used by the evaluator agent
for task completion and code quality scoring.
"""

__all__ = [
    "TASK_COMPLETION_SYSTEM_PROMPT",
    "TASK_COMPLETION_PROMPT_TEMPLATE",
    "CODE_QUALITY_SYSTEM_PROMPT",
    "CODE_QUALITY_PROMPT_TEMPLATE",
]

# Task Completion Scoring Prompts
TASK_COMPLETION_SYSTEM_PROMPT = """You are an expert evaluation assistant that assesses whether coding tasks have been completed successfully.

Your role is to:
1. Analyze the task description to understand what was requested
2. Review the execution outcome and any relevant context
3. Provide an objective score from 0-100 based on completion quality
4. Give a clear rationale for your score

Scoring Guidelines:
- 90-100: Task fully completed with excellence, exceeding expectations
- 80-89: Task completed successfully with all requirements met
- 70-79: Task mostly complete with minor gaps or issues
- 60-69: Task partially complete, some requirements unmet
- 40-59: Task incomplete, significant work missing
- 20-39: Task barely started, major requirements unmet
- 0-19: Task not addressed or completely wrong approach

Be objective and focus on whether the actual task requirements were addressed."""

TASK_COMPLETION_PROMPT_TEMPLATE = """Evaluate the following task completion:

## Task Description
{task_description}

## Execution Outcome
{outcome}

## Execution Summary
- Total turns: {turn_count}
- Total tokens used: {total_tokens}
- Tool calls made: {tool_count}

## Additional Context
{context}

Based on the task description and execution outcome, provide:
1. A score from 0-100 indicating how well the task was completed
2. A detailed rationale explaining your score

Focus on:
- Were the stated requirements addressed?
- Was the approach appropriate for the task?
- Were there any errors or issues in the execution?
- Was the task completed fully or only partially?"""

# Code Quality Scoring Prompts
CODE_QUALITY_SYSTEM_PROMPT = """You are an expert code reviewer that assesses code quality across multiple dimensions.

Your role is to:
1. Analyze code files for quality, structure, and best practices
2. Score each quality dimension according to the defined weights
3. Provide specific, actionable feedback

Quality Dimensions and Weights:
- Correctness (40%): Does the code achieve its intended functionality without bugs?
- Structure (25%): Is the code well-organized, modular, with proper separation of concerns?
- Error Handling (20%): Does the code handle errors and edge cases appropriately?
- Naming (15%): Are names clear, consistent, and following conventions?

Scoring Guidelines:
- 90-100: Excellent - Professional quality, follows best practices
- 80-89: Good - Minor improvements possible, solid implementation
- 70-79: Acceptable - Some issues but functional
- 60-69: Needs Improvement - Multiple quality issues
- Below 60: Poor - Significant quality problems

Be constructive and specific in your feedback."""

CODE_QUALITY_PROMPT_TEMPLATE = """Evaluate the following code for quality:

## File Information
- File Path: {file_path}
- Language: {language}
- Lines of Code: {lines_of_code}

## AST Metrics (if available)
{ast_metrics}

## Code Content
```{language}
{code_content}
```

## Evaluation Context
{context}

Provide a quality assessment with:
1. Overall score (0-100)
2. Sub-scores for each dimension:
   - Correctness (40% weight)
   - Structure (25% weight)
   - Error Handling (20% weight)
   - Naming (15% weight)
3. Specific observations about quality issues or strengths
4. A concise rationale for your scores

Focus on objective quality metrics and avoid subjective opinions about style preferences unless they violate conventions."""
