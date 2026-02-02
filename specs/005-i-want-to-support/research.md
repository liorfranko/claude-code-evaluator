# Research: Brownfield Repository Support

## Overview

This document captures technical research and decisions for implementing brownfield repository support in claude-evaluator. The feature enables users to evaluate Claude's ability to add features to existing codebases by cloning external GitHub repositories instead of starting from empty workspaces.

## Technical Unknowns

### Unknown 1: Git Clone Integration Pattern

**Question**: How should the system clone repositories and integrate with the existing workspace management?

**Options Considered**:
1. **Subprocess Git Commands** - Use `subprocess.run()` to invoke git CLI directly
2. **GitPython Library** - Use the `gitpython` package for programmatic git access
3. **pygit2 Library** - Use `pygit2` for libgit2 bindings

**Decision**: Subprocess Git Commands

**Rationale**:
- Git CLI is universally available on target platforms (macOS, Linux)
- No new dependencies required (aligns with constitution "minimize external dependencies")
- Simpler error handling - can parse git's stderr for user-friendly messages
- Shallow clone and branch selection are straightforward CLI flags
- Existing codebase doesn't use GitPython, so no pattern to follow

**Trade-offs**:
- Less programmatic control than a library
- Need to parse CLI output for error handling
- Depends on git being installed (reasonable assumption for developer tool)

**Sources**:
- Git documentation: `git clone --depth` for shallow clones
- Git documentation: `git clone --branch` for ref selection

### Unknown 2: URL Validation Approach

**Question**: How should the system validate GitHub HTTPS URLs before cloning?

**Options Considered**:
1. **Regex Pattern Matching** - Use regex to validate URL format
2. **urllib.parse + Manual Validation** - Parse URL and validate components
3. **Pydantic Field Validator** - Use Pydantic's built-in URL validation with custom rules

**Decision**: Pydantic Field Validator with custom validation

**Rationale**:
- Consistent with existing codebase patterns (Pydantic models throughout)
- Provides automatic validation at model instantiation
- Custom validator can check:
  - URL scheme is "https"
  - Host is "github.com"
  - Path has owner/repo structure
- Integrates with existing error handling patterns

**Trade-offs**:
- Slightly more complex than simple regex
- Coupled to Pydantic (acceptable given existing usage)

**Sources**:
- Pydantic v2 documentation: Field validators
- Existing usage in `src/claude_evaluator/config/models.py`

### Unknown 3: Change Detection Method

**Question**: How should the system detect and summarize changes made during evaluation?

**Options Considered**:
1. **Git Status Parsing** - Run `git status --porcelain` and parse output
2. **Git Diff Parsing** - Run `git diff --name-status` to get detailed changes
3. **File System Comparison** - Snapshot file tree before/after, compare

**Decision**: Git Status Parsing with `--porcelain` flag

**Rationale**:
- Porcelain format is machine-readable and stable across git versions
- Directly shows modified (M), added (?? for untracked, A for staged), deleted (D) files
- Minimal overhead - single git command
- Handles edge cases like renamed files

**Trade-offs**:
- Requires git repository (guaranteed for brownfield mode)
- Only detects uncommitted changes (which is correct for this use case)

**Sources**:
- Git documentation: `git status --porcelain` format specification

### Unknown 4: Workspace Directory Structure

**Question**: Where should brownfield workspaces be stored relative to existing workspace pattern?

**Options Considered**:
1. **Same Pattern as Temp Workspaces** - Use `tempfile.mkdtemp()` like current implementation
2. **Brownfield Subdirectory** - Create `brownfield/` subdirectory in a persistent location
3. **Configurable Path** - Allow user to specify workspace location

**Decision**: Brownfield subdirectory within existing workspace pattern, using unique identifiers

**Rationale**:
- Separates brownfield workspaces from temporary greenfield workspaces
- Enables easy cleanup and management of brownfield evaluations
- Unique naming (evaluation ID + timestamp) prevents collisions
- Matches existing workspace path pattern but with `brownfield/` prefix

**Trade-offs**:
- Requires managing persistent directories (not auto-cleaned by OS)
- Need to document workspace location for users

**Sources**:
- Existing implementation: `Evaluation.start()` in `src/claude_evaluator/core/evaluation.py`

### Unknown 5: Clone Performance and Size Warning

**Question**: How should the system estimate repository size and warn users?

**Options Considered**:
1. **GitHub API Query** - Use GitHub API to get repository size before cloning
2. **HEAD Request** - Make HTTP HEAD request to estimate download size
3. **Clone with Progress Callback** - Show progress during clone, warn if slow

**Decision**: GitHub API Query using `gh` CLI

**Rationale**:
- `gh` CLI is already a de facto dependency for GitHub interaction in dev workflows
- `gh api repos/{owner}/{repo}` returns repository size in KB
- Can warn before clone starts, not during
- Graceful degradation if `gh` not available (skip warning)

**Trade-offs**:
- Requires `gh` CLI for size estimation (optional feature)
- Extra API call adds slight latency
- Public repos don't require authentication

**Sources**:
- GitHub REST API: Repositories endpoint returns `size` field

### Unknown 6: Retry Logic Implementation

**Question**: How should clone retry logic be implemented?

**Options Considered**:
1. **Simple Retry with Sleep** - Try once, sleep 5s, try again
2. **Exponential Backoff** - Increase delay between retries
3. **tenacity Library** - Use retry library for robust retry handling

**Decision**: Simple Retry with Sleep

**Rationale**:
- Only one retry is specified in requirements
- 5-second fixed delay is sufficient for transient network issues
- No need for exponential backoff with single retry
- Avoids new dependency

**Trade-offs**:
- Less sophisticated than library solutions
- Hardcoded retry count (1) and delay (5s) - acceptable per spec

**Sources**:
- Spec clarification: "System waits 5 seconds, retries clone operation once"

## Key Findings

1. **Git CLI is Sufficient**: No need for git libraries - CLI provides all needed functionality with simpler integration
2. **Pydantic Consistency**: URL validation should use Pydantic validators to match existing patterns
3. **Workspace Preservation**: Brownfield workspaces need different lifecycle than temp workspaces - must not be cleaned up automatically
4. **Change Summary via Git**: `git status --porcelain` provides stable, machine-readable output for change detection
5. **Optional Size Warning**: Repository size estimation is nice-to-have and should gracefully degrade if `gh` CLI unavailable
6. **CRITICAL - Existing Git Init Must Be Bypassed**: The existing `RunEvaluationCommand._init_git_repo()` method in `cli/commands/evaluation.py` performs several operations that would corrupt a cloned repository:
   - `git init` - would reinitialize an existing repo
   - Creates `.gitkeep` file and initial commit - adds unwanted files
   - Creates a dummy bare remote at `eval_folder/remote.git` - overwrites GitHub origin
   - Pushes to the dummy remote

   For brownfield mode, **all of this must be skipped**. The clone operation replaces the entire git init flow.

## Recommendations

### Implementation Approach

1. **Create RepositorySource Model**: New Pydantic model in `config/models.py` with URL validation
2. **Add Git Operations Module**: New module `core/git_operations.py` for clone, status, and change detection
3. **Extend EvaluationConfig**: Add optional `repository_source` field
4. **Modify Evaluation.start()**: Check for repository source, clone if present instead of creating empty workspace
5. **Extend EvaluationReport**: Add `ChangeSummary` to report model
6. **Update Report Generator**: Generate change summary from git status after evaluation

### Integration Points

| Component | Change Type | Description |
|-----------|-------------|-------------|
| `config/models.py` | Add model | RepositorySource with URL validation |
| `cli/commands/evaluation.py` | **Modify (CRITICAL)** | Skip `_init_git_repo()` for brownfield, add `_clone_repository()` |
| `cli/commands/suite.py` | Modify | Pass repository_source from EvaluationConfig to run_evaluation() |
| `core/evaluation.py` | Modify | Support cloning in `start()` method |
| `core/git_operations.py` | New file | Clone, status, change detection functions |
| `report/models.py` | Add model | ChangeSummary dataclass |
| `report/generator.py` | Modify | Generate change summary |
| `workflows/base.py` | Modify | Pass repository source to evaluation |
