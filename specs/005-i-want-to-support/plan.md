# Implementation Plan: Brownfield Repository Support

**Feature**: Brownfield Repository Support
**Branch**: `005-i-want-to-support`
**Date**: 2026-02-02
**Status**: Ready for Implementation

---

## Technical Context

### Language & Runtime

| Aspect | Value |
|--------|-------|
| Primary Language | Python |
| Runtime/Version | Python 3.10+ |
| Package Manager | pip (setuptools) |

### Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| pydantic | >=2.0.0,<3.0.0 | Existing - Data validation and URL validation |
| pydantic-settings | >=2.0.0,<3.0.0 | Existing - Environment configuration |
| structlog | >=24.1.0,<25.0.0 | Existing - Structured logging |
| pytest | >=7.0 | Existing (dev) - Test framework |
| pytest-asyncio | >=0.21.0 | Existing (dev) - Async test support |

**No new dependencies required** - Git operations use subprocess with system git CLI.

### Platform & Environment

| Aspect | Value |
|--------|-------|
| Target Platform | CLI tool (macOS, Linux) |
| Minimum Requirements | Python 3.10+, Git CLI installed |
| Environment Variables | None new required |

### Testing Approach

| Aspect | Value |
|--------|-------|
| Test Framework | pytest with pytest-asyncio |
| Test Location | tests/unit/, tests/integration/, tests/e2e/ |
| Required Coverage | Critical paths (URL validation, clone, change detection) |

**Test Types**:
- Unit: RepositorySource validation, ChangeSummary creation, git command building
- Integration: Clone operation with test repository, change detection
- E2E: Full brownfield evaluation with real repository clone

### Constraints

- **Git Dependency**: Git CLI must be installed on the system
- **Public Repositories Only**: Initial implementation supports only public GitHub repositories
- **HTTPS URLs Only**: SSH URLs are not supported
- **Workspace Preservation**: Brownfield workspaces must not be auto-cleaned
- **Network Required**: Clone operations require network access to GitHub

---

## Constitution Check

**Constitution Source**: `.projspec/memory/constitution.md`
**Check Date**: 2026-02-02

### Principle Compliance

| Principle | Description | Status | Notes |
|-----------|-------------|--------|-------|
| I. User-Centric Design | Features prioritize user experience | PASS | Clear error messages, workspace preserved for inspection |
| II. Maintainability First | Code clarity over cleverness | PASS | Simple subprocess git calls, follows existing patterns |
| III. Test-Driven Confidence | New functionality requires tests | PASS | Unit, integration, and E2E tests planned |
| IV. Documentation as Code | Documentation is a deliverable | PASS | Spec, research, data-model, plan documents created |
| V. Accuracy & Reliability | Results must be correct and reproducible | PASS | Git-based change detection is deterministic |
| VI. Extensibility | Easy addition of new features | PASS | RepositorySource model easily extended for future features |

### Compliance Details

#### Principles with Full Compliance (PASS)

- **I. User-Centric Design**: Implementation prioritizes UX:
  - Clear error messages for invalid URLs with suggestions
  - Workspace path included in report for easy navigation
  - Change summary provides immediate insight into modifications

- **II. Maintainability First**: Simple, readable implementation:
  - Git operations via subprocess (universally understood)
  - Follows existing Pydantic model patterns
  - No complex abstractions or new paradigms

- **III. Test-Driven Confidence**: Comprehensive test coverage:
  - Unit tests for URL validation edge cases
  - Integration tests for actual git clone operations
  - E2E tests for complete brownfield workflow

- **IV. Documentation as Code**: Full documentation suite:
  - spec.md with requirements and acceptance criteria
  - research.md with technical decisions
  - data-model.md with entity definitions
  - plan.md (this document) with implementation guidance

- **V. Accuracy & Reliability**: Deterministic change detection:
  - Git status provides consistent results
  - No heuristics or estimation in change tracking
  - Clone operation success/failure is unambiguous

- **VI. Extensibility**: Future-ready design:
  - RepositorySource model can add authentication fields later
  - ChangeSummary can include diff content if needed
  - Clone operation can support other git hosts

### Gate Status

**Constitution Check Result**: PASS

**Criteria**: All principles are PASS with documented compliance

**Action Required**: None - proceed to project structure

---

## Project Structure

### Documentation Layout

```
specs/005-i-want-to-support/
├── spec.md              # Feature specification (requirements, scenarios)
├── research.md          # Technical research and decisions
├── data-model.md        # Entity definitions and schemas
├── plan.md              # Implementation plan (this document)
├── quickstart.md        # Getting started guide
└── tasks.md             # Implementation task list (to be generated)
```

### Source Code Layout

Based on project type: **Python CLI Tool**

```
src/claude_evaluator/
├── config/
│   └── models.py          # MODIFY: Add RepositorySource model
├── core/
│   ├── evaluation.py      # MODIFY: Support repository cloning in start()
│   ├── git_operations.py  # CREATE: Git clone and status operations
│   └── exceptions.py      # MODIFY: Add git-related exceptions
├── report/
│   ├── models.py          # MODIFY: Add ChangeSummary, workspace_path
│   └── generator.py       # MODIFY: Generate change summary for brownfield
└── workflows/
    ├── base.py            # MODIFY: Pass repository source through workflow
    └── direct.py          # MODIFY: Support brownfield mode
```

### Directory Purposes

| Directory | Purpose |
|-----------|---------|
| src/claude_evaluator/config/ | Configuration models and loading |
| src/claude_evaluator/core/ | Core evaluation logic and agents |
| src/claude_evaluator/report/ | Report generation and models |
| src/claude_evaluator/workflows/ | Workflow orchestration |
| tests/unit/ | Unit tests with mocked dependencies |
| tests/integration/ | Integration tests with real git operations |
| tests/e2e/ | End-to-end tests with full workflow |

### File-to-Requirement Mapping

| File | Requirements | Purpose |
|------|--------------|---------|
| config/models.py | FR-001, FR-002, FR-003 | RepositorySource model with URL validation |
| cli/commands/evaluation.py | FR-002, FR-003, FR-004 | **CRITICAL**: Clone repository, skip git init for brownfield |
| cli/commands/suite.py | FR-001, FR-002 | Pass repository_source from YAML config |
| core/evaluation.py | FR-002, FR-004 | Clone integration in start(), workspace preservation |
| core/git_operations.py | FR-002, FR-003, FR-005 | Clone, checkout, and change detection |
| core/exceptions.py | Edge Cases | Git-specific exception classes |
| report/models.py | FR-004, FR-005 | ChangeSummary model, workspace_path field |
| report/generator.py | FR-005, US-003 | Generate change summary from git status |

### New Files to Create

| File Path | Type | Description |
|-----------|------|-------------|
| src/claude_evaluator/core/git_operations.py | source | Git clone, checkout, status, and change detection |
| tests/unit/test_git_operations.py | test | Unit tests for git operations (mocked subprocess) |
| tests/unit/test_repository_source.py | test | Unit tests for RepositorySource validation |
| tests/integration/test_brownfield_clone.py | test | Integration tests with real git clone |
| tests/e2e/test_brownfield_evaluation.py | test | E2E tests for complete brownfield workflow |

### Files to Modify

| File Path | Type | Changes |
|-----------|------|---------|
| src/claude_evaluator/config/models.py | source | Add RepositorySource model, add to EvaluationConfig |
| src/claude_evaluator/cli/commands/evaluation.py | source | **CRITICAL**: Skip `_init_git_repo()` for brownfield, add `_clone_repository()` |
| src/claude_evaluator/cli/commands/suite.py | source | Pass `repository_source` from config to `run_evaluation()` |
| src/claude_evaluator/core/evaluation.py | source | Modify start() to support cloning, preserve brownfield workspaces |
| src/claude_evaluator/core/exceptions.py | source | Add CloneError, InvalidRepositoryError exceptions |
| src/claude_evaluator/report/models.py | source | Add ChangeSummary model, workspace_path to EvaluationReport |
| src/claude_evaluator/report/generator.py | source | Generate change summary for brownfield evaluations |
| src/claude_evaluator/workflows/base.py | source | Pass repository source configuration to evaluation |
| src/claude_evaluator/workflows/direct.py | source | Support brownfield mode in direct workflow |

---

## Implementation Guidance

### Phase 1: Data Models

**Goal**: Create RepositorySource and ChangeSummary Pydantic models

**Changes**:
1. Add `RepositorySource` model to `config/models.py`:
   - `url: str` with custom validator for GitHub HTTPS URLs
   - `ref: str | None` with default None
   - `depth: int | str` with default 1 and validation
2. Add `repository_source: RepositorySource | None` to `EvaluationConfig`
3. Add `ChangeSummary` model to `report/models.py`:
   - `files_modified: list[str]`
   - `files_added: list[str]`
   - `files_deleted: list[str]`
   - `total_changes: int` (computed)
4. Add `workspace_path: str | None` and `change_summary: ChangeSummary | None` to `EvaluationReport`

**Verification**: Unit tests for URL validation (valid HTTPS, invalid SSH, missing owner/repo)

### Phase 2: Git Operations Module

**Goal**: Create git_operations.py with clone and status functions

**Changes**:
1. Create `core/git_operations.py` with:
   - `async def clone_repository(source: RepositorySource, target_path: Path) -> CloneResult`
   - `def build_clone_command(source: RepositorySource, target_path: Path) -> list[str]`
   - `async def get_change_summary(workspace_path: Path) -> ChangeSummary`
   - `def parse_git_status(output: str) -> ChangeSummary`
2. Add exception classes to `core/exceptions.py`:
   - `CloneError(ClaudeEvaluatorError)` with attributes: url, error_message, retry_attempted
   - `InvalidRepositoryError(ClaudeEvaluatorError)` for URL validation failures
3. Implement retry logic:
   - On network failure, wait 5 seconds, retry once
   - On second failure, raise CloneError with details

**Key Implementation**:
```python
async def clone_repository(source: RepositorySource, target_path: Path) -> CloneResult:
    """Clone a repository to target path with retry logic."""
    cmd = build_clone_command(source, target_path)

    for attempt in range(2):  # Max 2 attempts
        try:
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                return CloneResult(success=True, workspace_path=str(target_path), ...)

            # Check if error is retriable (network-related)
            if attempt == 0 and is_network_error(stderr.decode()):
                await asyncio.sleep(5)
                continue

            raise CloneError(url=source.url, error_message=stderr.decode())
        except Exception as e:
            if attempt == 0:
                await asyncio.sleep(5)
                continue
            raise
```

**Verification**: Unit tests with mocked subprocess, integration tests with real git

### Phase 3: CLI Command Integration (CRITICAL)

**Goal**: Modify `RunEvaluationCommand` to support brownfield mode by skipping git init

**Context**: The existing `run_evaluation()` method in `cli/commands/evaluation.py` unconditionally calls `_init_git_repo()` which:
1. Runs `git init` (would corrupt a cloned repo)
2. Creates `.gitkeep` and initial commit (adds unwanted files)
3. Creates a dummy bare remote (overwrites GitHub remote)
4. Pushes to the dummy remote

For brownfield mode, we must **skip all of this** and clone the repository instead.

**Changes**:
1. Add `repository_source: RepositorySource | None = None` parameter to `run_evaluation()`
2. Add new `_clone_repository()` method that:
   - Clones the repository to workspace path
   - Checks out the specified ref (if provided)
   - Preserves the original remote
   - Optionally sets local git user.name/email if not set
3. Modify workspace setup logic:
   ```python
   if repository_source:
       # Brownfield mode: clone instead of init
       self._clone_repository(repository_source, workspace_path, verbose)
   else:
       # Greenfield mode: existing git init logic
       self._init_git_repo(workspace_path, eval_folder, verbose)
   ```
4. Update `RunSuiteCommand` to pass `repository_source` from `EvaluationConfig`

**Key Implementation**:
```python
def _clone_repository(
    self,
    source: RepositorySource,
    workspace_path: Path,
    verbose: bool
) -> None:
    """Clone a repository for brownfield evaluation."""
    # Build clone command
    cmd = ["git", "clone"]

    # Add depth flag (shallow clone)
    if source.depth != "full":
        cmd.extend(["--depth", str(source.depth)])

    # Add branch flag if ref specified
    if source.ref:
        cmd.extend(["--branch", source.ref])

    # Add URL and target path
    cmd.extend([source.url, str(workspace_path)])

    if verbose:
        print(f"Cloning {source.url}...")

    # Execute with retry logic
    for attempt in range(2):
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            if verbose:
                print(f"Cloned to {workspace_path}")
            return

        # Retry on network error
        if attempt == 0 and self._is_network_error(result.stderr):
            if verbose:
                print("Network error, retrying in 5 seconds...")
            time.sleep(5)
            continue

        raise CloneError(source.url, result.stderr)

    raise CloneError(source.url, "Clone failed after retry")
```

**Verification**: E2E test cloning real public repository, verifying no `.gitkeep` file created

### Phase 4: Evaluation Model Integration

**Goal**: Modify Evaluation.start() to support brownfield cloning

**Changes**:
1. Modify `Evaluation.start()` signature:
   - Add `repository_source: RepositorySource | None = None` parameter
2. Update workspace creation logic:
   - If `repository_source` is provided:
     - Create workspace in `brownfield/` directory with unique name
     - Clone repository to workspace
     - Set `_owns_workspace = False` to prevent cleanup
   - Otherwise: Use existing temp directory logic
3. Store clone result metadata for report generation
4. Add logging for brownfield mode entry

**Key Implementation**:
```python
def start(self, workspace_path: str | None = None,
          repository_source: RepositorySource | None = None) -> None:
    """Start evaluation with optional repository source for brownfield mode."""
    if repository_source:
        # Brownfield mode: clone repository
        workspace_base = Path(tempfile.gettempdir()) / "claude_evaluator" / "brownfield"
        workspace_base.mkdir(parents=True, exist_ok=True)
        workspace_name = f"eval_{self.id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.workspace_path = str(workspace_base / workspace_name)

        result = await clone_repository(repository_source, Path(self.workspace_path))
        if not result.success:
            raise CloneError(...)

        self._owns_workspace = False  # Never cleanup brownfield workspaces
    elif workspace_path:
        # Existing externally-provided workspace logic
        ...
    else:
        # Existing temp directory logic
        ...
```

**Verification**: Integration test cloning real repository, verifying workspace contents

### Phase 4: Change Summary Generation

**Goal**: Generate change summary after brownfield evaluation

**Changes**:
1. Add `async def generate_change_summary(workspace_path: Path) -> ChangeSummary` to git_operations.py
2. Implement `git status --porcelain` parsing:
   - `??` prefix = untracked (added)
   - `M ` prefix = modified
   - `D ` prefix = deleted
   - `A ` prefix = staged added
3. Modify `report/generator.py`:
   - Check if evaluation has repository_source
   - Generate change summary from workspace
   - Include in report

**Key Implementation**:
```python
def parse_git_status(output: str) -> ChangeSummary:
    """Parse git status --porcelain output into ChangeSummary."""
    modified, added, deleted = [], [], []

    for line in output.strip().split('\n'):
        if not line:
            continue
        status = line[:2]
        filepath = line[3:]

        if status in ('??', 'A '):
            added.append(filepath)
        elif status in ('M ', ' M', 'MM'):
            modified.append(filepath)
        elif status in ('D ', ' D'):
            deleted.append(filepath)

    return ChangeSummary(
        files_modified=modified,
        files_added=added,
        files_deleted=deleted,
    )
```

**Verification**: Unit tests with various git status outputs

### Phase 5: Workflow Integration

**Goal**: Wire brownfield support through workflows

**Changes**:
1. Update `EvaluationConfig` YAML parsing to handle `repository_source`
2. Modify workflow execution to pass repository_source to Evaluation
3. Update report generation to include change summary
4. Add size warning if repository estimated > 500MB

**Verification**: E2E test with full brownfield evaluation

### Phase 6: Error Handling and Edge Cases

**Goal**: Handle all error scenarios gracefully

**Changes**:
1. Invalid URL format → Clear error with format suggestion
2. SSH URL → Error suggesting HTTPS format
3. Repository not found → Error with attempted URL
4. Network failure → Retry once after 5s, then fail
5. Branch not found → Error listing available branches (if possible)
6. Empty repository → Allow clone, proceed normally

**Verification**: Unit tests for each error scenario

---

## Success Metrics

| Metric | Target | Verification |
|--------|--------|--------------|
| URL Validation | 100% of invalid URLs rejected | Unit tests with edge cases |
| Clone Success | Public repos clone successfully | Integration test with sample repos |
| Change Detection | 100% accuracy vs manual git status | Compare ChangeSummary to git status output |
| Workspace Preservation | All brownfield workspaces preserved | Verify workspace exists after evaluation |
| Error Clarity | All errors have actionable messages | Manual review of error strings |
| Clone Performance | < 30s for repos under 100MB | Timed integration test |

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Git not installed on system | Low | High | Check git availability at startup, clear error message |
| Network timeout during clone | Medium | Medium | 5-second retry with clear timeout message |
| Large repository slows evaluation | Medium | Low | Size warning before clone, shallow clone by default |
| Git status format varies by version | Low | Medium | Use --porcelain for stable output format |
| Workspace path conflicts | Low | Low | Use UUID + timestamp in workspace names |
| GitHub rate limiting | Low | Medium | Use authenticated requests if gh available (future) |

---

## Next Steps

The implementation plan is complete. To continue:

**Recommended**: Generate implementation tasks
```
/spectra:tasks
```

This will:
1. Read the plan artifacts (spec.md, research.md, data-model.md, plan.md)
2. Generate a dependency-ordered task list
3. Create tasks.md with actionable implementation steps
