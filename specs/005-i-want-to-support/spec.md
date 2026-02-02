# Feature Specification: Brownfield Repository Support

## Metadata

| Field | Value |
|-------|-------|
| Branch | `005-i-want-to-support` |
| Date | 2026-02-02 |
| Status | Ready for Review |
| Input | Support brownfield repositories by allowing users to provide a remote GitHub repository and run a prompt to add a new feature to that repo |

---

## User Scenarios & Testing

### Primary Scenarios

#### US-001: Clone and Modify External Repository

**As a** developer using the evaluator
**I want to** provide a GitHub repository URL and a feature prompt
**So that** I can test Claude's ability to add features to existing codebases (brownfield development)

**Acceptance Criteria:**
- [ ] User can specify a GitHub repository URL in the evaluation configuration
- [ ] The system clones the specified repository into an isolated workspace
- [ ] The Worker agent executes the feature prompt against the cloned codebase
- [ ] The modified workspace is preserved after execution for user inspection

**Priority:** High

#### US-002: Use Specific Branch from Repository

**As a** developer targeting a specific version of a codebase
**I want to** specify a branch or ref when cloning a repository
**So that** I can test feature additions against a particular state of the codebase

**Acceptance Criteria:**
- [ ] User can optionally specify a branch, tag, or commit ref
- [ ] The system clones only the specified ref (not the full history)
- [ ] The evaluation report indicates which ref was used

**Priority:** Medium

#### US-003: Review Changes After Evaluation

**As a** developer reviewing evaluation results
**I want to** see what changes Claude made to the repository
**So that** I can assess the quality of the feature implementation

**Acceptance Criteria:**
- [ ] The evaluation report includes the workspace path for inspection
- [ ] The report includes a summary of modified, added, and deleted files
- [ ] The user can navigate to the workspace and view git diff of changes

**Priority:** High

### Edge Cases

| Case | Expected Behavior |
|------|-------------------|
| Invalid repository URL format | System validates URL format before attempting clone and returns descriptive error |
| SSH URL provided | System rejects with error message suggesting HTTPS URL format instead |
| Repository requires authentication | System detects authentication failure and reports clear error with remediation steps |
| Repository does not exist or is inaccessible | System reports repository not found error with the URL that was attempted |
| Empty repository (no commits) | System clones empty repository and proceeds with evaluation normally |
| Network failure during clone | System waits 5 seconds, retries clone operation once, then fails with network error message |
| Specified branch/ref does not exist | System reports branch not found error and lists available branches if possible |

---

## Requirements

### Functional Requirements

#### FR-001: Repository URL Configuration

The system shall accept an optional repository URL in the evaluation configuration. The URL must be an HTTPS GitHub URL in one of these formats: `https://github.com/owner/repo` or `https://github.com/owner/repo.git`. SSH-style URLs (`git@github.com:`) are not supported in the initial implementation.

**Verification:** Create evaluation config with repository_url field; verify system parses and validates the URL format correctly; verify SSH URLs are rejected with a helpful error message.

#### FR-002: Repository Cloning

When a repository URL is provided, the system shall clone the repository into the evaluation workspace instead of initializing an empty repository. The clone operation shall use shallow clone with a configurable depth parameter. The depth parameter accepts an integer (number of commits) or the value "full" for complete history. The default depth is 1 commit.

**Verification:** Run evaluation with repository_url; verify workspace contains cloned repository content. Test with depth=1 (default), depth=10, and depth="full" to verify history availability.

#### FR-003: Branch/Ref Selection

The system shall accept an optional branch, tag, or commit ref when cloning a repository. When specified, the system shall clone only that ref. When not specified, the system shall clone the repository's default branch.

**Verification:** Run evaluation with repository_url and branch specified; verify workspace is checked out to the specified branch.

#### FR-004: Workspace Preservation

When a repository is cloned (brownfield mode), the system shall preserve the workspace after evaluation completion regardless of the cleanup configuration. Brownfield workspaces shall be stored in a `brownfield/` subdirectory within the existing workspace directory pattern. The workspace path shall be included in the evaluation report.

**Verification:** Run brownfield evaluation; verify workspace exists in the brownfield/ subdirectory after completion and report contains the full workspace path.

#### FR-005: Change Summary in Report

The evaluation report shall include a summary of changes made to the repository during evaluation. This summary shall include counts of modified, added, and deleted files, as well as the list of affected file paths.

**Verification:** Run brownfield evaluation that modifies files; verify report contains accurate change summary.

### Constraints

| Constraint | Description |
|------------|-------------|
| Authentication | Initial implementation supports only public repositories; authenticated access is out of scope |
| URL Format | Only HTTPS GitHub URLs are supported; SSH URLs (git@github.com:) are not supported |
| Clone Depth | Default shallow clone depth of 1 commit; configurable via depth parameter (integer or "full") |
| Repository Size | No explicit size limit; system displays informational warning for repositories estimated over 500MB |

---

## Key Entities

### Repository Source

**Description:** Represents the external repository to be cloned for brownfield evaluation

| Attribute | Description | Constraints |
|-----------|-------------|-------------|
| url | The GitHub repository URL | Required; must be valid HTTPS GitHub URL format |
| ref | Optional branch, tag, or commit to checkout | Optional; defaults to repository's default branch |
| depth | Clone depth (number of commits or "full") | Optional; positive integer or "full"; defaults to 1 |

### Change Summary

**Description:** Represents the modifications made to the repository during evaluation

| Attribute | Description | Constraints |
|-----------|-------------|-------------|
| files_modified | List of file paths that were changed | May be empty |
| files_added | List of new file paths created | May be empty |
| files_deleted | List of file paths removed | May be empty |
| total_changes | Total count of all changes | Non-negative integer |

### Entity Relationships

- Evaluation contains optional Repository Source (one-to-one)
- Evaluation Report contains Change Summary (one-to-one)
- Repository Source determines initial workspace state

---

## Success Criteria

### SC-001: Clone Performance

**Measure:** Time to clone repository and begin evaluation
**Target:** Repository clone completes within 30 seconds for repositories under 100MB
**Verification Method:** Time the clone operation for sample repositories of varying sizes

### SC-002: Change Tracking Accuracy

**Measure:** Accuracy of change summary in evaluation report
**Target:** 100% accuracy in reporting modified, added, and deleted files
**Verification Method:** Compare report change summary against actual `git status` output in workspace

### SC-003: Workspace Integrity

**Measure:** Preservation of workspace after brownfield evaluation
**Target:** 100% of brownfield evaluations preserve workspace with all changes intact
**Verification Method:** Verify workspace exists and contains expected changes after evaluation completion

---

## Assumptions

| ID | Assumption | Impact if Wrong | Validated |
|----|------------|-----------------|-----------|
| A-001 | Users have network access to GitHub from the evaluation environment | Clone operations would fail; need offline mode or alternative | No |
| A-002 | Public repositories are sufficient for initial implementation | Would need to add authentication support sooner | No |
| A-003 | Shallow clone provides sufficient context for Claude to understand the codebase | May need to support full clone option if context is insufficient | No |

---

## Open Questions

### Q-001: Authentication method for private repositories

- **Question**: How should users authenticate to access private GitHub repositories in future iterations?
- **Why Needed**: Determines security architecture and user experience for accessing private codebases
- **Suggested Default**: Defer private repository support to a future iteration; initial release supports public repos only
- **Status**: Open
- **Impacts**: FR-001, Constraints table

### Q-002: Post-evaluation workflow for changes

- **Question**: Should the system offer to create a branch or commit the changes automatically, or leave all git operations to the user?
- **Why Needed**: Affects whether we need additional git automation features in the report/output phase
- **Suggested Default**: Leave workspace with uncommitted changes; user decides how to handle them
- **Status**: Open
- **Impacts**: FR-004, US-003

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-02 | Claude (spectra) | Initial draft from feature description |
| 0.2 | 2026-02-02 | Claude (spectra/clarify) | Resolved 5 clarifications: HTTPS-only URLs, configurable clone depth, workspace location, retry delay (5s), size warning threshold (500MB) |
