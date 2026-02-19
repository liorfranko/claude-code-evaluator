# Development Guide

This guide covers contributing to Claude Code Evaluator, including setup, testing, and extending the tool.

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- Docker (optional, for sandbox testing)

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/claude-code-evaluator.git
cd claude-code-evaluator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Verify installation
claude-evaluator --help
pytest --version
ruff --version
```

### Project Structure

```
claude-code-evaluator/
├── src/
│   └── claude_evaluator/
│       ├── cli/           # CLI entry point and commands
│       ├── agents/        # Worker and Developer agents
│       ├── workflows/     # Workflow strategies
│       ├── evaluation/    # Evaluation orchestration
│       ├── scoring/       # Scoring and analysis
│       ├── benchmark/     # Benchmark system
│       ├── config/        # Configuration and settings
│       ├── models/        # Pydantic data models
│       ├── metrics/       # Metrics collection
│       ├── report/        # Report generation
│       └── sandbox/       # Execution isolation
├── tests/                 # Test suite
├── benchmarks/           # Example benchmark configs
├── docs/                 # Documentation
├── pyproject.toml        # Project configuration
└── Dockerfile            # Container build
```

## Code Conventions

### Python Style

- **Python 3.10+** — Use modern syntax (`X | None` not `Optional[X]`)
- **Type annotations** — Required on all public functions
- **Docstrings** — Google style format
- **Line length** — 88 characters (ruff default)
- **Quotes** — Double quotes for strings
- **Imports** — Sorted: stdlib, third-party, local

### Example

```python
"""Module docstring describing purpose."""

from pathlib import Path

import structlog
from pydantic import BaseModel

from claude_evaluator.models.base import BaseSchema

__all__ = ["MyClass"]

logger = structlog.get_logger(__name__)


class MyClass(BaseSchema):
    """Class docstring.

    Args:
        name: The name parameter.
        value: Optional value with default.
    """

    name: str
    value: int | None = None

    def process(self, data: list[str]) -> dict[str, int]:
        """Process data and return counts.

        Args:
            data: List of strings to process.

        Returns:
            Dictionary mapping strings to counts.
        """
        logger.info("processing_data", count=len(data))
        return {item: data.count(item) for item in set(data)}
```

### Module Structure

Each module should:
- Have a clear single responsibility
- Define `__all__` explicitly
- Use relative imports within the package
- Include module-level docstring

### Pydantic Models

```python
from claude_evaluator.models.base import BaseSchema


class MyModel(BaseSchema):
    """Model description."""

    required_field: str
    optional_field: int | None = None
    list_field: list[str] = []
```

### Logging

```python
from structlog import get_logger

logger = get_logger(__name__)

# Use structured logging
logger.info("operation_started", param=value)
logger.error("operation_failed", error=str(e), context=ctx)
```

### Exceptions

```python
from claude_evaluator.exceptions import ClaudeEvaluatorError


class MyModuleError(ClaudeEvaluatorError):
    """Base exception for this module."""
    pass


class SpecificError(MyModuleError):
    """Raised when specific condition occurs."""
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_workflows.py

# Run specific test
pytest tests/test_workflows.py::test_direct_workflow

# Run with coverage
pytest --cov=claude_evaluator --cov-report=html
```

### Test Structure

```
tests/
├── conftest.py           # Shared fixtures
├── test_cli/             # CLI tests
├── test_workflows/       # Workflow tests
├── test_agents/          # Agent tests
├── test_scoring/         # Scoring tests
├── test_benchmark/       # Benchmark tests
└── test_integration/     # Integration tests
```

### Writing Tests

```python
import pytest
from claude_evaluator.workflows.direct import DirectWorkflow


class TestDirectWorkflow:
    """Tests for DirectWorkflow."""

    @pytest.fixture
    def workflow(self, mock_worker, mock_developer):
        """Create workflow with mocked agents."""
        return DirectWorkflow(
            worker=mock_worker,
            developer=mock_developer,
        )

    def test_execute_success(self, workflow, evaluation):
        """Test successful workflow execution."""
        workflow.execute(evaluation)

        assert evaluation.status == EvaluationStatus.COMPLETED
        assert evaluation.metrics.total_cost_usd > 0

    def test_execute_handles_error(self, workflow, evaluation, mock_worker):
        """Test error handling during execution."""
        mock_worker.execute_query.side_effect = RuntimeError("API error")

        with pytest.raises(WorkflowError):
            workflow.execute(evaluation)

        assert evaluation.status == EvaluationStatus.FAILED
```

### Fixtures

Common fixtures in `conftest.py`:

```python
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_worker():
    """Mock WorkerAgent."""
    worker = MagicMock()
    worker.execute_query.return_value = QueryMetrics(...)
    return worker


@pytest.fixture
def evaluation():
    """Sample evaluation for testing."""
    return Evaluation(
        id="test-eval",
        task="Test task",
        workflow_type=WorkflowType.DIRECT,
    )
```

## Linting and Formatting

### Ruff

```bash
# Check for issues
ruff check src/

# Fix auto-fixable issues
ruff check src/ --fix

# Check formatting
ruff format --check src/

# Apply formatting
ruff format src/
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Extending the Tool

### Adding a New Workflow

1. **Create workflow class**:

```python
# src/claude_evaluator/workflows/my_workflow.py
from claude_evaluator.workflows.base import BaseWorkflow


class MyWorkflow(BaseWorkflow):
    """Custom workflow implementation."""

    def _execute_workflow(self, evaluation: Evaluation) -> None:
        """Execute the workflow logic."""
        # Phase 1
        self.worker.execute_query(
            prompt="Phase 1 prompt",
            permission_mode=PermissionMode.PLAN,
        )

        # Phase 2
        self.worker.execute_query(
            prompt="Phase 2 prompt",
            permission_mode=PermissionMode.ACCEPT_EDITS,
        )
```

2. **Add to WorkflowType enum**:

```python
# src/claude_evaluator/models/enums.py
class WorkflowType(str, Enum):
    DIRECT = "direct"
    PLAN_THEN_IMPLEMENT = "plan_then_implement"
    MULTI_COMMAND = "multi_command"
    MY_WORKFLOW = "my_workflow"  # Add here
```

3. **Register in factory**:

```python
# src/claude_evaluator/workflows/agent_factory.py
def create_workflow(workflow_type: WorkflowType, ...) -> BaseWorkflow:
    if workflow_type == WorkflowType.MY_WORKFLOW:
        return MyWorkflow(...)
```

4. **Add tests**:

```python
# tests/test_workflows/test_my_workflow.py
class TestMyWorkflow:
    def test_execute(self, workflow, evaluation):
        workflow.execute(evaluation)
        assert evaluation.status == EvaluationStatus.COMPLETED
```

### Adding a New Command

1. **Create command class**:

```python
# src/claude_evaluator/cli/commands/my_command.py
from claude_evaluator.cli.commands.base import BaseCommand, CommandResult


class MyCommand(BaseCommand):
    """My custom command."""

    def execute(self) -> CommandResult:
        """Execute the command."""
        # Implementation
        return CommandResult(success=True, message="Done")
```

2. **Register in `__init__.py`**:

```python
# src/claude_evaluator/cli/commands/__init__.py
from .my_command import MyCommand

__all__ = [..., "MyCommand"]
```

3. **Add argument parsing**:

```python
# src/claude_evaluator/cli/parser.py
def create_parser() -> argparse.ArgumentParser:
    parser.add_argument(
        "--my-command",
        action="store_true",
        help="Run my custom command",
    )
```

4. **Update dispatch**:

```python
# src/claude_evaluator/cli/main.py
def _dispatch(args: argparse.Namespace) -> int:
    if args.my_command:
        command = MyCommand(args)
        result = command.execute()
        return 0 if result.success else 1
```

### Adding a New Reviewer

1. **Create reviewer class**:

```python
# src/claude_evaluator/scoring/reviewers/my_reviewer.py
from claude_evaluator.scoring.reviewers.base import BaseReviewer


class MyReviewer(BaseReviewer):
    """Custom reviewer for specific dimension."""

    dimension = "my_dimension"

    def review(self, context: ReviewContext) -> ReviewerOutput:
        """Perform the review."""
        # Analysis logic
        return ReviewerOutput(
            score=score,
            reasoning="...",
            details={...},
        )
```

2. **Register in registry**:

```python
# src/claude_evaluator/scoring/reviewers/registry.py
from .my_reviewer import MyReviewer

REVIEWERS = {
    ...,
    "my_dimension": MyReviewer,
}
```

### Adding a Quality Check

```python
# src/claude_evaluator/scoring/checks/my_checks.py
from claude_evaluator.scoring.checks.base import BaseCheck, Issue


class MyCheck(BaseCheck):
    """Check for specific code pattern."""

    def check(self, code: str, filename: str) -> list[Issue]:
        """Run the check."""
        issues = []

        # Detection logic
        if problematic_pattern_found:
            issues.append(Issue(
                type="my_issue",
                severity="medium",
                location=f"{filename}:{line}",
                description="Description of the issue",
            ))

        return issues
```

## Documentation

### Building Docs

Documentation is in Markdown in the `docs/` directory.

### Style Guide

- Use headers hierarchically (H1 for title, H2 for sections)
- Include code examples with language hints
- Use tables for configuration options
- Add cross-references between documents

## Release Process

1. **Update version**:
```bash
# In pyproject.toml
version = "X.Y.Z"
```

2. **Update changelog**:
```bash
# In CHANGELOG.md
## [X.Y.Z] - YYYY-MM-DD
### Added
- New feature
### Fixed
- Bug fix
```

3. **Create release**:
```bash
git tag -a vX.Y.Z -m "Release X.Y.Z"
git push origin vX.Y.Z
```

## Troubleshooting

### Common Issues

**Import errors**:
```bash
# Ensure package is installed
pip install -e .
```

**Test failures**:
```bash
# Run with verbose output
pytest -v --tb=long
```

**Linting failures**:
```bash
# Auto-fix what's possible
ruff check src/ --fix
ruff format src/
```

### Debug Mode

```python
import structlog

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
)
```

Or via environment:
```bash
export CLAUDE_LOG_LEVEL=DEBUG
claude-evaluator --benchmark file.yaml --verbose
```
