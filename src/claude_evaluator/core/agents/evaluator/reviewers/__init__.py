"""Phase reviewers for multi-phase evaluation.

This module provides auto-registration of phase reviewer implementations.
Each reviewer module in this package should define a REVIEWER variable
that implements the BasePhaseReviewer protocol.

The auto-registration system:
1. Discovers all Python modules in the reviewers/ directory
2. Imports modules that define a REVIEWER variable
3. Collects reviewers into the REVIEWERS dictionary keyed by phase name
"""

import importlib
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING, Dict

import structlog

if TYPE_CHECKING:
    from claude_evaluator.core.agents.evaluator.protocols import BasePhaseReviewer

logger = structlog.get_logger()

# Dictionary to store registered reviewers
REVIEWERS: Dict[str, "BasePhaseReviewer"] = {}


def _discover_reviewers() -> None:
    """Discover and register all phase reviewers in this package.

    This function is called automatically on module import to populate
    the REVIEWERS dictionary with all available reviewer implementations.

    Each reviewer module should:
    1. Define a REVIEWER variable that implements BasePhaseReviewer
    2. Have the reviewer's phase_name match the expected phase
    """
    package_dir = Path(__file__).parent

    # Iterate through all Python files in the reviewers directory
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name.startswith("_"):
            # Skip private modules
            continue

        try:
            # Import the module
            module = importlib.import_module(
                f"claude_evaluator.core.agents.evaluator.reviewers.{module_info.name}"
            )

            # Check if it has a REVIEWER variable
            if hasattr(module, "REVIEWER"):
                reviewer = getattr(module, "REVIEWER")

                # Register the reviewer by its phase name
                if hasattr(reviewer, "phase_name"):
                    phase_name = reviewer.phase_name
                    REVIEWERS[phase_name] = reviewer
                    logger.debug(
                        "Registered phase reviewer",
                        phase=phase_name,
                        module=module_info.name,
                    )
                else:
                    logger.warning(
                        "Reviewer missing phase_name attribute",
                        module=module_info.name,
                    )

        except Exception as e:
            logger.error(
                "Failed to load reviewer module",
                module=module_info.name,
                error=str(e),
            )


# Auto-discover reviewers on import
_discover_reviewers()

__all__ = ["REVIEWERS"]