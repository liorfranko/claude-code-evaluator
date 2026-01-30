# Project Constitution

> Foundational principles and constraints governing all development decisions.

**Version:** 1.0.0
**Effective Date:** 2026-01-30
**Last Amended:** 2026-01-30

---

## Core Principles

### I. User-Centric Design

All features must prioritize user experience and accessibility. CLI output should be clear, error messages helpful, and the tool intuitive to use.

### II. Maintainability First

Code should be written for humans to read and maintain. Prefer clarity over cleverness, and document complex logic.

### III. Test-Driven Confidence

New functionality requires accompanying tests. Tests provide confidence for refactoring and ensure reliability.

### IV. Documentation as Code

Documentation is a first-class deliverable. User-facing features require corresponding documentation updates.

### V. Accuracy & Reliability

Evaluation results must be correct and reproducible. The tool's core value depends on trustworthy output.

### VI. Extensibility

The architecture should support easy addition of new evaluation criteria, plugins, or output formats without major refactoring.

---

## Constraints

### Technology Constraints

- **Python Version:** Python 3.10+ required for modern language features
- **Dependencies:** Minimize external dependencies; prefer standard library, add packages only when necessary
- **Compatibility:** Must support macOS and Linux environments

### Compliance Constraints

- **Credentials:** No hardcoded credentials; use environment variables or configuration files for secrets

### Policy Constraints

- **Breaking Changes:** Document any breaking changes in release notes
- **Code Style:** Follow PEP 8 and project linting configuration

---

## Development Workflow

### Required Processes

1. **Specification:** Features must be specified before implementation
2. **Planning:** Implementation plans must be reviewed for constitution compliance
3. **Testing:** Automated tests must pass before merge
4. **Documentation:** User-facing changes require documentation updates
5. **Review:** Self-review changes before committing (solo developer)

### Quality Gates

- [ ] Lint checks pass (ruff/flake8)
- [ ] Type checks pass (mypy)
- [ ] Unit tests pass (pytest)
- [ ] Integration tests pass (if applicable)
- [ ] Documentation updated for user-facing changes
- [ ] Constitution compliance verified

---

## Governance

### Amendment Process

**Solo Developer Mode:** Author can amend the constitution freely with documentation. Changes should be committed with clear commit messages explaining the rationale.

### Override Rules

- Violations require explicit justification in the plan or implementation notes
- Emergency overrides must be documented and addressed in follow-up work
- No override is permanent; violations must be remediated or the constitution amended

### Principle Hierarchy

In case of conflict between principles:

1. Security and compliance constraints take precedence
2. Accuracy & Reliability overrides convenience or performance
3. User-centric design overrides technical preferences
4. Maintainability overrides premature optimization

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0.0 | 2026-01-30 | Initial constitution established | Claude (projspec) |

---

*This constitution is checked during `/projspec:plan` execution. Violations must be justified in the Complexity Tracking section of the plan.*
