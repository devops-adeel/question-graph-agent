## Summary

<!-- Provide a brief summary of your changes in 2-3 bullet points -->

- 
- 
- 

## Related Issue(s)

<!-- Link to related issues. Use "Closes #XX" for issues that will be closed by this PR -->

Closes #

<!-- For phase sub-tasks, also reference the parent phase issue -->
<!-- Parent Phase: #XX -->

## Type of Change

<!-- Mark the relevant option with an "x" -->

- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📝 Documentation update
- [ ] ♻️ Refactoring (no functional changes)
- [ ] 🧪 Test improvement
- [ ] 🏗️ Infrastructure/configuration change

## Implementation Details

### Changes Made

<!-- Describe your changes in detail. Include technical approach if relevant -->

### Files Modified

<!-- List key files created or modified -->

- **Created**: 
- **Modified**: 
- **Deleted**: 

### Phase Progress Update

<!-- If this PR completes a phase sub-task, update the status -->
<!-- - [ ] Updated CLAUDE.md progress markers -->
<!-- - [ ] Phase X.X marked as complete -->

## Testing

### Test Coverage

- [ ] Unit tests written/updated
- [ ] Integration tests written/updated
- [ ] End-to-end tests written/updated (if applicable)
- [ ] All tests passing locally

### Test Details

<!-- Describe your testing approach -->

```bash
# Commands used to test
pytest tests/test_module.py -v
```

### Coverage Report

<!-- Include coverage percentage for modified files -->

```
File                        Stmts   Miss  Cover
-----------------------------------------------
module_name.py               100      5    95%
```

## Documentation

- [ ] Code includes appropriate docstrings
- [ ] README.md updated (if needed)
- [ ] CLAUDE.md updated (if implementation details changed)
- [ ] API documentation updated (if public API changed)
- [ ] Example usage provided (for new features)

## Breaking Changes

<!-- If this PR includes breaking changes, describe them here -->

### Migration Guide

<!-- If breaking changes, provide migration instructions -->

```python
# Old way
old_function()

# New way
new_function(param)
```

## Performance Impact

<!-- Describe any performance implications -->

- [ ] No significant performance impact
- [ ] Performance improved
- [ ] Performance degraded (justified by: )

## Security Considerations

<!-- Mark any that apply -->

- [ ] No security implications
- [ ] Input validation added/improved
- [ ] Authentication/authorization changes
- [ ] Sensitive data handling improved

## Pre-submission Checklist

### Code Quality

- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My changes generate no new warnings

### Formatting & Linting

- [ ] Code formatted with `black .`
- [ ] Imports sorted with `isort .`
- [ ] Type hints added/updated
- [ ] `mypy` passes without errors
- [ ] `ruff check .` passes

### Dependencies

- [ ] No new dependencies added
- [ ] New dependencies added to `pyproject.toml` and `requirements.txt`
- [ ] Dependencies are justified and documented

### Integration

- [ ] Changes work with Neo4j connection
- [ ] Graphiti integration tested (if applicable)
- [ ] Configuration changes documented in `.env.example`

## Additional Notes

<!-- Any additional information that reviewers should know -->

## Reviewer Guidelines

<!-- For reviewers: key areas to focus on -->

- [ ] Verify test coverage is adequate
- [ ] Check for proper error handling
- [ ] Validate async/sync implementations
- [ ] Ensure documentation is clear
- [ ] Confirm phase progress is accurate