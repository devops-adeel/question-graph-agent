# Pull Request Guidelines

## Quick Checklist

### Before Creating PR
- [ ] Branch from latest `main`
- [ ] Follow branch naming convention
- [ ] One feature/fix per PR
- [ ] Write descriptive commit messages

### Code Quality
- [ ] Run `black .` for formatting
- [ ] Run `isort .` for imports
- [ ] Run `mypy .` for type checking
- [ ] Run `pytest` for tests
- [ ] Add tests for new code

### Documentation
- [ ] Update docstrings
- [ ] Update README if needed
- [ ] Update CLAUDE.md for phase work
- [ ] Add usage examples

### PR Description
- [ ] Use PR template
- [ ] Link related issues
- [ ] Describe changes clearly
- [ ] Note breaking changes

## Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Maintenance tasks

### Examples
```bash
feat: add Neo4j connection pool support

Implements connection pooling for improved performance
under high load. Includes retry logic and metrics.

Closes #23
```

```bash
fix(extraction): handle empty text input gracefully

Prevents KeyError when processing empty strings
in entity extraction pipeline.
```

## Phase-Specific Guidelines

For phase implementation work:

1. Reference parent issue and sub-task
2. Update CLAUDE.md progress markers
3. Follow phase naming: `feat(phase-X): implement X.X - description`

Example:
```bash
feat(phase-4): implement 4.3 - connection manager with retry logic

Creates robust connection management for Neo4j with:
- Automatic retry on transient failures
- Exponential backoff strategy
- Connection pool support
- Performance metrics

Parent issue: #4
Closes #23
```

## Review Response

When addressing review comments:

1. Respond to each comment
2. Push fixes as new commits (don't force-push during review)
3. Mark conversations as resolved
4. Request re-review when ready

## Merge Requirements

- All CI checks passing
- At least one approval
- No unresolved conversations
- Up to date with main branch