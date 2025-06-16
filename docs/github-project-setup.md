# GitHub Project Board Setup Guide

This guide explains how to set up and maintain the GitHub project board for the question-graph-agent repository.

## Quick Start

Run the automated setup script:

```bash
./setup-github-project.sh
```

This script will:
1. Create custom fields (Priority, Type, Phase, Effort)
2. Add all open issues to the project
3. Set field values based on issue labels
4. Guide you through manual steps for UI-only tasks

## Project Structure

### Columns (Status Field)
- **🗄️ Backlog** - Deprioritized items, not immediately actionable
- **📋 Ready** - Prioritized and ready to work on
- **🚧 In Progress** - Currently being worked on (limit: 1 item)
- **🚫 Blocked** - Waiting on external dependencies
- **✅ Done** - Completed items

### Custom Fields

#### Priority
- 🔴 Critical - Fix immediately
- 🟠 High - High priority
- 🟡 Medium - Medium priority
- 🟢 Low - Low priority

#### Type
- 🐛 Bug - Something isn't working
- ✨ Feature - New feature or enhancement
- 📚 Documentation - Documentation improvements
- 🔧 Refactor - Code refactoring
- 🧪 Test - Adding or improving tests

#### Phase
- Phase 6: Temporal
- Phase 7: Validation
- Phase 8: Queries
- Phase 9: Migration
- Phase 10: Testing
- Non-Phase Work

#### Effort
- 🟦 Small (< 1 day)
- 🟨 Medium (1-3 days)
- 🟧 Large (3-5 days)

## Views

### 🎯 Main Board
- **Purpose**: Primary work tracking
- **Layout**: Board grouped by Status
- **Sort**: Priority ↓, Effort ↑

### 🚨 Bug Dashboard
- **Purpose**: Track all bugs
- **Layout**: Board grouped by Priority
- **Filter**: Type is "🐛 Bug"

### 📊 Phase Progress
- **Purpose**: Track Graphiti integration phases
- **Layout**: Board grouped by Phase
- **Filter**: Phase is not "Non-Phase Work"

### 📅 Sprint Planning
- **Purpose**: Plan upcoming work
- **Layout**: Table view
- **Columns**: Title, Priority, Type, Phase, Effort, Status

## Label-to-Field Mapping

The setup script automatically maps GitHub issue labels to project fields:

### Priority Mapping
- `priority: critical` → 🔴 Critical
- `priority: high` → 🟠 High
- `priority: medium` → 🟡 Medium
- `priority: low` → 🟢 Low
- Default → 🟡 Medium

### Type Mapping
- `bug` or `type: bug` → 🐛 Bug
- `enhancement` or `type: feature` → ✨ Feature
- `documentation` or `type: docs` → 📚 Documentation
- `type: refactor` → 🔧 Refactor
- `type: test` → 🧪 Test
- Default → ✨ Feature

### Phase Mapping
- `phase: 6-temporal` → Phase 6: Temporal
- `phase: 7-validation` → Phase 7: Validation
- `phase: 8-queries` → Phase 8: Queries
- `phase: 9-migration` → Phase 9: Migration
- `phase: 10-testing` → Phase 10: Testing
- Default → Non-Phase Work

### Effort Mapping
- `effort: small` → 🟦 Small (< 1 day)
- `effort: medium` → 🟨 Medium (1-3 days)
- `effort: large` → 🟧 Large (3-5 days)
- Default → 🟨 Medium (1-3 days)

### Status Mapping
- High/Critical priority bugs → Ready
- Phase 6+ issues → Backlog
- All others → Backlog

## Maintenance

### Update Field Values
Re-run the field update script after adding new issues or changing labels:

```bash
./update-project-fields.sh
```

### Add New Issues
The script automatically adds all open issues. To add a specific issue:

```bash
gh project item-add 3 --owner @me --url "https://github.com/devops-adeel/question-graph-agent/issues/[NUMBER]"
```

### Manual Updates
For one-off field updates, use the GitHub UI:
1. Go to the project board
2. Click on an item
3. Update fields in the side panel

## Workflow Guidelines

### Work in Progress Limit
- Only 1 item in "In Progress" at a time
- Complete or move to "Blocked" before starting new work

### Priority Order
1. 🔴 Critical bugs
2. 🟠 High priority items
3. 🟡 Medium priority items
4. 🟢 Low priority items

### Phase Progression
- Complete all bugs before starting new phase work
- Work through phases sequentially (6 → 7 → 8 → 9 → 10)
- Keep future phase work in Backlog

### Definition of Done
- Code complete and tested
- PR reviewed and merged
- Documentation updated if needed
- Item moved to "Done" column

## Automation Recommendations

Consider setting up these GitHub Actions workflows:

### Auto-add Issues
Add new issues to project when labeled:
```yaml
on:
  issues:
    types: [labeled]
```

### Auto-archive Closed
Archive items when issues are closed:
```yaml
on:
  issues:
    types: [closed]
```

### Status Sync
Update status field based on issue state:
- Open → Ready/Backlog
- Closed → Done

## Troubleshooting

### Fields Not Showing
- Ensure you're viewing the correct project
- Check field visibility in view settings

### Items Missing
- Run setup script to add all issues
- Check filters in current view

### Wrong Field Values
- Re-run `./update-project-fields.sh`
- Check label accuracy on issues

## Related Files
- `setup-github-project.sh` - Main setup script
- `update-project-fields.sh` - Field update helper
- `project-fields.json` - Field configuration cache
- `open-issues.json` - Issue snapshot from setup