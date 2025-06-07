#!/bin/bash
# Script to set up GitHub Project board for Graphiti Integration Implementation
# Run this after ensuring proper authentication: gh auth refresh -s project

set -e

echo "🚀 Setting up GitHub Project board for Graphiti Integration Implementation"

# Check if gh is installed
if ! command -v gh &> /dev/null; then
    echo "❌ Error: GitHub CLI (gh) is not installed"
    echo "Install it from: https://cli.github.com/"
    exit 1
fi

# Check authentication
echo "🔐 Checking GitHub authentication..."
if ! gh auth status &> /dev/null; then
    echo "❌ Error: Not authenticated with GitHub"
    echo "Run: gh auth login"
    exit 1
fi

# Variables
OWNER="devops-adeel"
REPO="question-graph-agent"

# Create project
echo "📋 Creating project..."
PROJECT_OUTPUT=$(gh project create --owner "$OWNER" --title "Graphiti Integration Implementation" --format json 2>/dev/null || echo "EXISTS")

if [ "$PROJECT_OUTPUT" = "EXISTS" ]; then
    echo "ℹ️  Project might already exist, fetching project list..."
    PROJECT_NUMBER=$(gh project list --owner "$OWNER" --format json | jq -r '.projects[] | select(.title == "Graphiti Integration Implementation") | .number' | head -1)
else
    PROJECT_NUMBER=$(echo "$PROJECT_OUTPUT" | jq -r '.number')
fi

if [ -z "$PROJECT_NUMBER" ]; then
    echo "❌ Error: Could not create or find project"
    exit 1
fi

echo "✅ Using project number: $PROJECT_NUMBER"

# Create custom fields
echo "🏷️  Creating custom fields..."

# Create Phase field
gh project field-create "$PROJECT_NUMBER" --owner "$OWNER" --name "Phase" --data-type "SINGLE_SELECT" 2>/dev/null || echo "Phase field might already exist"

# Add phase options (this needs to be done through the web UI as the CLI doesn't support it yet)
echo "ℹ️  Note: You'll need to add Phase options manually through the web UI:
   - Phase 1 - Core Entity Models
   - Phase 2 - Relationship Models
   - Phase 3 - Entity Extraction
   - Phase 4 - Graphiti Infrastructure"

# Create Priority field
gh project field-create "$PROJECT_NUMBER" --owner "$OWNER" --name "Priority" --data-type "SINGLE_SELECT" 2>/dev/null || echo "Priority field might already exist"

# Create Completed On field
gh project field-create "$PROJECT_NUMBER" --owner "$OWNER" --name "Completed On" --data-type "DATE" 2>/dev/null || echo "Completed On field might already exist"

# Add all issues to the project
echo "📌 Adding issues to project..."
for i in {1..27}; do
    echo "Adding issue #$i..."
    gh project item-add "$PROJECT_NUMBER" --owner "$OWNER" --url "https://github.com/$OWNER/$REPO/issues/$i" 2>/dev/null || echo "Issue #$i might already be in project"
done

echo "
✅ Project board setup complete!

📊 Summary:
- Project Number: $PROJECT_NUMBER
- Total Issues Added: 27
- Completed Issues: 20 (#2-8, #9-14, #15-20, #21)
- Open Issues: 6 (#22-27)

🔗 View your project board:
   gh project view $PROJECT_NUMBER --owner $OWNER --web

📝 Next steps:
1. Open the project board in your browser
2. Configure the Phase field options
3. Assign Phase values to each issue
4. Set up board columns as needed
5. Create additional views (Timeline, Priority, etc.)

💡 Tips:
- Group by 'Phase' to see issues organized by implementation phase
- Use filters to focus on open issues
- Enable automation for moving cards when issues are closed
"

# Open in browser
read -p "Would you like to open the project board in your browser? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    gh project view "$PROJECT_NUMBER" --owner "$OWNER" --web
fi