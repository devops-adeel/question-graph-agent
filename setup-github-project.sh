#!/bin/bash
# Comprehensive GitHub Project Setup Script
# This single script handles all aspects of project board configuration

set -e  # Exit on any error

# ============================================================================
# Configuration
# ============================================================================

# Colors for output
readonly GREEN='\033[0;32m'
readonly BLUE='\033[0;34m'
readonly YELLOW='\033[1;33m'
readonly RED='\033[0;31m'
readonly NC='\033[0m' # No Color

# Project settings
readonly PROJECT_NUMBER=3
readonly OWNER="@me"
readonly REPO="devops-adeel/question-graph-agent"

# Cache directory for API responses
readonly CACHE_DIR=".github-project-cache"
mkdir -p "$CACHE_DIR"

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to make GraphQL queries with error handling
graphql_query() {
    local query=$1
    local output_file=$2
    local description=${3:-"GraphQL query"}
    
    log_info "Executing: $description"
    
    if ! gh api graphql -f query="$query" > "$output_file" 2>&1; then
        log_error "Failed to execute: $description"
        cat "$output_file"
        return 1
    fi
    
    # Check for GraphQL errors in response
    if jq -e '.errors' "$output_file" >/dev/null 2>&1; then
        log_error "GraphQL error in: $description"
        jq '.errors' "$output_file"
        return 1
    fi
    
    return 0
}

# Function to update field value with proper error handling
update_field_value() {
    local item_id=$1
    local field_id=$2
    local option_id=$3
    local project_id=$4
    local field_name=$5
    local issue_num=$6
    
    local mutation="
    mutation {
        updateProjectV2ItemFieldValue(input: {
            projectId: \"$project_id\"
            itemId: \"$item_id\"
            fieldId: \"$field_id\"
            value: {singleSelectOptionId: \"$option_id\"}
        }) {
            projectV2Item {
                id
            }
        }
    }"
    
    if graphql_query "$mutation" "$CACHE_DIR/update_result.json" "Update $field_name for #$issue_num" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# ============================================================================
# Step 1: Verify Project Exists
# ============================================================================

log_info "Verifying project #$PROJECT_NUMBER exists..."

if ! graphql_query "{ user(login: \"devops-adeel\") { projectV2(number: $PROJECT_NUMBER) { id name } } }" \
    "$CACHE_DIR/project_info.json" "Fetch project info"; then
    log_error "Project #$PROJECT_NUMBER not found!"
    exit 1
fi

PROJECT_ID=$(jq -r '.data.user.projectV2.id' "$CACHE_DIR/project_info.json")
PROJECT_NAME=$(jq -r '.data.user.projectV2.name' "$CACHE_DIR/project_info.json")

log_success "Found project: $PROJECT_NAME (ID: $PROJECT_ID)"

# ============================================================================
# Step 2: Create/Verify Custom Fields
# ============================================================================

log_info "Setting up custom fields..."

# Get existing fields
graphql_query "{ user(login: \"devops-adeel\") { projectV2(number: $PROJECT_NUMBER) { fields(first: 100) { nodes { name dataType } } } } }" \
    "$CACHE_DIR/existing_fields.json" "Fetch existing fields"

EXISTING_FIELDS=$(jq -r '.data.user.projectV2.fields.nodes[].name' "$CACHE_DIR/existing_fields.json")

# Define fields to create
declare -A FIELDS_TO_CREATE=(
    ["Priority"]="🔴 Critical,🟠 High,🟡 Medium,🟢 Low"
    ["Type"]="🐛 Bug,✨ Feature,📚 Documentation,🔧 Refactor,🧪 Test"
    ["Phase"]="Phase 6: Temporal,Phase 7: Validation,Phase 8: Queries,Phase 9: Migration,Phase 10: Testing,Non-Phase Work"
    ["Effort"]="🟦 Small (< 1 day),🟨 Medium (1-3 days),🟧 Large (3-5 days)"
)

# Create missing fields
for field_name in "${!FIELDS_TO_CREATE[@]}"; do
    if echo "$EXISTING_FIELDS" | grep -q "^$field_name$"; then
        log_info "Field '$field_name' already exists"
    else
        log_info "Creating field '$field_name'..."
        gh project field-create $PROJECT_NUMBER \
            --owner "$OWNER" \
            --name "$field_name" \
            --data-type "SINGLE_SELECT" \
            --single-select-options "${FIELDS_TO_CREATE[$field_name]}"
        log_success "Created field '$field_name'"
    fi
done

# ============================================================================
# Step 3: Cache All Field Information
# ============================================================================

log_info "Caching field configurations..."

graphql_query "{ user(login: \"devops-adeel\") { projectV2(number: $PROJECT_NUMBER) { 
    fields(first: 100) { 
        nodes { 
            ... on ProjectV2SingleSelectField { 
                id name options { id name } 
            } 
            ... on ProjectV2Field { 
                id name 
            } 
        } 
    } 
} } }" "$CACHE_DIR/all_fields.json" "Fetch all field details"

# Extract field IDs
STATUS_FIELD_ID=$(jq -r '.data.user.projectV2.fields.nodes[] | select(.name == "Status") | .id' "$CACHE_DIR/all_fields.json")
PRIORITY_FIELD_ID=$(jq -r '.data.user.projectV2.fields.nodes[] | select(.name == "Priority") | .id' "$CACHE_DIR/all_fields.json")
TYPE_FIELD_ID=$(jq -r '.data.user.projectV2.fields.nodes[] | select(.name == "Type") | .id' "$CACHE_DIR/all_fields.json")
PHASE_FIELD_ID=$(jq -r '.data.user.projectV2.fields.nodes[] | select(.name == "Phase") | .id' "$CACHE_DIR/all_fields.json")
EFFORT_FIELD_ID=$(jq -r '.data.user.projectV2.fields.nodes[] | select(.name == "Effort") | .id' "$CACHE_DIR/all_fields.json")

# ============================================================================
# Step 4: Add All Open Issues to Project
# ============================================================================

log_info "Adding open issues to project..."

# Get all open issues
gh issue list --repo "$REPO" --state open --limit 200 --json number,title,labels > "$CACHE_DIR/open_issues.json"

TOTAL_ISSUES=$(jq '. | length' "$CACHE_DIR/open_issues.json")
log_info "Found $TOTAL_ISSUES open issues"

# Check which issues are already in the project
graphql_query "{ user(login: \"devops-adeel\") { projectV2(number: $PROJECT_NUMBER) { 
    items(first: 100) { 
        nodes { 
            id 
            content { 
                ... on Issue { number } 
            } 
        } 
    } 
} } }" "$CACHE_DIR/project_items.json" "Fetch existing project items"

# Add missing issues
ADDED_COUNT=0
while IFS= read -r issue_num; do
    # Check if already in project
    if jq -e --arg num "$issue_num" '.data.user.projectV2.items.nodes[] | select(.content.number == ($num | tonumber))' \
        "$CACHE_DIR/project_items.json" >/dev/null 2>&1; then
        continue
    fi
    
    # Add to project
    echo -n "Adding issue #$issue_num... "
    if gh project item-add $PROJECT_NUMBER --owner "$OWNER" \
        --url "https://github.com/$REPO/issues/$issue_num" >/dev/null 2>&1; then
        echo "✓"
        ((ADDED_COUNT++))
    else
        echo "✗"
    fi
done < <(jq -r '.[].number' "$CACHE_DIR/open_issues.json")

log_success "Added $ADDED_COUNT new issues to project"

# ============================================================================
# Step 5: Update All Field Values Based on Labels
# ============================================================================

log_info "Setting field values based on issue labels..."

# Refresh project items list after adding new ones
graphql_query "{ user(login: \"devops-adeel\") { projectV2(number: $PROJECT_NUMBER) { 
    items(first: 100) { 
        nodes { 
            id 
            content { 
                ... on Issue { 
                    number 
                    title
                    labels(first: 20) { 
                        nodes { name } 
                    } 
                } 
            } 
        } 
    } 
} } }" "$CACHE_DIR/all_project_items.json" "Fetch all project items with labels"

# Define mapping functions
get_status_value() {
    local labels=$1
    local issue_num=$2
    
    # High priority bugs go to Ready
    if [[ "$labels" == *"bug"* ]] && [[ "$labels" == *"priority: critical"* || "$labels" == *"priority: high"* ]]; then
        echo "Ready"
    # Specific bugs we identified
    elif [[ "$issue_num" == "85" || "$issue_num" == "91" || "$issue_num" == "89" || "$issue_num" == "93" ]]; then
        echo "Ready"
    else
        echo "Backlog"
    fi
}

get_priority_value() {
    local labels=$1
    case "$labels" in
        *"priority: critical"*) echo "🔴 Critical" ;;
        *"priority: high"*) echo "🟠 High" ;;
        *"priority: medium"*) echo "🟡 Medium" ;;
        *"priority: low"*) echo "🟢 Low" ;;
        *"bug"*) echo "🟠 High" ;;  # Default bugs to high
        *) echo "🟡 Medium" ;;      # Default others to medium
    esac
}

get_type_value() {
    local labels=$1
    case "$labels" in
        *"bug"*|*"type: bug"*) echo "🐛 Bug" ;;
        *"documentation"*|*"type: docs"*) echo "📚 Documentation" ;;
        *"type: refactor"*) echo "🔧 Refactor" ;;
        *"type: test"*) echo "🧪 Test" ;;
        *) echo "✨ Feature" ;;  # Default to feature
    esac
}

get_phase_value() {
    local labels=$1
    case "$labels" in
        *"phase: 6-temporal"*) echo "Phase 6: Temporal" ;;
        *"phase: 7-validation"*) echo "Phase 7: Validation" ;;
        *"phase: 8-queries"*) echo "Phase 8: Queries" ;;
        *"phase: 9-migration"*) echo "Phase 9: Migration" ;;
        *"phase: 10-testing"*) echo "Phase 10: Testing" ;;
        *) echo "Non-Phase Work" ;;
    esac
}

get_effort_value() {
    local labels=$1
    local issue_num=$2
    
    # Specific bug efforts
    if [[ "$issue_num" == "85" || "$issue_num" == "91" ]]; then
        echo "🟦 Small (< 1 day)"
    elif [[ "$issue_num" == "89" ]]; then
        echo "🟨 Medium (1-3 days)"
    elif [[ "$issue_num" == "93" ]]; then
        echo "🟧 Large (3-5 days)"
    else
        # Label-based
        case "$labels" in
            *"effort: small"*) echo "🟦 Small (< 1 day)" ;;
            *"effort: large"*) echo "🟧 Large (3-5 days)" ;;
            *) echo "🟨 Medium (1-3 days)" ;;  # Default to medium
        esac
    fi
}

# Function to get option ID for a field value
get_option_id() {
    local field_name=$1
    local option_name=$2
    jq -r --arg field "$field_name" --arg opt "$option_name" \
        '.data.user.projectV2.fields.nodes[] | select(.name == $field) | .options[] | select(.name == $opt) | .id' \
        "$CACHE_DIR/all_fields.json"
}

# Process each item
UPDATE_COUNT=0
TOTAL_ITEMS=$(jq '.data.user.projectV2.items.nodes | length' "$CACHE_DIR/all_project_items.json")

jq -c '.data.user.projectV2.items.nodes[]' "$CACHE_DIR/all_project_items.json" | while read -r item; do
    ITEM_ID=$(echo "$item" | jq -r '.id')
    ISSUE_NUM=$(echo "$item" | jq -r '.content.number // empty')
    
    if [ -z "$ISSUE_NUM" ]; then
        continue
    fi
    
    # Get all labels as a single string
    LABELS=$(echo "$item" | jq -r '.content.labels.nodes[].name' | tr '\n' ' ')
    
    echo -n "Processing issue #$ISSUE_NUM... "
    
    # Update Status
    STATUS_VALUE=$(get_status_value "$LABELS" "$ISSUE_NUM")
    STATUS_OPTION_ID=$(get_option_id "Status" "$STATUS_VALUE")
    if [ -n "$STATUS_OPTION_ID" ]; then
        update_field_value "$ITEM_ID" "$STATUS_FIELD_ID" "$STATUS_OPTION_ID" "$PROJECT_ID" "Status" "$ISSUE_NUM"
    fi
    
    # Update Priority
    PRIORITY_VALUE=$(get_priority_value "$LABELS")
    PRIORITY_OPTION_ID=$(get_option_id "Priority" "$PRIORITY_VALUE")
    if [ -n "$PRIORITY_OPTION_ID" ]; then
        update_field_value "$ITEM_ID" "$PRIORITY_FIELD_ID" "$PRIORITY_OPTION_ID" "$PROJECT_ID" "Priority" "$ISSUE_NUM"
    fi
    
    # Update Type
    TYPE_VALUE=$(get_type_value "$LABELS")
    TYPE_OPTION_ID=$(get_option_id "Type" "$TYPE_VALUE")
    if [ -n "$TYPE_OPTION_ID" ]; then
        update_field_value "$ITEM_ID" "$TYPE_FIELD_ID" "$TYPE_OPTION_ID" "$PROJECT_ID" "Type" "$ISSUE_NUM"
    fi
    
    # Update Phase
    PHASE_VALUE=$(get_phase_value "$LABELS")
    PHASE_OPTION_ID=$(get_option_id "Phase" "$PHASE_VALUE")
    if [ -n "$PHASE_OPTION_ID" ]; then
        update_field_value "$ITEM_ID" "$PHASE_FIELD_ID" "$PHASE_OPTION_ID" "$PROJECT_ID" "Phase" "$ISSUE_NUM"
    fi
    
    # Update Effort
    EFFORT_VALUE=$(get_effort_value "$LABELS" "$ISSUE_NUM")
    EFFORT_OPTION_ID=$(get_option_id "Effort" "$EFFORT_VALUE")
    if [ -n "$EFFORT_OPTION_ID" ]; then
        update_field_value "$ITEM_ID" "$EFFORT_FIELD_ID" "$EFFORT_OPTION_ID" "$PROJECT_ID" "Effort" "$ISSUE_NUM"
    fi
    
    echo "✓"
    ((UPDATE_COUNT++))
done

log_success "Updated $UPDATE_COUNT items"

# ============================================================================
# Step 6: Verify Setup
# ============================================================================

log_info "Verifying project setup..."

# Count items by status
graphql_query "{ user(login: \"devops-adeel\") { projectV2(number: $PROJECT_NUMBER) { 
    items(first: 100) { 
        nodes { 
            fieldValueByName(name: \"Status\") { 
                ... on ProjectV2ItemFieldSingleSelectValue { name } 
            } 
        } 
    } 
} } }" "$CACHE_DIR/verify_status.json" "Verify status values"

echo -e "\nStatus Distribution:"
jq -r '.data.user.projectV2.items.nodes[].fieldValueByName.name // "No Status"' "$CACHE_DIR/verify_status.json" | \
    sort | uniq -c | while read count status; do
    echo "  $status: $count items"
done

# Count items by type
graphql_query "{ user(login: \"devops-adeel\") { projectV2(number: $PROJECT_NUMBER) { 
    items(first: 100) { 
        nodes { 
            fieldValueByName(name: \"Type\") { 
                ... on ProjectV2ItemFieldSingleSelectValue { name } 
            } 
        } 
    } 
} } }" "$CACHE_DIR/verify_type.json" "Verify type values"

echo -e "\nType Distribution:"
jq -r '.data.user.projectV2.items.nodes[].fieldValueByName.name // "No Type"' "$CACHE_DIR/verify_type.json" | \
    sort | uniq -c | while read count type; do
    echo "  $type: $count items"
done

# ============================================================================
# Step 7: Provide Instructions for Manual Steps
# ============================================================================

echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ GitHub Project Setup Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo -e "\n${BLUE}📋 Manual Steps Required:${NC}"
echo -e "\n1. ${YELLOW}Create Project Views${NC}"
echo "   Go to: https://github.com/users/devops-adeel/projects/$PROJECT_NUMBER"
echo ""
echo "   a) 🎯 Main Board (Board layout)"
echo "      - Group by: Status"
echo "      - Sort: Priority ↓, Effort ↑"
echo ""
echo "   b) 🚨 Bug Dashboard (Board layout)"
echo "      - Filter: Type is '🐛 Bug'"
echo "      - Group by: Priority"
echo ""
echo "   c) 📊 Phase Progress (Board layout)"
echo "      - Filter: Phase is not 'Non-Phase Work'"
echo "      - Group by: Phase"
echo ""
echo "   d) 📅 Sprint Planning (Table layout)"
echo "      - Show: Title, Priority, Type, Phase, Effort, Status"
echo "      - Sort: Priority ↓, Effort ↑"

echo -e "\n2. ${YELLOW}Optional: Set up Automation${NC}"
echo "   - Auto-add new issues when labeled"
echo "   - Auto-archive when issues close"

echo -e "\n${BLUE}🔗 Quick Links:${NC}"
echo "   Project Board: https://github.com/users/devops-adeel/projects/$PROJECT_NUMBER"
echo "   Bug Dashboard: https://github.com/users/devops-adeel/projects/$PROJECT_NUMBER/views/3"
echo "   Phase Progress: https://github.com/users/devops-adeel/projects/$PROJECT_NUMBER/views/4"

echo -e "\n${BLUE}📝 Files Created:${NC}"
echo "   - $CACHE_DIR/ (temporary cache, can be deleted)"
echo "   - Run this script again anytime to update field values"

# ============================================================================
# Cleanup
# ============================================================================

# Optionally remove cache directory
# rm -rf "$CACHE_DIR"

echo -e "\n${GREEN}✨ Setup complete! Your project board is ready to use.${NC}"