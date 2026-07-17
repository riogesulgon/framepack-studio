#!/usr/bin/env bash
#
# Git History Sensitive Information Audit
# Scans the full git history for leaked secrets, credentials, and PII.
#
# Usage:
#   ./check_git_sensitive_info.sh [REPO_PATH]
#
# If REPO_PATH is not provided, defaults to the parent directory of this script's repo.
#
# Exit codes:
#   0 - All checks passed (no issues found)
#   1 - Potential issues found (review output)
#
# Based on the manual audit performed on 2025-07-17 for FP-Studio/framepack-studio.

set -uo pipefail

# --- Configuration ---
REPO_PATH="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
ISSUES_FOUND=0
VERBOSE="${VERBOSE:-0}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================"
echo " Git Sensitive Information Audit"
echo " Repository: ${REPO_PATH}"
echo " Date:       $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo "============================================"
echo ""

cd "$REPO_PATH"

# Verify we're in a git repo
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo -e "${RED}ERROR: '${REPO_PATH}' is not a git repository${NC}"
    exit 1
fi

# Helper: report a finding
report_issue() {
    local check_name="$1"
    local detail="$2"
    echo -e "${RED}[FAIL] ${check_name}${NC}"
    echo -e "       ${detail}"
    ISSUES_FOUND=1
}

report_pass() {
    local check_name="$1"
    echo -e "${GREEN}[PASS]${NC} ${check_name}"
}

report_warn() {
    local check_name="$1"
    local detail="$2"
    echo -e "${YELLOW}[WARN]${NC} ${check_name}"
    echo -e "       ${detail}"
}

# Helper: search git history for a pattern in diffs (added lines only)
# Usage: git_search_diff "pattern" "description" [file_globs...]
git_search_diff() {
    local pattern="$1"
    local description="$2"
    shift 2
    local files=("$@")

    local result
    if [ ${#files[@]} -gt 0 ]; then
        result=$(git log --all -p -S "$pattern" -- "${files[@]}" 2>/dev/null \
            | grep -E '^\+' \
            | grep -v '^+++' \
            | head -5 || true)
    else
        result=$(git log --all -p -S "$pattern" 2>/dev/null \
            | grep -E '^\+' \
            | grep -v '^+++' \
            | head -5 || true)
    fi

    if [ -n "$result" ]; then
        report_issue "$description" "Pattern '$pattern' found in git history"
        echo "$result"
    else
        report_pass "$description"
    fi
}

# ==============================================
# 1. Hardcoded API Keys & Tokens
# ==============================================
echo "--- 1. Hardcoded API Keys & Tokens ---"

# OpenAI / generic sk- keys (look for sk- followed by typical key format)
RESULT=$(git log --all -p -S "sk-" -- '*.py' '*.json' '*.yaml' '*.yml' '*.toml' '*.cfg' '*.ini' '*.env' 2>/dev/null \
    | grep -E '^\+' \
    | grep -v '^+++' \
    | grep -oE 'sk-[A-Za-z0-9]{20,}' \
    | head -5 || true)
if [ -n "$RESULT" ]; then
    report_issue "OpenAI/generic sk- API keys found" "Actual key values detected in git history"
    echo "$RESULT"
else
    report_pass "No OpenAI/generic sk- API keys found"
fi

# GitHub personal access tokens (ghp_, gho_, github_pat_)
RESULT=""
for pat in "ghp_" "gho_" "github_pat_"; do
    RESULT=$(git log --all -p -S "$pat" 2>/dev/null \
        | grep -E '^\+' \
        | grep -v '^+++' \
        | grep -oE "${pat}[A-Za-z0-9_]{20,}" \
        | head -5 || true)
    if [ -n "$RESULT" ]; then break; fi
done
if [ -n "$RESULT" ]; then
    report_issue "GitHub PATs found" "Actual token values detected in git history"
    echo "$RESULT"
else
    report_pass "No GitHub PATs found"
fi

# HuggingFace tokens (hardcoded literal values, not env reads)
RESULT=$(git log --all -p -S "hf_" -- '*.py' '*.json' '*.yaml' '*.yml' 2>/dev/null \
    | grep -E '^\+' \
    | grep -v '^+++' \
    | grep -v 'os\.environ\|os\.getenv\|get_token\|hf_hub_\|hf_download\|HF_HOME\|HF_HUB\|hf_login\|hf_token\s*=\s*os\.\|hf_mesh\|hf_clip\|hf_' \
    | grep -oE 'hf_[A-Za-z0-9]{20,}' \
    | head -5 || true)
if [ -n "$RESULT" ]; then
    report_issue "Hardcoded HF tokens found" "Actual token values detected in git history"
    echo "$RESULT"
else
    report_pass "No hardcoded HuggingFace tokens found"
fi

# AWS access key IDs (start with AKIA, 20 chars uppercase+digits)
RESULT=$(git log --all -p -S "AKIA" 2>/dev/null \
    | grep -E '^\+' \
    | grep -v '^+++' \
    | grep -oE 'AKIA[A-Z0-9]{16}' \
    | head -5 || true)
if [ -n "$RESULT" ]; then
    report_issue "AWS access key IDs found" "Actual key values detected in git history"
    echo "$RESULT"
else
    report_pass "No AWS access key IDs found"
fi

# AWS secret keys
RESULT=""
for kw in "aws_secret_access_key" "AwsSecretAccessKey"; do
    RESULT=$(git log --all -p -S "$kw" -- '*.py' '*.json' '*.yaml' '*.yml' 2>/dev/null \
        | grep -E '^\+' | grep -v '^+++' | head -5 || true)
    if [ -n "$RESULT" ]; then break; fi
done
if [ -n "$RESULT" ]; then
    report_issue "AWS secret keys found" "Matches in git history"
    echo "$RESULT"
else
    report_pass "No AWS secret keys found"
fi

# Generic api_key / secret_key patterns (actual assignments, not just key names)
RESULT=$(git log --all -p -- '*.py' '*.json' '*.yaml' '*.yml' '*.toml' '*.cfg' '*.ini' '*.env' 2>/dev/null \
    | grep -E '^\+' \
    | grep -v '^+++' \
    | grep -iE "(api_key|secret_key|private_key)\s*=\s*['\"][A-Za-z0-9_\-]{10,}['\"]" \
    | head -5 || true)
if [ -n "$RESULT" ]; then
    report_issue "Hardcoded api_key/secret_key assignments found" "Actual secret values in code"
    echo "$RESULT"
else
    report_pass "No hardcoded api_key/secret_key assignments found"
fi

# Password patterns (assignments with string literals, not just YAML key names)
RESULT=$(git log --all -p -- '*.py' '*.json' '*.env' 2>/dev/null \
    | grep -E '^\+' \
    | grep -v '^+++' \
    | grep -iE "password\s*=\s*['\"][^'\"]{4,}['\"]" \
    | head -5 || true)
if [ -n "$RESULT" ]; then
    report_issue "Hardcoded password assignments found" "Actual password values in code"
    echo "$RESULT"
else
    report_pass "No hardcoded password assignments found"
fi

echo ""

# ==============================================
# 2. Private Keys & Certificates
# ==============================================
echo "--- 2. Private Keys & Certificates ---"

RESULT=""
for marker in "-----BEGIN RSA" "-----BEGIN PRIVATE" "-----BEGIN CERTIFICATE" "-----BEGIN PGP"; do
    RESULT=$(git log --all -p -S "$marker" 2>/dev/null | head -20 || true)
    if [ -n "$RESULT" ]; then break; fi
done
if [ -n "$RESULT" ]; then
    report_issue "Private keys/certificates found" "BEGIN markers in git history"
    echo "$RESULT"
else
    report_pass "No private keys or certificates found"
fi

RESULT=$(git log --all --name-only --diff-filter=A 2>/dev/null | sort -u | grep -iE '\.(pem|key|p12|pfx|secret|cred|token)$')
if [ -n "$RESULT" ]; then
    report_issue "Sensitive file types committed" "Files: ${RESULT}"
    echo "$RESULT"
else
    report_pass "No sensitive file types (.pem, .key, .p12, etc.) committed"
fi

echo ""

# ==============================================
# 3. Environment / Config Files
# ==============================================
echo "--- 3. Environment & Config Files ---"

RESULT=$(git log --all -p -- '*.env' '.env.*' 2>/dev/null | head -20 || true)
if [ -n "$RESULT" ]; then
    report_issue ".env files found in git history" "Review for leaked secrets"
    echo "$RESULT"
else
    report_pass "No .env files found in git history"
fi

# Check if .gitignore covers .env
if [ -f ".gitignore" ]; then
    if grep -qE '^\.env' .gitignore 2>/dev/null; then
        report_pass ".gitignore includes .env"
    else
        report_warn ".gitignore does not include .env" "Consider adding .env to .gitignore"
    fi
else
    report_warn "No .gitignore file found" "Consider creating one to exclude sensitive files"
fi

# Check settings.json and queue.json are gitignored
for SENSITIVE_FILE in settings.json queue.json; do
    if [ -f ".gitignore" ]; then
        if grep -qE "^${SENSITIVE_FILE}" .gitignore 2>/dev/null; then
            report_pass ".gitignore excludes ${SENSITIVE_FILE}"
        else
            report_warn ".gitignore does not exclude ${SENSITIVE_FILE}" "Consider adding ${SENSITIVE_FILE} to .gitignore"
        fi
    fi
done

echo ""

# ==============================================
# 4. CI/CD Secrets Leakage
# ==============================================
echo "--- 4. CI/CD Secrets Leakage ---"

CI_FILES=$(git ls-files -- '.github/workflows/*.yml' '.github/workflows/*.yaml' '.gitlab-ci.yml' 'Jenkinsfile' 'Dockerfile' 'docker-compose.yml' 'docker-compose.yaml' 2>/dev/null)
if [ -n "$CI_FILES" ]; then
    # Check for hardcoded secrets in CI files (not using GitHub Actions secrets)
    RESULT=$(for f in $CI_FILES; do git show "HEAD:$f" 2>/dev/null; done \
        | grep -iE '(password|secret|token|key)\s*:\s*[^$\s{{][A-Za-z0-9]{8,}' \
        | grep -v 'secrets\.\|\${{' | head -10 || true)
    if [ -n "$RESULT" ]; then
        report_issue "Hardcoded secrets in CI/CD files" "Use secret references instead"
        echo "$RESULT"
    else
        report_pass "CI/CD files use secret references (no hardcoded secrets)"
    fi
else
    report_pass "No CI/CD configuration files found"
fi

# Check for DOCKERHUB credentials hardcoded
RESULT=$(git log --all -p -S "DOCKERHUB" 2>/dev/null \
    | grep -E '^\+' | grep -v '^+++' \
    | grep -v 'secrets\.' | grep -v '${{' | head -5 || true)
if [ -n "$RESULT" ]; then
    report_issue "Hardcoded Docker Hub credentials found" "Matches in git history"
    echo "$RESULT"
else
    report_pass "No hardcoded Docker Hub credentials found"
fi

echo ""

# ==============================================
# 5. Authentication Patterns in Code
# ==============================================
echo "--- 5. Authentication Patterns in Code ---"

# Bearer tokens in code (actual token values, not variable references)
RESULT=$(git log --all -p -- '*.py' '*.js' '*.ts' '*.json' '*.yaml' '*.yml' 2>/dev/null \
    | grep -E '^\+' | grep -v '^+++' \
    | grep -iE 'Bearer\s+[A-Za-z0-9_\-]{20,}' \
    | grep -viE 'Bearer.*\$\{|Bearer.*{hf_token}|Bearer.*os\.|Bearer.*get_token|Bearer.*token' \
    | head -5 || true)
if [ -n "$RESULT" ]; then
    report_issue "Hardcoded Bearer tokens found" "Actual token values detected in git history"
    echo "$RESULT"
else
    report_pass "No hardcoded Bearer tokens found"
fi

# Authorization headers with embedded credentials
RESULT=$(git log --all -p -- '*.py' '*.js' '*.ts' 2>/dev/null \
    | grep -E '^\+' | grep -v '^+++' \
    | grep -iE 'Authorization.*[A-Za-z0-9]{30,}' \
    | grep -viE 'os\.environ\|get_token\|\$\{|Bearer.*{hf' \
    | head -5 || true)
if [ -n "$RESULT" ]; then
    report_issue "Hardcoded Authorization headers found" "Matches in git history"
    echo "$RESULT"
else
    report_pass "No hardcoded Authorization headers found"
fi

# os.environ reads of sensitive variables (informational, not necessarily a problem)
RESULT=$(grep -r -nE "os\.environ\.get\s*\(\s*['\"]?(HF_TOKEN|HUGGING_FACE_HUB_TOKEN|API_KEY|SECRET|PASSWORD|TOKEN)['\"]?" \
    --include="*.py" --exclude-dir=venv --exclude-dir=.git 2>/dev/null | head -10 || true)
if [ -n "$RESULT" ]; then
    report_warn "Environment variable reads for sensitive keys found" "Ensure these are only used to read from env, not hardcoded"
    echo "$RESULT"
else
    report_pass "No suspicious os.environ reads for sensitive keys"
fi

echo ""

# ==============================================
# 6. Personal Information (PII)
# ==============================================
echo "--- 6. Personal Information (PII) ---"

# Email addresses (filter out known non-personal patterns and Python decorators)
RESULT=$(git log --all -p -- '*.py' '*.json' '*.yaml' '*.yml' '*.md' '*.txt' '*.cfg' '*.ini' 2>/dev/null \
    | grep -E '^\+' | grep -v '^+++' \
    | grep -oE '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}' \
    | grep -ivE '(noreply|example\.com|test\.com|users\.noreply\.github|example\.org|github\.com|huggingface\.|\.post$|\.main$|\.dataclass$|\.no_grad$|\.inference_mode$)' \
    | grep -E '@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}' \
    | grep -vE '@(torch|app|dataclasses|hydra|pytest|typing|abc|overrides|deprecated|functools|contextlib|collections)\.' \
    | sort -u | head -10 || true)
if [ -n "$RESULT" ]; then
    report_warn "Personal email addresses found in committed files" "Review for PII concerns"
    echo "$RESULT"
else
    report_pass "No personal email addresses found in committed files"
fi

# Internal/private IP addresses
RESULT=$(git log --all -p -- '*.py' '*.json' '*.yaml' '*.yml' '*.conf' 2>/dev/null \
    | grep -E '^\+' | grep -v '^+++' \
    | grep -oE '(192\.168\.[0-9]{1,3}\.[0-9]{1,3}|10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3})' \
    | sort -u | head -10 || true)
if [ -n "$RESULT" ]; then
    report_warn "Private IP addresses found in committed files" "Review for internal network exposure"
    echo "$RESULT"
else
    report_pass "No private IP addresses found in committed files"
fi

# Local filesystem paths with usernames
RESULT=$(git log --all -p 2>/dev/null \
    | grep -E '^\+' | grep -v '^+++' \
    | grep -oE '/home/[a-zA-Z0-9_]+/' \
    | sort -u | head -10 || true)
if [ -n "$RESULT" ]; then
    report_warn "Local /home/ paths with usernames found" "May expose local usernames; verify these are not your own"
    echo "$RESULT"
else
    report_pass "No local /home/ paths with usernames found"
fi

RESULT=$(git log --all -p 2>/dev/null \
    | grep -E '^\+' | grep -v '^+++' \
    | grep -oE 'C:/Users/[a-zA-Z0-9_]+/' | sort -u | head -10 || true)
if [ -n "$RESULT" ]; then
    report_warn "Windows user paths found" "May expose Windows usernames"
    echo "$RESULT"
else
    report_pass "No Windows user paths found"
fi

echo ""

# ==============================================
# 7. Suspicious URLs & Webhooks
# ==============================================
echo "--- 7. Suspicious URLs & Webhooks ---"

RESULT=$(git log --all -p 2>/dev/null \
    | grep -iE '^\+.*(discord\.com/api/webhooks|hooks\.slack\.com)' | head -5 || true)
if [ -n "$RESULT" ]; then
    report_issue "Discord/Slack webhook URLs found" "These may contain embedded tokens"
    echo "$RESULT"
else
    report_pass "No Discord/Slack webhook URLs found"
fi

# Non-standard URLs with embedded auth parameters
RESULT=$(git log --all -p -- '*.py' '*.json' '*.yaml' '*.yml' '*.cfg' '*.ini' 2>/dev/null \
    | grep -E '^\+' | grep -v '^+++' \
    | grep -oE 'https?://[^ '"'"'")\]]+' \
    | grep -iE '(api_key|token|secret|password|auth)=' | head -10 || true)
if [ -n "$RESULT" ]; then
    report_issue "URLs with embedded auth parameters found" "Matches in git history"
    echo "$RESULT"
else
    report_pass "No URLs with embedded auth parameters found"
fi

echo ""

# ==============================================
# 8. Sensitive Files in Working Tree
# ==============================================
echo "--- 8. Sensitive Files in Working Tree ---"

for PATTERN in ".env" ".env.local" ".env.production" "*.pem" "*.key" "*.p12" "*.pfx" "id_rsa*" "id_ed25519*" "credentials.json" "service-account*.json"; do
    FOUND=$(find . -maxdepth 3 -name "$PATTERN" -not -path "./venv/*" -not -path "./.git/*" 2>/dev/null || true)
    if [ -n "$FOUND" ]; then
        for f in $FOUND; do
            if git check-ignore -q "$f" 2>/dev/null; then
                report_pass "$f exists but is gitignored"
            else
                report_issue "$f exists and is NOT gitignored" "Add to .gitignore immediately"
            fi
        done
    fi
done

echo ""

# ==============================================
# 9. Stash & Reflog Check
# ==============================================
echo "--- 9. Stash & Reflog Check ---"

STASH_COUNT=$(git stash list 2>/dev/null | wc -l || true)
if [ "$STASH_COUNT" -gt 0 ]; then
    report_warn "${STASH_COUNT} git stash entries found" "Stashes may contain sensitive data not visible in commit history"
    git stash list 2>/dev/null | head -5 || true
else
    report_pass "No git stash entries found"
fi

echo ""

# ==============================================
# 10. Untracked Sensitive Files
# ==============================================
echo "--- 10. Untracked Sensitive Files ---"

UNTRACKED=$(git ls-files --others --exclude-standard -- '*.env' '*.pem' '*.key' '*.p12' '*.secret' 2>/dev/null || true)
if [ -n "$UNTRACKED" ]; then
    report_warn "Untracked sensitive files found (but gitignored, not a threat)" "Files: ${UNTRACKED}"
    echo "$UNTRACKED"
else
    report_pass "No untracked sensitive files (that aren't gitignored)"
fi

# Check for files that are tracked but shouldn't be
TRACKED_SENSITIVE=$(git ls-files -- '*.env' '*.pem' '*.key' '*.p12' '*.secret' 'credentials.json' 'service-account*.json' 2>/dev/null || true)
if [ -n "$TRACKED_SENSITIVE" ]; then
    report_issue "Tracked sensitive files found" "These should be removed from git and added to .gitignore"
    echo "$TRACKED_SENSITIVE"
else
    report_pass "No tracked sensitive files (.env, .pem, .key, etc.)"
fi

echo ""

# ==============================================
# Summary
# ==============================================
echo "============================================"
if [ "$ISSUES_FOUND" -eq 0 ]; then
    echo -e "${GREEN}Audit complete. No sensitive information leaks detected.${NC}"
else
    echo -e "${RED}Audit complete. ISSUES FOUND — review the [FAIL] entries above.${NC}"
fi
echo "============================================"

exit "$ISSUES_FOUND"