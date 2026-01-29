#!/bin/bash
# =============================================================================
# Bash Gym - Default Verification Script
# =============================================================================
# This script is the default verifier for The Judge.
# Exit code 0 = verification passed (triggers Data Designer pipeline)
# Exit code non-zero = verification failed (trace archived for DPO)
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  Bash Gym - Verification"
echo "=========================================="

# Track test results
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test and track results
run_test() {
    local test_name="$1"
    local test_cmd="$2"
    
    TESTS_RUN=$((TESTS_RUN + 1))
    echo -n "Running: $test_name... "
    
    if eval "$test_cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}PASSED${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# =============================================================================
# Test Discovery and Execution
# =============================================================================

# 1. Check for pytest
if command -v pytest &> /dev/null; then
    echo "Found pytest, running Python tests..."
    
    if [ -d "tests" ] || ls test_*.py 1> /dev/null 2>&1 || ls *_test.py 1> /dev/null 2>&1; then
        if pytest -v --tb=short -x; then
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
        TESTS_RUN=$((TESTS_RUN + 1))
    fi
fi

# 2. Check for bats (Bash Automated Testing System)
if command -v bats &> /dev/null; then
    echo "Found bats, running Bash tests..."
    
    if ls *.bats 1> /dev/null 2>&1 || [ -d "test" ]; then
        if bats *.bats test/*.bats 2>/dev/null; then
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
        TESTS_RUN=$((TESTS_RUN + 1))
    fi
fi

# 3. Check for npm test
if [ -f "package.json" ]; then
    if grep -q '"test"' package.json; then
        echo "Found npm test script..."
        if npm test; then
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
        TESTS_RUN=$((TESTS_RUN + 1))
    fi
fi

# 4. Check for Makefile test target
if [ -f "Makefile" ]; then
    if grep -q "^test:" Makefile; then
        echo "Found Makefile test target..."
        if make test; then
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
        TESTS_RUN=$((TESTS_RUN + 1))
    fi
fi

# 5. Run custom verification if defined
if [ -f ".bashgym_verify" ]; then
    echo "Running custom verification..."
    source .bashgym_verify
fi

# =============================================================================
# Basic Sanity Checks
# =============================================================================

echo ""
echo "Running sanity checks..."

# Check that Python files have valid syntax
for pyfile in $(find . -name "*.py" -not -path "./.*" 2>/dev/null | head -20); do
    run_test "Syntax check: $pyfile" "python -m py_compile '$pyfile'"
done

# Check that shell scripts have valid syntax
for shfile in $(find . -name "*.sh" -not -path "./.*" 2>/dev/null | head -10); do
    run_test "Syntax check: $shfile" "bash -n '$shfile'"
done

# =============================================================================
# Results Summary
# =============================================================================

echo ""
echo "=========================================="
echo "  Verification Results"
echo "=========================================="
echo "Tests Run:    $TESTS_RUN"
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
echo "=========================================="

# Determine exit code
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}VERIFICATION FAILED${NC}"
    exit 1
elif [ $TESTS_RUN -eq 0 ]; then
    echo -e "${YELLOW}NO TESTS FOUND - Assuming success${NC}"
    exit 0
else
    echo -e "${GREEN}VERIFICATION PASSED${NC}"
    exit 0
fi
