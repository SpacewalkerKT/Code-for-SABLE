#!/usr/bin/env bash
###############################################################################
# SABLE - Execution Script
#
# Purpose: One-click execution entry point
#
# Usage:
#   ./run.sh                              # Use default configuration
#   CONFIG=path/to/config.yaml ./run.sh   # Use custom configuration
#
# Default: experiments/configs/ablation/sable_full.yaml
#           (Full SABLE model; see YAML for exact training budget)
###############################################################################

set -euo pipefail

# Project root directory (auto-detect)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Default configuration file
DEFAULT_CONFIG="experiments/configs/ablation/sable_full.yaml"

# Read configuration from environment variable or use default
CONFIG="${CONFIG:-$DEFAULT_CONFIG}"

###############################################################################
# Helper Functions
###############################################################################

log_info() {
    echo "[INFO] $*"
}

log_warn() {
    echo "[WARN] $*"
}

log_error() {
    echo "[ERROR] $*" >&2
}

print_banner() {
    cat <<'EOF'
###############################################################################
# SABLE: Sparse Bayesian Meta-Learning for Few-Shot Traffic Incident Recognition #
# Release Execution Script                                                       #
###############################################################################
EOF
}

###############################################################################
# Pre-execution Checks
###############################################################################

check_prerequisites() {
    log_info "Running environment pre-checks..."

    # Check Python
    if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
        log_error "Python interpreter not found"
        exit 1
    fi

    # Detect Python command
    PYTHON_CMD="python3"
    if ! command -v python3 &> /dev/null; then
        PYTHON_CMD="python"
    fi
    log_info "Python interpreter: $PYTHON_CMD"

    # Check configuration file
    if [ ! -f "$CONFIG" ]; then
        log_error "Configuration file not found: $CONFIG"
        log_error "Please verify the path or set CONFIG environment variable"
        exit 1
    fi
    log_info "Configuration file: $CONFIG"

    # Check main.py
    if [ ! -f "main.py" ]; then
        log_error "main.py not found"
        exit 1
    fi

    log_info "Pre-checks completed"
}

###############################################################################
# Main Execution
###############################################################################

main() {
    print_banner
    echo ""

    check_prerequisites
    echo ""

    # Build command
    log_info "Starting SABLE experiment..."
    echo ""

    CMD="$PYTHON_CMD main.py --config \"$CONFIG\""

    # Support additional command-line arguments (e.g., --episodes, --device)
    if [ $# -gt 0 ]; then
        CMD="$CMD $*"
    fi

    # Execute command
    eval $CMD
    exit_code=$?

    echo ""
    if [ $exit_code -eq 0 ]; then
        log_info "Experiment completed successfully"
    else
        log_error "Experiment failed (exit code: $exit_code)"
    fi

    exit $exit_code
}

###############################################################################
# Entry Point
###############################################################################

main "$@"
