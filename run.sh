#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

command_name="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$command_name" in
  build)
    python3 src/build_contexts.py "$@"
    ;;
  run)
    python3 src/run_experiment.py "$@"
    ;;
  classify)
    python3 src/classify_response.py "$@"
    ;;
  analyze)
    python3 src/analyze_results.py "$@"
    ;;
  help|-h|--help)
    cat <<'EOF'
Usage: ./run.sh COMMAND [OPTIONS]

Commands:
  build       Build data/nq_open/contexts.json
  run         Run one model/trial (GPU recommended)
  classify    Classify one model's saved responses
  analyze     Aggregate all runs and regenerate curated figures

Examples:
  ./run.sh build
  ./run.sh run --model qwen2.5:3b --trial 1 --limit 5
  ./run.sh classify --model qwen2.5:3b
  ./run.sh analyze
EOF
    ;;
  *)
    echo "Unknown command: $command_name" >&2
    echo "Run './run.sh help' for usage." >&2
    exit 2
    ;;
esac
