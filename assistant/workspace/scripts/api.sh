#!/bin/bash
# BashGym API helper for peony skills
# Usage: api.sh METHOD ENDPOINT [JSON_BODY]
# Examples:
#   api.sh GET /api/health
#   api.sh GET "/api/traces?status=gold"
#   api.sh POST /api/training/start '{"strategy":"sft","base_model":"<model-id>"}'
#   api.sh DELETE /api/orchestrate/job-123

set -euo pipefail

METHOD="${1:?Usage: api.sh METHOD ENDPOINT [BODY]}"
ENDPOINT="${2:?Usage: api.sh METHOD ENDPOINT [BODY]}"
BODY="${3:-}"
API_KEY="${BASHGYM_API_KEY:-}"

if [[ "$ENDPOINT" != /* ]]; then
  ENDPOINT="/$ENDPOINT"
fi

if [[ "$ENDPOINT" == /api/* ]]; then
  ENDPOINT="${ENDPOINT#/api}"
fi

normalize_api_base() {
  local raw="${1:-}"
  raw="${raw%/}"
  if [[ -z "$raw" ]]; then
    return 1
  fi
  if [[ "$raw" == */api ]]; then
    printf '%s\n' "$raw"
  else
    printf '%s/api\n' "$raw"
  fi
}

append_candidate() {
  local raw="${1:-}"
  local base
  base="$(normalize_api_base "$raw" 2>/dev/null || true)"
  if [[ -z "$base" ]]; then
    return 0
  fi
  if [[ " ${API_CANDIDATES[*]} " != *" $base "* ]]; then
    API_CANDIDATES+=("$base")
  fi
}

append_candidates_from_list() {
  local list="${1:-}"
  list="${list//,/ }"
  list="${list//;/ }"
  local item
  for item in $list; do
    append_candidate "$item"
  done
}

append_candidate_file() {
  local path="${1:-}"
  if [[ -n "$path" && -f "$path" ]]; then
    append_candidate "$(head -n 1 "$path")"
  fi
}

probe_api_base() {
  local base="$1"
  local timeout="${BASHGYM_API_DISCOVERY_TIMEOUT:-1.5}"
  curl -fsS --max-time "$timeout" \
    "$base/health" \
    -H "X-API-Key: $API_KEY" \
    >/dev/null 2>&1
}

API_CANDIDATES=()
append_candidate "${BASHGYM_API_URL:-}"
append_candidates_from_list "${BASHGYM_API_URLS:-}"
append_candidate_file "${BASHGYM_API_URL_FILE:-}"
append_candidate_file "$HOME/.bashgym/api_url"
append_candidate_file "$HOME/.config/bashgym/api_url"
if [[ -n "${BASHGYM_API_PORT:-}" ]]; then
  append_candidate "http://127.0.0.1:${BASHGYM_API_PORT}"
  append_candidate "http://localhost:${BASHGYM_API_PORT}"
fi
append_candidate "http://127.0.0.1:8003"
append_candidate "http://localhost:8003"
append_candidate "http://host.docker.internal:8003"
append_candidate "http://bashgym-api:8003"
append_candidate "http://127.0.0.1:8000"
append_candidate "http://localhost:8000"

API_BASE=""
for candidate in "${API_CANDIDATES[@]}"; do
  if probe_api_base "$candidate"; then
    API_BASE="$candidate"
    break
  fi
done

if [[ -z "$API_BASE" ]]; then
  echo "ERROR: Could not find a live BashGym API. Set BASHGYM_API_URL, BASHGYM_API_URLS, BASHGYM_API_PORT, or BASHGYM_API_URL_FILE." >&2
  echo "Tried: ${API_CANDIDATES[*]}" >&2
  exit 1
fi

CURL_ARGS=(
  -s
  -w "\n%{http_code}"
  -X "$METHOD"
  "${API_BASE}${ENDPOINT}"
  -H "Content-Type: application/json"
  -H "X-API-Key: $API_KEY"
)

if [ -n "$BODY" ]; then
  CURL_ARGS+=(-d "$BODY")
fi

RESPONSE=$(curl "${CURL_ARGS[@]}")
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY_OUT=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" -ge 400 ]; then
  echo "ERROR ($HTTP_CODE): $BODY_OUT" >&2
  exit 1
fi

echo "$BODY_OUT" | jq . 2>/dev/null || echo "$BODY_OUT"
