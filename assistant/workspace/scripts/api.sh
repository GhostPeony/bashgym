#!/bin/bash
# BashGym API helper for peony skills
# Usage: api.sh METHOD ENDPOINT [JSON_BODY]
# Examples:
#   api.sh GET /health
#   api.sh GET /traces?status=gold
#   api.sh POST /training/start '{"strategy":"sft","base_model":"Qwen/Qwen2.5-Coder-1.5B-Instruct"}'
#   api.sh DELETE /orchestrator/job-123

set -euo pipefail

METHOD="${1:?Usage: api.sh METHOD ENDPOINT [BODY]}"
ENDPOINT="${2:?Usage: api.sh METHOD ENDPOINT [BODY]}"
BODY="${3:-}"
API_URL="${BASHGYM_API_URL:?BASHGYM_API_URL not set}"

CURL_ARGS=(
  -s
  -w "\n%{http_code}"
  -X "$METHOD"
  "${API_URL}${ENDPOINT}"
  -H "Content-Type: application/json"
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

echo "$BODY_OUT"
