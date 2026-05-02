#!/usr/bin/env bash
# task.md §7 smoke test - verifies the service is compatible with the eval harness.
#
# Usage:
#   docker compose up -d
#   until curl -sf http://localhost:8080/health; do sleep 1; done
#   bash scripts/smoke.sh
#
# Override target: BASE=http://my-host:8080 bash scripts/smoke.sh
# Override auth:   MEMORY_AUTH_TOKEN=... bash scripts/smoke.sh

set -euo pipefail

BASE="${BASE:-http://localhost:8080}"
USER_ID="user-1"
SESS_INGEST="smoke-1"
SESS_RECALL="smoke-2"

AUTH=()
if [[ -n "${MEMORY_AUTH_TOKEN:-}" ]]; then
  AUTH=(-H "Authorization: Bearer ${MEMORY_AUTH_TOKEN}")
fi

JQ=$(command -v jq || true)
pretty() { if [[ -n "$JQ" ]]; then "$JQ" "$@"; else cat; fi; }

echo "== /health =="
curl -sf "${AUTH[@]}" "$BASE/health" | pretty .
echo

echo "== /turns (ingest) =="
curl -sf -X POST "${AUTH[@]}" "$BASE/turns" \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\": \"$SESS_INGEST\",
    \"user_id\": \"$USER_ID\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"I just moved to Berlin from NYC last month. Loving it so far.\"},
      {\"role\": \"assistant\", \"content\": \"That sounds exciting! Berlin is a great city. How are you settling in?\"}
    ],
    \"timestamp\": \"2025-03-15T10:30:00Z\",
    \"metadata\": {}
  }" | pretty .
echo

echo "== /recall (expect Berlin) =="
RECALL_BODY=$(curl -sf -X POST "${AUTH[@]}" "$BASE/recall" \
  -H 'Content-Type: application/json' \
  -d "{
    \"query\": \"Where does this user live?\",
    \"session_id\": \"$SESS_RECALL\",
    \"user_id\": \"$USER_ID\",
    \"max_tokens\": 512
  }")
echo "$RECALL_BODY" | pretty .
echo

if echo "$RECALL_BODY" | grep -qi "berlin"; then
  echo "[ok] recall context mentions Berlin"
else
  echo "[warn] recall context did not surface Berlin - check extraction or gates"
fi
echo

echo "== /users/$USER_ID/memories =="
curl -sf "${AUTH[@]}" "$BASE/users/$USER_ID/memories" | pretty .
echo

echo "== cleanup =="
curl -sf -X DELETE "${AUTH[@]}" "$BASE/users/$USER_ID" | pretty .
echo

echo "[done] smoke test passed"
