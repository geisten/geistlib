#!/bin/sh
# agent-status.sh <issue-or-pr> <status> -- set the agent pipeline's status
# label (.gitea/workflows/agent.yml, AGENT.md §9): removes `agent` and every
# other status:* label, adds status:<status>, keeps the rest (ci:full, ...).
# Needs TOKEN and API (…/api/v1/repos/<owner>/<repo>) in the environment.
set -eu
n=$1 status=$2
auth="Authorization: token $TOKEN"
curl -sSf -H "$auth" "$API/issues/$n/labels" |
  jq --arg s "status:$status" \
     '{labels: ([.[].name | select(. != "agent" and (startswith("status:") | not))] + [$s])}' |
  curl -sSf -o /dev/null -X PUT -H "$auth" -H 'Content-Type: application/json' \
       -d @- "$API/issues/$n/labels"
