#!/bin/sh
set -eu

response=$(
    printf '%s\n' '{"type":"health"}' |
        socat -T 4 - UNIX-CONNECT:/data/geist.sock
)

test "$response" = '{"type":"health.result","protocol":"dynamic-tools-v1","status":"ready"}'
