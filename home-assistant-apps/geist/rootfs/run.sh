#!/bin/sh
set -eu

runtime=/data/geist-home
socket=/data/geist.sock

if [ ! -x "$runtime" ]; then
    echo "geist_app status=runtime_missing"
    echo "P2.3.2 must install the verified embedded-model runtime at /data/geist-home."
    exit 1
fi

rm -f "$socket"
exec "$runtime" --serve "$socket"
