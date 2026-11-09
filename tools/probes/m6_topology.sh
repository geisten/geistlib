#!/bin/sh
# m6_topology.sh MODEL — the M6's CPU topology as macOS reports it, and the
# thread count geist's cpu_neon pool takes from it (#374). MODEL unused.
OUT=${PROBE_OUT:-probe-out}
mkdir -p "$OUT"
{ echo '```'; sysctl hw.ncpu hw.physicalcpu hw.logicalcpu hw.nperflevels 2>&1
  sysctl -a 2>/dev/null | grep -E '^hw\.perflevel[0-9]+\.(name|physicalcpu|logicalcpu|cpusperl2)'
  env | grep -E '^OMP_|^GEIST_|^KMP_' ; echo '```'; } | tee "$OUT/topology.md"
