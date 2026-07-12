# Geist Home Assistant app

This is the protected-app scaffold for the resident Geist runtime. It supports
`aarch64` and `amd64`, exposes no host port, requests no Supervisor, Home
Assistant, Docker, device, audio, video or host-namespace access, and mounts no
Home Assistant directory. `/data` is the only persistent location.

The app healthcheck sends the model-free dynamic-tools-v1 health frame over
`/data/geist.sock`. The app contains no HTTP/REST server.

## Current scaffold limitation

P2.3.1 deliberately does not download or embed an unverified runtime. Until
P2.3.2 installs the signed/checksummed embedded-model artifact, place a matching
executable at `/data/geist-home` only in a development installation. There is no
supported end-user installation yet.

The future HA OS integration transport must carry dynamic-tools-v1 privately on
the app network without publishing a host port. It must not mount `/config` to
share this internal Unix socket.
