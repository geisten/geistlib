# Documentation ownership

This repository documents the application-neutral Geist engine:

- `QUICKSTART.md`: C-library entry points
- `ARCHITECTURE.md`: engine layers and runtime architecture
- `DEPLOY.md`: building the library and consuming the packaged SDK
- `CI_COVERAGE.md`: what each CI job actually verifies
- the tool-use agent, the resident daemon and the normative
  `dynamic-tools-v1` contract live in
  [geisten/geistagent](https://github.com/geisten/geistagent)
- `../benchmark/`: engine and general-agent methodology/results

Home Assistant integration, policy, installation, app packaging, evaluations,
and implementation phases belong to
[`geisten/geist-home-assistant`](https://github.com/geisten/geist-home-assistant).
Core documentation may use HA as an adapter example but must not contain HA
setup instructions, product roadmaps, credentials, policy code, or release
requirements.
