# Documentation ownership

This repository documents the application-neutral Geist engine:

- `QUICKSTART.md`: CLI and C-library entry points
- `ARCHITECTURE.md`: engine layers and runtime architecture
- `DEPLOY.md`: embedded/external models and resident dynamic-tools service
- `agent.md`: generic tool-use agent and adapter contract
- `proposals/dynamic-tools-v1.md`: normative dynamic-tools wire contract
- `../benchmark/`: engine and general-agent methodology/results

Home Assistant integration, policy, installation, app packaging, evaluations,
and implementation phases belong to
[`geisten/geist-home-assistant`](https://github.com/geisten/geist-home-assistant).
Core documentation may use HA as an adapter example but must not contain HA
setup instructions, product roadmaps, credentials, policy code, or release
requirements.
