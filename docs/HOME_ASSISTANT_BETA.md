# Home Assistant beta: fresh host to useful feedback

This is the external-tester path for the local geist Home Assistant appliance.
Use a non-critical HA instance or expose only one harmless light at first.

## Three-minute setup

1. Download the `geist-home-<platform>.tar.gz` asset and `SHA256SUMS` from the
   same release. Verify it with `sha256sum -c SHA256SUMS --ignore-missing`.
2. From the source/integration bundle, run:

   ```sh
   scripts/setup-home-assistant.sh \
     --ha-config /path/to/ha-config \
     --binary /path/to/geist-home \
     --activate
   ```

3. Restart HA if the script could not identify its container. Add **geist
   Conversation**, select it in an Assist pipeline, and expose one test light.
4. Try a status question, on, off, an ambiguous name, and a name that is not
   exposed. The last request must not operate the hidden entity.

## Report template

Open an issue using this body; never attach tokens, HA configuration, entity
registry files, private hostnames, IP addresses, or unredacted logs.

```text
Platform/OS:
RAM:
Home Assistant version/install type:
geist release + SHA256 verified: yes/no
Language: de/en
Setup completed in (minutes):
Status/on/off result:
Hidden-entity denial result:
Median perceived response time:
Unexpected restart or memory growth:
Redacted error/output:
```

## Three-minute demo recording script

This shot list doubles as the acceptance flow and avoids editing tricks:

- 0:00 show release asset and successful checksum verification;
- 0:20 run guided setup and explain that Phase-3 actions execute inside HA;
- 0:50 show active `geist-home` and private Unix socket permissions;
- 1:10 expose `light.flur` in HA and add the conversation integration;
- 1:35 ask status, turn it on, then turn it off;
- 2:10 request an existing but unexposed test entity and show it remains off;
- 2:35 show `scripts/check-home-assistant.sh` returning PASS;
- 2:50 show the local-only architecture and measured Pi 5 p50/p95.

Record the full terminal clock and HA state changes in one take. Blur only
tokens, private entity names, IP addresses, and user names.
