FROM ghcr.io/home-assistant/base:3.22

ARG BUILD_ARCH
ARG BUILD_VERSION=0.1.0

RUN apk add --no-cache socat

COPY rootfs /
RUN chmod 0755 /run.sh /healthcheck.sh

LABEL \
    io.hass.version="${BUILD_VERSION}" \
    io.hass.type="app" \
    io.hass.arch="${BUILD_ARCH}"

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["/healthcheck.sh"]
CMD ["/run.sh"]
