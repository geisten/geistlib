"""Dependency-free Unix-socket transport for the resident geist daemon."""

from __future__ import annotations

import socket


def ask_geist(path: str, text: str, timeout: float) -> str:
    """Send one utterance and read the daemon's EOF-framed UTF-8 answer."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(path)
        connection.sendall(text.encode("utf-8").replace(b"\n", b" ") + b"\n")
        connection.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while True:
            chunk = connection.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace").strip()
