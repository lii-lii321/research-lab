# -*- coding: utf-8 -*-
"""轮询等待一个本地 URL 就绪（供 smoke 与 CI 使用；仅允许本机回环地址）。

用法：python scripts/wait_for.py http://127.0.0.1:8011/health [timeout_seconds]
"""
import ipaddress
import socket
import sys
import time
import urllib.parse
import urllib.request

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _assert_local(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    if parsed.scheme != "http" or host not in ALLOWED_HOSTS:
        raise SystemExit(f"only local http URLs are allowed: {url}")
    for _family, _type, _proto, _canonname, sockaddr in socket.getaddrinfo(host, 0):
        if not ipaddress.ip_address(sockaddr[0]).is_loopback:
            raise SystemExit(f"hostname does not resolve to loopback: {url}")


def wait_ready(url: str, timeout_s: float = 30.0) -> None:
    _assert_local(url)
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    print(f"READY {url}")
                    return
        except Exception as exc:  # 连接被拒/超时都属于"还没就绪"
            last_error = exc
        time.sleep(1.0)
    raise SystemExit(f"NOT READY within {timeout_s}s: {url} ({last_error})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python wait_for.py URL [timeout_seconds]")
    wait_ready(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 30.0)
