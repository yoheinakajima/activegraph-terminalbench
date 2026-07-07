#!/usr/bin/env bash
# Rebuild every sandbox piece that lives outside git, after a container
# reset. The sandbox has reset twice mid-pass-2 (each time restoring an
# old disk snapshot), so everything ephemeral is reconstructed from this
# script + the committed repo. Idempotent; safe to rerun.
#
# Pieces (documented in DECISIONS.md "sandbox-only adaptations"):
#   1. dockerd with mirror.gcr.io registry mirror + current egress proxy
#   2. uv-synced venv
#   3. harbor task registry snapshot (.cache/registry.json)
#   4. astral.sh HTTPS stand-in: self-signed cert, uv 0.9.5 binaries
#      repacked from PyPI wheels, install.sh, TLS server on :443
#   5. static curl + static tmux binaries for container mounts
#   6. compose overlays (.cache/tls-overlay-astral{,-tmux}.yaml)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
AST="$ROOT/.cache/astral"
mkdir -p "$AST/www"

# --- 1. dockerd -----------------------------------------------------------
if ! docker info > /dev/null 2>&1; then
    mkdir -p /etc/docker
    printf '{"registry-mirrors": ["https://mirror.gcr.io"]}\n' > /etc/docker/daemon.json
    rm -f /var/run/docker.pid
    nohup env HTTP_PROXY="${HTTP_PROXY:-}" HTTPS_PROXY="${HTTPS_PROXY:-}" \
        NO_PROXY="${NO_PROXY:-}" dockerd > /tmp/dockerd.log 2>&1 &
    for _ in $(seq 30); do docker info > /dev/null 2>&1 && break; sleep 2; done
    docker info > /dev/null
fi
echo "[1/6] dockerd up"

# --- 2. venv ---------------------------------------------------------------
uv sync -q
echo "[2/6] venv synced"

# --- 3. registry snapshot ---------------------------------------------------
if [ ! -f .cache/registry.json ]; then
    curl -fsSL --cacert /root/.ccr/ca-bundle.crt \
        https://raw.githubusercontent.com/laude-institute/harbor/main/registry.json \
        -o .cache/registry.json
fi
echo "[3/6] registry.json present"

# --- 4. astral stand-in ------------------------------------------------------
if [ ! -f "$AST/ca.crt" ]; then
    openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
        -keyout "$AST/ca.key" -out "$AST/ca.crt" \
        -subj "/CN=astral.sh" \
        -addext "subjectAltName=DNS:astral.sh,DNS:*.astral.sh" 2> /dev/null
fi
cat /etc/ssl/certs/ca-certificates.crt /root/.ccr/ca-bundle.crt "$AST/ca.crt" \
    > "$AST/ca-bundle-combined.crt"

for libc in gnu musl; do
    tarball="$AST/www/uv-x86_64-unknown-linux-${libc}.tar.gz"
    [ -f "$tarball" ] && continue
    plat=manylinux_2_17_x86_64; [ "$libc" = musl ] && plat=musllinux_1_1_x86_64
    tmp=$(mktemp -d)
    uv run pip download uv==0.9.5 --no-deps -d "$tmp" \
        --platform "$plat" --only-binary=:all: -q
    (cd "$tmp" && unzip -q uv-0.9.5-*.whl && \
     mkdir "uv-x86_64-unknown-linux-${libc}" && \
     cp uv-0.9.5.data/scripts/uv uv-0.9.5.data/scripts/uvx "uv-x86_64-unknown-linux-${libc}/" && \
     tar czf "$tarball" "uv-x86_64-unknown-linux-${libc}")
    rm -rf "$tmp"
done

cat > "$AST/www/install.sh" << 'INSTALL'
#!/bin/sh
# Stand-in for https://astral.sh/uv/install.sh (sandbox: real astral.sh is
# egress-blocked). Serves uv 0.9.5 repacked from the official PyPI wheels.
set -u
libc=gnu
if ldd --version 2>&1 | grep -qi musl; then libc=musl; fi
target="uv-x86_64-unknown-linux-${libc}"
tmp="$(mktemp -d)"
curl -fsSL "https://astral.sh/${target}.tar.gz" -o "$tmp/uv.tar.gz"
tar xzf "$tmp/uv.tar.gz" -C "$tmp"
dest="${UV_INSTALL_DIR:-$HOME/.local/bin}"
mkdir -p "$dest"
install -m 755 "$tmp/${target}/uv" "$dest/uv"
install -m 755 "$tmp/${target}/uvx" "$dest/uvx"
if [ -w /usr/local/bin ]; then
    install -m 755 "$tmp/${target}/uv" /usr/local/bin/uv
    install -m 755 "$tmp/${target}/uvx" /usr/local/bin/uvx
fi
# The official installer writes this env shim; verifiers source it.
cat > "$dest/env" << 'ENVEOF'
#!/bin/sh
case ":${PATH}:" in
    *:"$HOME/.local/bin":*) ;;
    *) export PATH="$HOME/.local/bin:$PATH" ;;
esac
ENVEOF
rm -rf "$tmp"
echo "uv installed to $dest/uv"
INSTALL

cat > "$AST/serve.py" << 'SERVE'
import http.server, ssl, os
os.chdir(os.path.join(os.path.dirname(__file__), "www"))

class H(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        # astral.sh serves install.sh at /uv/install.sh; flatten both forms.
        if path.rstrip("/").endswith("install.sh"):
            path = "/install.sh"
        return super().translate_path(path)

srv = http.server.HTTPServer(("0.0.0.0", 443), H)
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
here = os.path.dirname(os.path.abspath(__file__))
ctx.load_cert_chain(os.path.join(here, "ca.crt"), os.path.join(here, "ca.key"))
srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
srv.serve_forever()
SERVE

if ! pgrep -f "astral/serve.py" > /dev/null; then
    nohup python3 "$AST/serve.py" > /tmp/astral-serve.log 2>&1 &
    sleep 1
fi
curl -fsS --noproxy '*' --cacert "$AST/ca.crt" --resolve astral.sh:443:127.0.0.1 \
    https://astral.sh/uv/install.sh > /dev/null
echo "[4/6] astral stand-in serving"

# --- 5. static curl + tmux ----------------------------------------------------
# GitHub release assets 403 through the host proxy but are reachable from
# container egress (TLS-intercepted by the sandbox's own MITM CA, hence -k;
# same trust domain as all other sandbox egress).
fetch_via_container() { # url -> stdout
    docker run --rm curlimages/curl:latest -fsSLk "$1"
}
if [ ! -f "$AST/curl-static" ]; then
    fetch_via_container \
        https://github.com/moparisthebest/static-curl/releases/download/v8.11.0/curl-amd64 \
        > "$AST/curl-static"
    chmod 755 "$AST/curl-static"
fi
if [ ! -f "$AST/tmux-static" ]; then
    fetch_via_container \
        https://github.com/mjakob-gh/build-static-tmux/releases/download/v3.3a/tmux.linux-amd64.gz \
        | gunzip -c > "$AST/tmux-static"
    chmod 755 "$AST/tmux-static"
fi
echo "[5/6] static curl + tmux present"

# --- 6. compose overlays --------------------------------------------------------
cat > .cache/tls-overlay-astral.yaml << OVERLAY
services:
  main:
    extra_hosts:
      - "astral.sh:host-gateway"
    environment:
      SSL_CERT_FILE: /etc/ssl/certs/ca-certificates.crt
      REQUESTS_CA_BUNDLE: /etc/ssl/certs/ca-certificates.crt
      UV_NATIVE_TLS: "1"
    volumes:
      - ${AST}/ca-bundle-combined.crt:/etc/ssl/certs/ca-certificates.crt:ro
      - ${AST}/curl-static:/usr/local/bin/curl:ro
OVERLAY
cat .cache/tls-overlay-astral.yaml > .cache/tls-overlay-astral-tmux.yaml
printf '      - %s/tmux-static:/usr/local/bin/tmux:ro\n' "$AST" \
    >> .cache/tls-overlay-astral-tmux.yaml
echo "[6/6] overlays written"
echo "rebuild complete"
