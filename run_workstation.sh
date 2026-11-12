#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
root=$PWD
python="$root/.venv/bin/python"
if [[ ! -x "$python" ]]; then
    printf '%s\n' 'BioTrace needs a .venv with the AI engine dependencies. See the setup steps in README.md.' >&2
    exit 1
fi
if ! command -v mvn >/dev/null || ! command -v java >/dev/null; then
    printf '%s\n' 'BioTrace needs Java and Maven on PATH.' >&2
    exit 1
fi

# Pick a loopback port so an existing workstation can keep running.
port=$("$python" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
export BIOTRACE_API_URL="http://127.0.0.1:$port"
BIOTRACE_WORKER_THREADS="${BIOTRACE_WORKER_THREADS:-1}" "$python" -m uvicorn biotrace.main:create_app --factory --app-dir ai-engine/src \
    --host 127.0.0.1 --port "$port" &
backend_pid=$!
cleanup() {
    kill "$backend_pid" 2>/dev/null || true
    wait "$backend_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'Waiting for BioTrace backend at %s\n' "$BIOTRACE_API_URL"
for ((attempt = 0; attempt < 120; attempt++)); do
    if ! kill -0 "$backend_pid" 2>/dev/null; then
        printf '%s\n' 'BioTrace backend exited before it became ready.' >&2
        exit 1
    fi
    if curl --fail --silent --max-time 1 "$BIOTRACE_API_URL/api/v1/health" >/dev/null; then
        cd "$root/java-client"
        mvn -q compile exec:java
        exit
    fi
    sleep 0.5
done
printf '%s\n' 'Timed out waiting for the BioTrace backend.' >&2
exit 1
