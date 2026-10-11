#!/usr/bin/env bash
set -euo pipefail

export DISPLAY="${DISPLAY:-:99}"
export HOME="${HOME:-/tmp/tokimon-browser-home}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/tokimon-browser-runtime}"
SCREEN_RESOLUTION="${SCREEN_RESOLUTION:-1400x900x24}"
VIEWPORT_SIZE="${VIEWPORT_SIZE:-1400,900}"
PROFILE_DIR="${PROFILE_DIR:-/profile}"
CLEAR_STALE_PROFILE_LOCKS="${CLEAR_STALE_PROFILE_LOCKS:-1}"
TOKIMON_RB_ALLOW_CHROMIUM_FALLBACK="${TOKIMON_RB_ALLOW_CHROMIUM_FALLBACK:-0}"

find_chrome() {
  local candidate

  for candidate in \
    /usr/bin/google-chrome-stable \
    /usr/bin/google-chrome; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  if [[ "$TOKIMON_RB_ALLOW_CHROMIUM_FALLBACK" == "1" ]]; then
    for candidate in \
    /usr/bin/chromium \
    /usr/bin/chromium-browser \
    /ms-playwright/chromium-*/chrome-linux/chrome \
    /ms-playwright/chromium-*/chrome-linux64/chrome; do
      if [[ -x "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return 0
      fi
    done
  fi

  echo "Google Chrome Stable is required by default. Install google-chrome-stable in the image, or set TOKIMON_RB_ALLOW_CHROMIUM_FALLBACK=1 for an explicit test/CI fallback." >&2
  return 1
}

mkdir -p "$PROFILE_DIR"
mkdir -p "$HOME" "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR" || true
if [[ "$CLEAR_STALE_PROFILE_LOCKS" == "1" ]]; then
  rm -f \
    "$PROFILE_DIR/SingletonCookie" \
    "$PROFILE_DIR/SingletonLock" \
    "$PROFILE_DIR/SingletonSocket"
fi

Xvfb "$DISPLAY" -screen 0 "$SCREEN_RESOLUTION" -nolisten tcp &
XVFB_PID="$!"

DISPLAY_READY_DEADLINE=$((SECONDS + 10))
while ! timeout 1s xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; do
  if ! kill -0 "$XVFB_PID" 2>/dev/null; then
    echo "Xvfb exited before display $DISPLAY became ready" >&2
    exit 1
  fi
  if (( SECONDS >= DISPLAY_READY_DEADLINE )); then
    echo "X display $DISPLAY did not become ready within 10 seconds" >&2
    exit 1
  fi
  sleep 0.1
done

openbox >/tmp/openbox.log 2>&1 &
OPENBOX_PID="$!"

x11vnc -display "$DISPLAY" -forever -shared -nopw -localhost -rfbport 5900 -quiet &
X11VNC_PID="$!"

websockify --web /usr/share/novnc/ 6080 localhost:5900 >/tmp/websockify.log 2>&1 &
WEBSOCKIFY_PID="$!"

socat TCP-LISTEN:9223,fork,bind=0.0.0.0,reuseaddr TCP:127.0.0.1:9222 &
CDP_PROXY_PID="$!"

CHROME="$(find_chrome)"
"$CHROME" \
  --disable-dev-shm-usage \
  --disable-features=Translate \
  --no-default-browser-check \
  --no-first-run \
  --remote-debugging-address=127.0.0.1 \
  --remote-debugging-port=9222 \
  --user-data-dir="$PROFILE_DIR" \
  --window-size="$VIEWPORT_SIZE" \
  about:blank &
CHROME_PID="$!"

for _ in $(seq 1 90); do
  if curl -fsS http://127.0.0.1:9222/json/version >/dev/null; then
    break
  fi
  sleep 0.5
done

if ! curl -fsS http://127.0.0.1:9222/json/version >/dev/null; then
  echo "Chrome CDP did not become ready" >&2
  exit 1
fi

wait -n "$XVFB_PID" "$OPENBOX_PID" "$X11VNC_PID" "$WEBSOCKIFY_PID" "$CDP_PROXY_PID" "$CHROME_PID"
