FROM mcr.microsoft.com/playwright/mcp:latest@sha256:3d871c22ea2d4cca0966e2cfb1860e1cb03eb7353725a3d6cffd133296fb04eb

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      gnupg \
      novnc \
      openbox \
      procps \
      socat \
      websockify \
      x11-utils \
      x11vnc \
      xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://dl.google.com/linux/linux_signing_key.pub \
      | gpg --dearmor -o /etc/apt/keyrings/google-linux.gpg \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/google-linux.gpg] https://dl.google.com/linux/chrome/deb/ stable main" \
      > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

COPY start-session-browser.sh /usr/local/bin/start-session-browser.sh
RUN chmod 755 /usr/local/bin/start-session-browser.sh

EXPOSE 6080 9223

ENTRYPOINT ["/usr/local/bin/start-session-browser.sh"]
