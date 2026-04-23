# NeMo base (CUDA/cuDNN/PyTorch/NeMo preinstalled)
FROM nvcr.io/nvidia/nemo:25.09

# Avoid interactive APT prompts
ENV DEBIAN_FRONTEND=noninteractive
RUN chmod 1777 /tmp

# Switch to root for package installs (NeMo images often default to user "nemo")
USER root

# System deps (add more if you compile native libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git vim unzip build-essential ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Modern pip toolchain (uses whatever Python is preinstalled in the NeMo image)
RUN python3 -m pip install --upgrade pip setuptools wheel

# ---- Poetry ----
ARG POETRY_VERSION=1.8.3
RUN pip install "poetry==${POETRY_VERSION}"

# App workspace; give ownership to the non-root user if present
WORKDIR /app
# If the "nemo" user exists in this image, chown to it; if not, this line is harmless.
# RUN if id -u nemo >/dev/null 2>&1; then chown -R nemo:nemo /app; fi

# Copy only dependency manifests first for layer caching
COPY pyproject.toml /app/pyproject.toml

# Make sure system CAs exist
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    update-ca-certificates && rm -rf /var/lib/apt/lists/*
# Set Poetry environment variables
ENV POETRY_VIRTUALENVS_CREATE=false
# Override base-image venv pointers that break Requests/Poetry
# Use system CA path (Debian/Ubuntu): /etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# (Optional but helpful) Show what certifi resolves to now for sanity
RUN python3 - <<'PY'
import certifi, sys
print("certifi.where() ->", certifi.where(), file=sys.stderr)
PY


# Install into the global environment (no venv in image)
RUN poetry config virtualenvs.create false \
 && poetry config installer.max-workers 10 


# Install runtime deps only by default (change INSTALL_DEV to true to include dev/test groups)
ARG INSTALL_DEV=false
RUN if [ "$INSTALL_DEV" = "true" ]; then \
        poetry lock --no-update \
        && poetry install --no-interaction --no-ansi --all-extras; \
    else \
        poetry lock --no-update \
        && poetry install --no-interaction --no-ansi --only main --no-root; \
    fi
# Bring in the rest of your project
COPY . .

# If your project is an installable package (has [tool.poetry] name/version),
# you can install it into the image (drop --no-root above and uncomment next line):
# RUN poetry install --no-interaction --no-ansi --only main --without dev

# Optional: AWS CLI v2 (comment out if not needed)
RUN curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
 && unzip -q awscliv2.zip \
 && ./aws/install \
 && rm -rf awscliv2.zip ./aws \
 && aws --version

# Python runtime niceties
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Drop privileges back to "nemo" if present
# RUN if id -u nemo >/dev/null 2>&1; then chown -R nemo:nemo /app; fi
# USER nemo

# Dev-friendly default; replace with your entrypoint for prod
CMD ["tail", "-f", "/dev/null"]
