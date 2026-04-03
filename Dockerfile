# Stage 1 — build Rust binary
FROM rust:latest AS rust-builder
WORKDIR /build
COPY rust-core/ .
RUN cargo build --release

# Stage 2 — final image (Python + Rust binary + PyQt6)
FROM python:3.12-slim
WORKDIR /app

# PyQt6 needs these system libs for xcb/X11 rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libdbus-1-3 \
    libxcb-cursor0 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Rust sidecar binary
COPY --from=rust-builder /build/target/release/localai-core /usr/local/bin/localai-core

COPY assistant.py main.py ./

# Start Rust sidecar in background, launch GUI
CMD ["sh", "-c", "localai-core & python main.py"]
