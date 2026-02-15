# ============================================================
# Stage 1: Build the peony binary
# ============================================================
FROM golang:1.26.0-alpine AS builder

RUN apk add --no-cache git make

WORKDIR /src

# Cache Go dependencies
COPY picoclaw/go.mod picoclaw/go.sum ./
RUN go mod download

# Copy peony source and build
COPY picoclaw/ .
RUN make build

# ============================================================
# Stage 2: Minimal runtime image
# ============================================================
FROM alpine:3.21

RUN apk add --no-cache ca-certificates tzdata curl bash dos2unix jq

# Copy peony binary
COPY --from=builder /src/build/peony /usr/local/bin/peony

# Initialize peony home
RUN /usr/local/bin/peony onboard 2>/dev/null || true

# Copy our workspace (overridden by volume mount in dev)
COPY workspace/ /root/.peony/workspace/

# Fix Windows line endings in scripts
RUN find /root/.peony/workspace -name "*.sh" -exec dos2unix {} \;

# Copy config template (overridden by volume mount in dev)
COPY config/config.example.json /root/.peony/config.json

# Create non-root user and transfer ownership
RUN adduser -D -h /home/peony peony && \
    mv /root/.peony /home/peony/.peony && \
    chown -R peony:peony /home/peony/.peony
USER peony

ENTRYPOINT ["peony"]
CMD ["gateway"]
