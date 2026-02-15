# ============================================================
# Stage 1: Build the picoclaw binary
# ============================================================
FROM golang:1.26.0-alpine AS builder

RUN apk add --no-cache git make

WORKDIR /src

# Cache Go dependencies
COPY picoclaw/go.mod picoclaw/go.sum ./
RUN go mod download

# Copy picoclaw source and build
COPY picoclaw/ .
RUN make build

# ============================================================
# Stage 2: Minimal runtime image
# ============================================================
FROM alpine:3.21

RUN apk add --no-cache ca-certificates tzdata curl bash dos2unix

# Copy picoclaw binary
COPY --from=builder /src/build/picoclaw /usr/local/bin/picoclaw

# Initialize picoclaw home
RUN /usr/local/bin/picoclaw onboard 2>/dev/null || true

# Copy our workspace (overridden by volume mount in dev)
COPY workspace/ /root/.picoclaw/workspace/

# Fix Windows line endings in scripts
RUN find /root/.picoclaw/workspace -name "*.sh" -exec dos2unix {} \;

# Copy config template (overridden by volume mount in dev)
COPY config/config.example.json /root/.picoclaw/config.json

ENTRYPOINT ["picoclaw"]
CMD ["gateway"]
