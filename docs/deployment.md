# Deployment Guide

## Local Development

```bash
make install          # Create venv + install deps
make run-cloud        # Start cloud service (foreground)
# In another terminal:
make run-edge         # Run edge simulator
```

## Docker Deployment

### Build and Start

```bash
make docker-build     # Build cloud and edge images
make docker-up        # Start services
make docker-logs      # Follow logs
make docker-down      # Stop services
```

### Requirements

- Docker Engine 20+
- Docker Compose v2+
- Partitions exported to `partitions/` directory
- (For edge replay) Preprocessed data in `data/processed/`

### Services

| Service | Port | Description |
|---------|------|-------------|
| cloud | 8000 | FastAPI inference service |
| edge | — | Edge simulator (connects to cloud) |

### Resource Limits

The Docker Compose file simulates edge constraints:
- Cloud: 4 GB RAM, 4 CPUs
- Edge: 256 MB RAM, 1 CPU

### Health Checks

```bash
# Cloud health
curl http://localhost:8000/health

# Cloud readiness
curl http://localhost:8000/ready

# Available splits
curl http://localhost:8000/splits
```

## Environment Variables

See `.env.example` for all configurable variables.

Key variables:
- `UNISPLIT_CLOUD_PORT` — Cloud service port (default: 8000)
- `UNISPLIT_BACKEND_TYPE` — Inference backend (default: pytorch_cpu)
- `UNISPLIT_MEMORY_BUDGET` — Edge memory budget in bytes
- `UNISPLIT_CLOUD_URL` — Cloud URL for edge client
