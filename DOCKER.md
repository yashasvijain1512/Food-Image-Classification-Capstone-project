# Docker Setup Guide

This guide explains how to build and run the Food Image Classification application using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, for docker-compose commands)
- Pre-trained model files in the `models/` directory

## Quick Start with Docker Compose

The easiest way to run the application is using Docker Compose:

```bash
# Build and start the container
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

The API will be available at `http://localhost:8000`

## Manual Docker Commands

### Build the Docker image

```bash
docker build -t food-classifier:latest .
```

### Run the container

```bash
docker run -d \
  --name food-classifier \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e MODEL_DIR=/app/models \
  -e IMG_SIZE=224 \
  -e TOP_K=3 \
  food-classifier:latest
```

### Access the API

- API Documentation: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc
- Health check: http://localhost:8000/health

### View logs

```bash
docker logs -f food-classifier
```

### Stop and remove the container

```bash
docker stop food-classifier
docker rm food-classifier
```

## Environment Variables

The following environment variables can be configured:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `/app/models` | Path to the models directory |
| `IMG_SIZE` | `224` | Image size for preprocessing |
| `TOP_K` | `3` | Number of top predictions to return |

## Building for Production

For production deployments, consider:

1. **Multi-stage build**: The Dockerfile already uses multi-stage builds to minimize image size
2. **Image optimization**: Remove development dependencies
3. **Security**: Use specific version tags instead of latest
4. **Registry**: Push to Docker Hub or your container registry

```bash
# Build with specific version
docker build -t food-classifier:1.0.0 .

# Tag for registry
docker tag food-classifier:1.0.0 your-registry/food-classifier:1.0.0

# Push to registry
docker push your-registry/food-classifier:1.0.0
```

## Troubleshooting

### Container fails to start
- Check that model files exist in `models/` directory
- Verify the Dockerfile has correct file paths
- Review logs: `docker logs food-classifier`

### Out of memory errors
Increase Docker's memory allocation in Docker Desktop settings

### Model loading issues
Ensure `best_model.h5` or `saved_model/` directory exists in `models/`
Ensure `class_names.txt` is present in `models/` directory

## Health Check

The container includes a health check that verifies the API is responding. View health status:

```bash
docker ps --filter name=food-classifier
```

The health check endpoint is: `GET /health`
