# Nebius Container Registry Guide

## Overview

This guide documents the correct way to push and pull Docker images to/from Nebius Container Registry based on the official [Nebius Container Registry Quickstart](https://docs.nebius.com/container-registry/quickstart).

## Key Learnings

### Registry Path Format

**❌ INCORRECT** (using registry name):
```bash
cr.eu-north1.nebius.cloud/csa-hiring-project-registry/gpu_cluster_testing:latest
```

**✅ CORRECT** (using registry ID suffix):
```bash
cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing:latest
```

### Understanding Registry IDs

When you create a registry, Nebius assigns it a full ID like:
```
registry-e00tnz9wpyxva2s992
```

The Docker image path uses only the **suffix after the hyphen**:
```
e00tnz9wpyxva2s992
```

## Setup Instructions

### 1. Install Prerequisites

**macOS:**
```bash
# Install Docker Desktop
brew install --cask docker

# Install Nebius CLI
curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash

# Install jq for JSON parsing
brew install jq
```

**Ubuntu:**
```bash
# Install Docker Engine
# See: https://docs.docker.com/engine/install/ubuntu/

# Install Nebius CLI
curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash

# Install jq
sudo apt-get install jq
```

### 2. Configure Nebius CLI

```bash
# Create/activate profile
nebius profile create

# Set your project ID (copy from web console)
nebius config set parent-id <your-project-id>

# Set region
export NB_REGION_ID=eu-north1  # or your region
```

### 3. Get Registry Information

```bash
# List your registries
nebius registry list

# Get registry details
nebius registry get <registry-id> --format json

# Extract registry path suffix
export NB_REGISTRY_PATH=$(nebius registry get <registry-id> --format json | jq -r '.metadata.id' | cut -d- -f 2)

echo $NB_REGISTRY_PATH
# Output: e00tnz9wpyxva2s992
```

### 4. Configure Docker Authentication

**One-time setup:**
```bash
nebius registry configure-helper
```

This adds credential helpers to `~/.docker/config.json`:
```json
{
  "credHelpers": {
    "cr.eu-north1.nebius.cloud": "nebius",
    "cr.eu-north2.nebius.cloud": "nebius",
    "cr.eu-west1.nebius.cloud": "nebius",
    "cr.me-west1.nebius.cloud": "nebius",
    "cr.uk-south1.nebius.cloud": "nebius",
    "cr.us-central1.nebius.cloud": "nebius"
  }
}
```

**Verify authentication:**
```bash
cat ~/.docker/config.json
```

### 5. Build, Tag, and Push

```bash
# Build image
docker build -t myapp:latest .

# Tag for Nebius registry (use registry path suffix!)
docker tag myapp:latest cr.$NB_REGION_ID.nebius.cloud/$NB_REGISTRY_PATH/myapp:latest

# Push to registry
docker push cr.$NB_REGION_ID.nebius.cloud/$NB_REGISTRY_PATH/myapp:latest
```

### 6. Pull and Run

```bash
# Pull from registry
docker pull cr.$NB_REGION_ID.nebius.cloud/$NB_REGISTRY_PATH/myapp:latest

# Run
docker run --rm cr.$NB_REGION_ID.nebius.cloud/$NB_REGISTRY_PATH/myapp:latest
```

## Project-Specific Values

For this GPU Cluster Testing Tool:

```bash
# Registry details
REGISTRY_NAME=csa-hiring-project-registry
REGISTRY_ID=registry-e00tnz9wpyxva2s992
REGISTRY_PATH=e00tnz9wpyxva2s992  # ← Use this in image paths!
REGISTRY_REGION=eu-north1

# Full image path
IMAGE=cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing:latest
```

## Complete Workflow Example

```bash
# 1. Configure authentication (one-time)
nebius registry configure-helper

# 2. Build the image
docker build -t cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing:latest .

# 3. Push to registry
docker push cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing:latest

# 4. Remove local image (to test pull)
docker rmi cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing:latest

# 5. Pull from registry
docker pull cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing:latest

# 6. Run the container
docker run --rm --gpus all \
  cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing:latest \
  --model resnet50 --batch-size 32
```

## Troubleshooting

### Error: "repository name not known to registry"

**Problem:**
```
Error response from daemon: repository name not known to registry: 
Entity Registry not found by id csa-hiring-project-registry
```

**Solution:**
You're using the registry name instead of the registry ID suffix. Extract the suffix:

```bash
# Get the registry ID
nebius registry get registry-e00tnz9wpyxva2s992 --format json | jq -r '.metadata.id'
# Output: registry-e00tnz9wpyxva2s992

# Use only the part after the hyphen in your image path
# ✅ e00tnz9wpyxva2s992
```

### Error: "unauthorized: authentication required"

**Problem:**
Docker credential helper not configured.

**Solution:**
```bash
nebius registry configure-helper
```

### Error: "no active Nebius profile"

**Problem:**
Nebius CLI not authenticated.

**Solution:**
```bash
nebius profile create
nebius profile list  # Verify active profile
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Build and Push

on:
  push:
    branches: [main]

env:
  REGISTRY: cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992
  IMAGE_NAME: gpu_cluster_testing

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Install Nebius CLI
        run: |
          curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
          echo "$HOME/.local/bin" >> $GITHUB_PATH
      
      - name: Configure Nebius authentication
        env:
          NEBIUS_TOKEN: ${{ secrets.NEBIUS_TOKEN }}
        run: |
          nebius config profile create --token "$NEBIUS_TOKEN"
          nebius registry configure-helper
      
      - name: Build and push
        run: |
          docker build -t ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} .
          docker tag ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
                     ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
          docker push ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          docker push ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
```

## Best Practices

1. **Always use registry ID suffix** in image paths, not the registry name
2. **Configure credential helper once** - no need for `docker login`
3. **Use environment variables** for region and registry path to make scripts portable
4. **Tag with both SHA and latest** for traceability and convenience
5. **Test pull after push** to verify the complete workflow

## References

- [Nebius Container Registry Quickstart](https://docs.nebius.com/container-registry/quickstart)
- [Nebius CLI Installation](https://docs.nebius.com/cli/install)
- [Creating and Modifying Registries](https://docs.nebius.com/container-registry/registries/manage)
- [Authentication](https://docs.nebius.com/container-registry/authentication)
