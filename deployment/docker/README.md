# Docker Setup for Fraud Detection Project

This directory contains Docker configuration files to run the fraud detection project in a containerized environment.

## Quick Start

### Build and Run with Docker Compose

```bash
cd deployment/docker
docker-compose up --build
```

This will:
- Build a Docker image with all dependencies
- Start a Jupyter notebook server
- Mount your notebooks, data, and models directories

### Access Jupyter

Once the container is running, access Jupyter at:
- **URL**: http://localhost:8888
- The access token will be displayed in the terminal output

### Run with Docker directly

```bash
# Build the image
docker build -f deployment/docker/Dockerfile -t fraud-detection:latest .

# Run the container
docker run -p 8888:8888 \
  -v $(pwd)/notebooks:/app/notebooks \
  -v $(pwd)/Dataset:/app/Dataset \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  fraud-detection:latest
```

## What's Included

- Python 3.11
- All required data science libraries (pandas, numpy, scikit-learn, etc.)
- Jupyter Notebook server
- Pre-configured to access your notebooks and data

## Volumes

The following directories are mounted as volumes:
- `notebooks/` - Your Jupyter notebooks
- `Dataset/` - Your dataset files
- `data/` - Processed data
- `models/` - Trained models

Changes to these directories will be reflected immediately in the container.

## Stopping the Container

```bash
docker-compose down
```

Or if running directly:
```bash
docker stop fraud-detection-jupyter
```

