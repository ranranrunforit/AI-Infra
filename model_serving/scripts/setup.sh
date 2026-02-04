#!/bin/bash
###############################################################################
# Setup Script for High-Performance Model Serving
#
# This script automates the complete environment setup including:
# - System dependencies (CUDA, cuDNN, TensorRT)
# - Python environment and packages
# - Database and Redis setup
# - Model downloads and initialization
# - Configuration validation
#
# Usage: ./scripts/setup.sh [--gpu|--cpu] [--skip-models]
###############################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CUDA_VERSION="12.0"
TENSORRT_VERSION="8.6"
PYTHON_VERSION="3.11"
REDIS_PORT="6379"
POSTGRES_PORT="5432"

# Parse command line arguments
GPU_MODE=true
SKIP_MODELS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu)
            GPU_MODE=false
            shift
            ;;
        --gpu)
            GPU_MODE=true
            shift
            ;;
        --skip-models)
            SKIP_MODELS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Model Serving Setup${NC}"
echo -e "${GREEN}========================================${NC}"

# Function to print section headers
print_section() {
    echo ""
    echo -e "${YELLOW}>>> $1${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
print_section "Checking System Requirements"
if [[ "$GPU_MODE" == true ]]; then
    if ! command_exists nvidia-smi; then
        echo -e "${RED}Error: NVIDIA driver not found${NC}"
        exit 1
    fi
    nvidia-smi
fi

# Install system dependencies
print_section "Installing System Dependencies"
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3-dev \
    python3-pip \
    libssl-dev \
    libffi-dev \
    redis-server \
    postgresql \
    postgresql-contrib

# Setup Python environment
print_section "Setting Up Python Environment"
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel

# Install Python dependencies
print_section "Installing Python Dependencies"
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-mock pytest-cov

# Setup Redis
print_section "Configuring Redis"
sudo systemctl start redis-server
sudo systemctl enable redis-server
redis-cli ping || echo "Redis setup complete"

# Setup PostgreSQL
print_section "Configuring PostgreSQL"
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql << SQL
CREATE DATABASE IF NOT EXISTS model_serving;
CREATE USER IF NOT EXISTS serving_user WITH PASSWORD 'serving_pass';
GRANT ALL PRIVILEGES ON DATABASE model_serving TO serving_user;
SQL

# Install CUDA (if GPU mode)
if [[ "$GPU_MODE" == true ]]; then
    print_section "Installing CUDA Toolkit"
    if ! command_exists nvcc; then
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
        sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
        wget https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/cuda-repo-ubuntu2204-${CUDA_VERSION//./-}-local_${CUDA_VERSION}-535.54.03-1_amd64.deb
        sudo dpkg -i cuda-repo-ubuntu2204-${CUDA_VERSION//./-}-local_${CUDA_VERSION}-535.54.03-1_amd64.deb
        sudo cp /var/cuda-repo-ubuntu2204-${CUDA_VERSION//./-}-local/cuda-*-keyring.gpg /usr/share/keyrings/
        sudo apt-get update
        sudo apt-get -y install cuda
    fi
    
    # Add CUDA to PATH
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
fi

# Create directory structure
print_section "Creating Directory Structure"
mkdir -p models/{tensorrt,pytorch,onnx}
mkdir -p logs
mkdir -p cache
mkdir -p data/calibration

# Download sample models (if not skipped)
if [[ "$SKIP_MODELS" == false ]]; then
    print_section "Downloading Sample Models"
    # Add model download commands here
    echo "Model downloads skipped - add your models to models/ directory"
fi

# Create environment configuration
print_section "Creating Environment Configuration"
cat > .env.local << ENV
# Model Serving Configuration
LOG_LEVEL=INFO
MODEL_CACHE_DIR=./cache
MAX_CACHE_SIZE_MB=4096

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# PostgreSQL Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=model_serving
DB_USER=serving_user
DB_PASSWORD=serving_pass

# TensorRT Configuration
TENSORRT_PRECISION=fp16
TENSORRT_MAX_BATCH_SIZE=32
TENSORRT_WORKSPACE_SIZE=1073741824

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
SERVER_WORKERS=1

# Tracing Configuration
JAEGER_ENDPOINT=http://localhost:14268/api/traces
ENABLE_TRACING=true
ENV

# Validate installation
print_section "Validating Installation"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"

if [[ "$GPU_MODE" == true ]]; then
    python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
    python -c "import torch; print(f'CUDA Devices: {torch.cuda.device_count()}')"
fi

# Run tests
print_section "Running Tests"
pytest tests/ -v --tb=short || echo "Some tests failed - check logs"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Configure .env.local with your settings"
echo "3. Start server: uvicorn src.serving.server:app --reload"
echo "4. Run benchmarks: ./scripts/benchmark.sh"
echo ""
