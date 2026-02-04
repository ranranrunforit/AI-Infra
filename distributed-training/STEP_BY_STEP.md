# Step-by-Step Implementation Guide
## Project 09: Distributed Training Platform with Ray

This comprehensive guide walks through implementing the distributed training platform from scratch, explaining every design decision and implementation detail.

## Table of Contents

1. [Development Environment Setup](#1-development-environment-setup)
2. [Understanding Distributed Training](#2-understanding-distributed-training)
3. [Ray Train Architecture](#3-ray-train-architecture)
4. [Implementing the Training Loop](#4-implementing-the-training-loop)
5. [NCCL Optimization](#5-nccl-optimization)
6. [Fault Tolerance and Checkpointing](#6-fault-tolerance-and-checkpointing)
7. [Hyperparameter Tuning](#7-hyperparameter-tuning)
8. [Kubernetes Deployment](#8-kubernetes-deployment)
9. [Monitoring and Observability](#9-monitoring-and-observability)
10. [Performance Optimization](#10-performance-optimization)
11. [Production Best Practices](#11-production-best-practices)

---

## 1. Development Environment Setup

### 1.1 Local Development Environment

**Objective**: Set up a local environment for development and testing.

```bash
# Create project directory
mkdir -p ~/distributed-training-platform
cd ~/distributed-training-platform

# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install CUDA toolkit (if not already installed)
# Ubuntu/Debian:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-3

# Verify CUDA
nvcc --version
nvidia-smi

# Install PyTorch with CUDA support
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install Ray
pip install "ray[default,train]==2.9.0"

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import ray; print(f'Ray: {ray.__version__}')"
```

**Design Decision**: We use Python 3.11 for improved performance and type hint support. CUDA 12.3 provides the latest optimizations for modern GPUs.

### 1.2 Understanding Your Hardware

```python
# hardware_discovery.py
import torch

def discover_gpu_capabilities():
    """Discover GPU capabilities for optimization"""

    if not torch.cuda.is_available():
        print("No CUDA GPUs available")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")

    for i in range(num_gpus):
        print(f"\n=== GPU {i} ===")
        props = torch.cuda.get_device_properties(i)
        print(f"Name: {props.name}")
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"Multi-Processors: {props.multi_processor_count}")
        print(f"Max Threads per Block: {props.max_threads_per_block}")
        print(f"CUDA Cores: ~{props.multi_processor_count * 128}")  # Approximate

        # Check NVLink connectivity
        if torch.cuda.device_count() > 1:
            print(f"\nChecking NVLink connectivity...")
            # This requires nvidia-ml-py3
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                nvlink_count = pynvml.nvmlDeviceGetNvLinkState(handle, 0)
                print(f"NVLink Status: {nvlink_count}")
            except:
                print("NVLink status unavailable")

if __name__ == "__main__":
    discover_gpu_capabilities()
```

**Run this to understand your hardware**:
```bash
python hardware_discovery.py
```

This information is critical for:
- Setting optimal batch sizes
- Configuring NCCL parameters
- Understanding memory constraints
- Choosing parallelism strategies

---

## 2. Understanding Distributed Training

### 2.1 Distributed Training Strategies

**Data Parallelism** (what we implement):
- Each GPU has a complete copy of the model
- Data is split across GPUs
- Gradients are averaged using AllReduce
- Best for models that fit in single GPU memory

```
GPU 0: Model Copy + Data Batch 0 → Gradients → AllReduce
GPU 1: Model Copy + Data Batch 1 → Gradients → AllReduce
GPU 2: Model Copy + Data Batch 2 → Gradients → AllReduce
GPU 3: Model Copy + Data Batch 3 → Gradients → AllReduce
         ↓
    Updated Model Parameters (synchronized)
```

**Model Parallelism** (not implemented here, but important to understand):
- Model is split across GPUs
- Used when model doesn't fit in single GPU
- More complex communication patterns

**Pipeline Parallelism**:
- Model stages on different GPUs
- Micro-batching for efficiency
- Used for very large models

### 2.2 Communication Patterns

**AllReduce**: The core operation for data parallelism

```python
# Conceptual AllReduce
# Each GPU has gradients: [g0, g1, g2, g3]
# After AllReduce, each GPU has: [avg(g0), avg(g1), avg(g2), avg(g3)]

def all_reduce_visualization():
    """
    Ring AllReduce algorithm visualization
    This is what NCCL implements efficiently
    """
    import numpy as np

    # 4 GPUs, each with 4 gradient values
    gradients = {
        0: np.array([1.0, 2.0, 3.0, 4.0]),
        1: np.array([5.0, 6.0, 7.0, 8.0]),
        2: np.array([9.0, 10.0, 11.0, 12.0]),
        3: np.array([13.0, 14.0, 15.0, 16.0])
    }

    print("Before AllReduce:")
    for gpu, grads in gradients.items():
        print(f"GPU {gpu}: {grads}")

    # Ring AllReduce (simplified)
    # In practice, NCCL uses optimized ring algorithm
    averaged = np.mean([grads for grads in gradients.values()], axis=0)

    print("\nAfter AllReduce:")
    for gpu in gradients:
        print(f"GPU {gpu}: {averaged}")

    print(f"\nCommunication volume: {len(averaged) * 4 * 4} bytes per GPU")
    print("(4 values × 4 bytes/float × communication rounds)")

if __name__ == "__main__":
    all_reduce_visualization()
```

**Key Insight**: AllReduce is bandwidth-bound. Optimizing network and NCCL is crucial.

### 2.3 Scaling Efficiency

**Ideal Scaling**: Time(N GPUs) = Time(1 GPU) / N
**Reality**: Communication overhead reduces efficiency

```python
def calculate_scaling_efficiency():
    """Calculate expected scaling efficiency"""

    # Measured throughput (samples/sec)
    throughput = {
        1: 892,
        2: 1748,
        4: 3432,
        8: 6512
    }

    baseline = throughput[1]

    print("Scaling Analysis:")
    print(f"{'GPUs':<6} {'Throughput':<12} {'Speedup':<10} {'Efficiency':<12}")
    print("-" * 50)

    for gpus, tput in throughput.items():
        speedup = tput / baseline
        efficiency = speedup / gpus
        print(f"{gpus:<6} {tput:<12.1f} {speedup:<10.2f}x {efficiency:<12.1%}")

    # Analyze communication overhead
    print("\nCommunication Overhead Analysis:")
    for gpus in [2, 4, 8]:
        expected = baseline * gpus
        actual = throughput[gpus]
        overhead = (expected - actual) / expected
        print(f"{gpus} GPUs: {overhead:.1%} overhead")

if __name__ == "__main__":
    calculate_scaling_efficiency()
```

**Output**:
```
Scaling Analysis:
GPUs   Throughput   Speedup    Efficiency
--------------------------------------------------
1      892.0        1.00x      100.0%
2      1748.0       1.96x      98.0%
4      3432.0       3.85x      96.2%
8      6512.0       7.30x      91.2%

Communication Overhead Analysis:
2 GPUs: 2.0% overhead
4 GPUs: 3.8% overhead
8 GPUs: 8.8% overhead
```

**Insight**: Communication overhead increases with GPU count. NCCL optimization becomes critical at 8+ GPUs.

---

## 3. Ray Train Architecture

### 3.1 Why Ray Train?

**Ray Train provides**:
1. **Automatic distributed setup**: Handles process groups, rank assignment
2. **Fault tolerance**: Automatic restarts and checkpointing
3. **Resource management**: Efficient GPU allocation
4. **Scalability**: Easy to scale from 2 to 100+ GPUs
5. **Integration**: Works with PyTorch, TensorFlow, Horovod

**Architecture**:
```
┌─────────────────────────────────────────────────┐
│              Ray Cluster                         │
├─────────────────────────────────────────────────┤
│                                                  │
│  Ray Head Node (Orchestrator)                   │
│  - Job scheduling                                │
│  - Checkpoint coordination                       │
│  - Metrics aggregation                           │
│                                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐│
│  │ Worker 0   │  │ Worker 1   │  │ Worker 2   ││
│  │ Rank 0     │  │ Rank 1     │  │ Rank 2     ││
│  │ GPU 0-1    │  │ GPU 2-3    │  │ GPU 4-5    ││
│  │            │  │            │  │            ││
│  │ Training   │  │ Training   │  │ Training   ││
│  │ Loop       │  │ Loop       │  │ Loop       ││
│  └────────────┘  └────────────┘  └────────────┘│
│         │               │               │        │
│         └───────────────┴───────────────┘        │
│                NCCL AllReduce                    │
└─────────────────────────────────────────────────┘
```

### 3.2 Ray Train Components

```python
# components_explained.py
from ray.train import Trainer, ScalingConfig, Checkpoint
from ray.train.torch import TorchTrainer

def explain_ray_components():
    """Explain Ray Train components with examples"""

    # 1. ScalingConfig - defines distributed setup
    scaling_config = ScalingConfig(
        num_workers=4,              # Number of training workers
        use_gpu=True,               # Use GPUs
        resources_per_worker={
            "GPU": 2,               # 2 GPUs per worker
            "CPU": 8                # 8 CPUs per worker
        },
        placement_strategy="SPREAD"  # Spread workers across nodes
    )

    print("ScalingConfig defines:")
    print(f"- Total GPUs: {scaling_config.num_workers * 2} GPUs")
    print(f"- Workers per node: 1 (with SPREAD strategy)")
    print(f"- Total world size: {scaling_config.num_workers}")

    # 2. train_loop_config - hyperparameters passed to training function
    train_loop_config = {
        "lr": 0.1,
        "batch_size": 256,
        "epochs": 90
    }

    # 3. TorchTrainer - orchestrates distributed training
    trainer = TorchTrainer(
        train_loop_per_worker=lambda config: print("Training!"),
        train_loop_config=train_loop_config,
        scaling_config=scaling_config
    )

    print("\nTorchTrainer responsibilities:")
    print("- Initialize Ray workers")
    print("- Setup PyTorch distributed process group")
    print("- Distribute training function to workers")
    print("- Collect and aggregate results")
    print("- Handle failures and restarts")

if __name__ == "__main__":
    explain_ray_components()
```

### 3.3 Training Function Structure

```python
def train_func(config):
    """
    This function runs on EACH worker
    Ray Train handles the distributed setup automatically
    """

    # 1. Get distributed context (Ray handles this)
    from ray.train import get_context

    context = get_context()
    world_rank = context.get_world_rank()  # 0, 1, 2, 3, ...
    local_rank = context.get_local_rank()   # GPU ID on this node
    world_size = context.get_world_size()   # Total number of workers

    print(f"Worker initialized: rank {world_rank}/{world_size}, local GPU {local_rank}")

    # 2. Setup device
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # 3. Create model and wrap with DDP
    model = create_model()
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    # 4. Create distributed data loader
    # Ray Train automatically creates DistributedSampler
    train_loader = create_dataloader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True  # Shuffle within worker
    )

    # 5. Training loop
    for epoch in range(config["epochs"]):
        for batch in train_loader:
            # Forward pass
            outputs = model(batch)
            loss = criterion(outputs, targets)

            # Backward pass (gradients automatically synchronized via DDP)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # 6. Report metrics to Ray (aggregated across workers)
        from ray import train as ray_train
        ray_train.report({"loss": loss.item(), "epoch": epoch})

        # 7. Checkpoint (Ray handles distributed checkpointing)
        if epoch % 10 == 0:
            checkpoint = Checkpoint.from_dict({"epoch": epoch, "model": model.state_dict()})
            ray_train.report(metrics={"epoch": epoch}, checkpoint=checkpoint)
```

**Key Points**:
1. Ray Train calls this function on each worker
2. Each worker gets its rank and world size automatically
3. DDP handles gradient synchronization
4. Ray aggregates metrics and checkpoints

---

## 4. Implementing the Training Loop

### 4.1 Basic Training Loop (Single GPU)

Let's start simple and add complexity:

```python
# basic_training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def train_single_gpu():
    """Basic training loop on single GPU"""

    # Setup
    device = torch.device("cuda:0")

    # Model
    model = models.resnet50(weights=None, num_classes=10)
    model = model.to(device)

    # Data
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4
    )

    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(10):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} completed, Avg Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    train_single_gpu()
```

**Benchmark this**: `python basic_training.py`

Measure:
- Training time per epoch
- GPU utilization (`nvidia-smi` in another terminal)
- Memory usage

### 4.2 Adding Distributed Training

Now let's add PyTorch DDP:

```python
# distributed_training.py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_distributed(rank, world_size):
    """Initialize distributed training"""

    # Set environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # Use NCCL for GPU
        rank=rank,
        world_size=world_size
    )

    print(f"Initialized rank {rank}/{world_size}")

def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()

def train_distributed(rank, world_size):
    """Training function for each process/GPU"""

    # Setup distributed
    setup_distributed(rank, world_size)

    # Set device for this process
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Create model and wrap with DDP
    model = models.resnet50(weights=None, num_classes=10)
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    # Create distributed data loader
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        transform=transform
    )

    # Key: Use DistributedSampler to split data across GPUs
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,  # Per-GPU batch size
        sampler=train_sampler,  # Use sampler, not shuffle
        num_workers=8,
        pin_memory=True
    )

    # Optimizer and criterion
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(10):
        # IMPORTANT: Set epoch for sampler (affects shuffling)
        train_sampler.set_epoch(epoch)

        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward (DDP automatically handles gradient synchronization)
            optimizer.zero_grad()
            loss.backward()  # Gradients are averaged across GPUs
            optimizer.step()

            if rank == 0 and batch_idx % 10 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

    cleanup_distributed()

def main():
    """Main entry point"""
    world_size = torch.cuda.device_count()
    print(f"Training on {world_size} GPUs")

    # Spawn processes (one per GPU)
    mp.spawn(
        train_distributed,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
```

**Key Differences from Single GPU**:
1. `DistributedSampler` splits data across GPUs
2. `DDP` wraps model and synchronizes gradients
3. Each GPU runs in separate process
4. Effective batch size = `batch_size × world_size`

**Run it**: `python distributed_training.py`

Compare with single GPU:
- Should be ~2x faster on 2 GPUs
- Check GPU utilization on all GPUs

### 4.3 Adding Ray Train

Now let's use Ray Train for better orchestration:

```python
# ray_distributed_training.py
import ray
from ray.train import Trainer, ScalingConfig
from ray.train.torch import TorchTrainer

def train_func_ray(config):
    """Training function for Ray Train"""

    import torch
    from ray.train import get_context

    # Get distributed context (Ray handles setup)
    context = get_context()
    world_rank = context.get_world_rank()
    local_rank = context.get_local_rank()
    world_size = context.get_world_size()

    # Setup device
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Create model
    model = models.resnet50(weights=None, num_classes=10)
    model = model.to(device)

    # Ray Train automatically wraps with DDP
    # No need to manually wrap!

    # Create data loader (Ray handles DistributedSampler)
    train_dataset = datasets.CIFAR10(root='./data', train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,  # Ray handles distributed shuffling
        num_workers=8
    )

    # Training loop
    optimizer = optim.SGD(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["epochs"]):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Report metrics to Ray (automatically aggregated)
        from ray import train as ray_train
        ray_train.report({"loss": loss.item(), "epoch": epoch})

def main():
    """Main entry point with Ray Train"""

    # Initialize Ray
    ray.init()

    # Configure scaling
    scaling_config = ScalingConfig(
        num_workers=2,  # 2 GPU workers
        use_gpu=True,
        resources_per_worker={"GPU": 1}
    )

    # Create trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_ray,
        train_loop_config={
            "lr": 0.1,
            "batch_size": 256,
            "epochs": 10
        },
        scaling_config=scaling_config
    )

    # Train
    result = trainer.fit()
    print(f"Training completed: {result.metrics}")

    ray.shutdown()

if __name__ == "__main__":
    main()
```

**Advantages of Ray Train**:
1. Automatic DDP setup
2. Built-in checkpoint management
3. Fault tolerance
4. Easy to scale to multi-node
5. Better resource management

---

## 5. NCCL Optimization

NCCL (NVIDIA Collective Communications Library) is critical for multi-GPU performance.

### 5.1 Understanding NCCL

```python
# nccl_explained.py
import torch
import torch.distributed as dist

def test_nccl_performance():
    """Test NCCL communication performance"""

    # Initialize distributed
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    # Test AllReduce performance with different sizes
    sizes = [1024, 1024*1024, 10*1024*1024, 100*1024*1024]  # 1KB to 100MB

    print(f"Rank {rank}: Testing NCCL AllReduce performance")

    for size in sizes:
        # Create tensor
        tensor = torch.randn(size // 4, device=device)  # 4 bytes per float

        # Warm up
        for _ in range(10):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()

        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        iterations = 100
        start.record()
        for _ in range(iterations):
            dist.all_reduce(tensor)
        end.record()
        torch.cuda.synchronize()

        time_ms = start.elapsed_time(end) / iterations
        bandwidth_gbps = (size * 4) / (time_ms / 1000) / 1e9  # Convert to GB/s

        if rank == 0:
            print(f"Size: {size/1024/1024:.1f}MB, Time: {time_ms:.2f}ms, "
                  f"Bandwidth: {bandwidth_gbps:.2f} GB/s")

    dist.destroy_process_group()
```

### 5.2 NCCL Environment Variables

**Critical environment variables for optimization**:

```bash
# nccl_config.sh

# Enable debug output (helpful during setup)
export NCCL_DEBUG=INFO

# InfiniBand settings (if available)
export NCCL_IB_DISABLE=0              # Enable InfiniBand
export NCCL_NET_GDR_LEVEL=5           # GPU Direct RDMA level
export NCCL_IB_GID_INDEX=3            # InfiniBand GID index

# Network interface
export NCCL_SOCKET_IFNAME=eth0        # Primary network interface
# Or for multiple interfaces:
export NCCL_SOCKET_IFNAME=eth0,eth1

# Performance tuning
export NCCL_NSOCKS_PERTHREAD=4        # Sockets per thread (higher = more parallel)
export NCCL_BUFFSIZE=2097152          # Buffer size (2MB default)
export NCCL_MIN_NRINGS=8              # Minimum number of rings
export NCCL_MAX_NRINGS=16             # Maximum number of rings

# Topology awareness
export NCCL_TOPO_FILE=/path/to/topo.xml  # Custom topology file

# For AWS with EFA (Elastic Fabric Adapter)
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1

# For debugging
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV
```

**How to tune for your setup**:

```python
# nccl_tuning_guide.py

def generate_nccl_config(hardware_profile):
    """Generate optimized NCCL configuration"""

    config = {}

    if hardware_profile == "nvlink":
        # NVLink between GPUs (best case)
        config.update({
            "NCCL_P2P_LEVEL": "NVL",
            "NCCL_NVLINK_ENABLE": "1",
            "NCCL_MIN_NRINGS": "16",
            "NCCL_MAX_NRINGS": "16"
        })
        print("Optimized for NVLink (400-600 GB/s)")

    elif hardware_profile == "infiniband":
        # InfiniBand network (100-200 Gbps)
        config.update({
            "NCCL_IB_DISABLE": "0",
            "NCCL_NET_GDR_LEVEL": "5",
            "NCCL_IB_GID_INDEX": "3",
            "NCCL_MIN_NRINGS": "8",
            "NCCL_MAX_NRINGS": "16"
        })
        print("Optimized for InfiniBand (12-25 GB/s)")

    elif hardware_profile == "ethernet_10g":
        # 10 Gbps Ethernet
        config.update({
            "NCCL_IB_DISABLE": "1",
            "NCCL_SOCKET_IFNAME": "eth0",
            "NCCL_NSOCKS_PERTHREAD": "2",
            "NCCL_MIN_NRINGS": "4",
            "NCCL_MAX_NRINGS": "8"
        })
        print("Optimized for 10G Ethernet (1.25 GB/s)")

    elif hardware_profile == "cloud":
        # Cloud environment (AWS/GCP/Azure)
        config.update({
            "NCCL_SOCKET_IFNAME": "eth0",
            "NCCL_NSOCKS_PERTHREAD": "4",
            "NCCL_MIN_NRINGS": "4",
            "NCCL_MAX_NRINGS": "8",
            "NCCL_BUFFSIZE": "2097152"
        })
        print("Optimized for cloud networking")

    return config

def apply_nccl_config(config):
    """Apply NCCL configuration"""
    import os
    for key, value in config.items():
        os.environ[key] = str(value)
        print(f"Set {key}={value}")

# Example usage
hardware = "nvlink"  # Change based on your hardware
config = generate_nccl_config(hardware)
apply_nccl_config(config)
```

### 5.3 Measuring NCCL Performance

```python
# measure_nccl.py
import torch
import torch.distributed as dist
import time

def benchmark_collective(operation, tensor_size_mb=100):
    """Benchmark a collective operation"""

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    # Create tensor
    tensor_size = int(tensor_size_mb * 1024 * 1024 / 4)  # Convert MB to float elements
    tensor = torch.randn(tensor_size, device=device)

    # Warm up
    for _ in range(10):
        if operation == "allreduce":
            dist.all_reduce(tensor)
        elif operation == "broadcast":
            dist.broadcast(tensor, src=0)
        elif operation == "allgather":
            tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(tensor_list, tensor)

    torch.cuda.synchronize()

    # Benchmark
    iterations = 100
    start_time = time.time()

    for _ in range(iterations):
        if operation == "allreduce":
            dist.all_reduce(tensor)
        elif operation == "broadcast":
            dist.broadcast(tensor, src=0)
        elif operation == "allgather":
            tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(tensor_list, tensor)

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / iterations * 1000

    # Calculate bandwidth
    data_size_mb = tensor_size_mb
    if operation == "allgather":
        data_size_mb *= world_size  # All processes receive world_size copies

    bandwidth_gbps = (data_size_mb / 1024) / (avg_time_ms / 1000)

    if rank == 0:
        print(f"{operation}: {avg_time_ms:.2f}ms, {bandwidth_gbps:.2f} GB/s")

    return avg_time_ms, bandwidth_gbps

if __name__ == "__main__":
    dist.init_process_group(backend='nccl')

    for size_mb in [10, 100, 500]:
        print(f"\nTensor size: {size_mb}MB")
        benchmark_collective("allreduce", size_mb)
        benchmark_collective("broadcast", size_mb)
        benchmark_collective("allgather", size_mb)

    dist.destroy_process_group()
```

**Expected Results**:
- NVLink: 200-400 GB/s
- InfiniBand (200Gbps): 20-25 GB/s
- 10G Ethernet: 1-1.2 GB/s

---

## 6. Fault Tolerance and Checkpointing

### 6.1 Checkpoint Strategy

**What to save**:
1. Model state dict
2. Optimizer state dict
3. Learning rate scheduler state
4. Training step/epoch
5. Random number generator states
6. Best metrics

```python
# checkpointing.py
import torch
import os
from pathlib import Path
import shutil

class CheckpointManager:
    """Manages training checkpoints with versioning"""

    def __init__(self, checkpoint_dir, keep_last_n=3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n

    def save(self, state_dict, epoch, step, is_best=False):
        """Save checkpoint"""

        # Create checkpoint filename
        checkpoint_name = f"checkpoint_epoch{epoch}_step{step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Save checkpoint
        torch.save(state_dict, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # Save as latest
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        shutil.copy(checkpoint_path, latest_path)

        # Save as best if applicable
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            shutil.copy(checkpoint_path, best_path)
            print(f"New best checkpoint saved")

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keep last N"""

        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        # Keep last N checkpoints
        for checkpoint in checkpoints[self.keep_last_n:]:
            checkpoint.unlink()
            print(f"Removed old checkpoint: {checkpoint.name}")

    def load(self, checkpoint_path=None):
        """Load checkpoint"""

        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "checkpoint_latest.pt"
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path)
        print(f"Checkpoint loaded: {checkpoint_path}")

        return state_dict

    def get_latest_checkpoint(self):
        """Get path to latest checkpoint"""
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        return latest_path if latest_path.exists() else None
```

**Usage**:

```python
# In training loop
checkpoint_manager = CheckpointManager("/mnt/checkpoints", keep_last_n=3)

# Save checkpoint
checkpoint_data = {
    "epoch": epoch,
    "step": global_step,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "best_accuracy": best_accuracy,
    "rng_state": torch.get_rng_state(),
    "cuda_rng_state": torch.cuda.get_rng_state()
}

checkpoint_manager.save(checkpoint_data, epoch, global_step, is_best=is_best)

# Load checkpoint
checkpoint_data = checkpoint_manager.load()
model.load_state_dict(checkpoint_data["model_state_dict"])
optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
start_epoch = checkpoint_data["epoch"]
```

### 6.2 Automatic Recovery

Ray Train provides automatic recovery:

```python
# fault_tolerant_training.py
from ray.train import Checkpoint, RunConfig, FailureConfig

def train_with_fault_tolerance():
    """Training with automatic fault tolerance"""

    # Configure failure handling
    failure_config = FailureConfig(
        max_failures=3,  # Retry up to 3 times
        fail_fast=False  # Don't fail immediately
    )

    run_config = RunConfig(
        name="fault-tolerant-training",
        checkpoint_config=CheckpointConfig(
            checkpoint_frequency=1000,  # Checkpoint every 1000 steps
            checkpoint_at_end=True
        ),
        failure_config=failure_config
    )

    trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config
    )

    result = trainer.fit()
    print(f"Training completed with {result.num_failures} failures")
```

---

Continuing to sections 7-11...

[Due to length constraints, sections 7-11 cover:
- Hyperparameter Tuning with Ray Tune
- Kubernetes Deployment (GPU scheduling, resource quotas)
- Monitoring Setup (Prometheus, Grafana, DCGM)
- Performance Optimization techniques
- Production Best Practices]

---

## Summary

This guide walked through:
1. ✅ Setting up development environment
2. ✅ Understanding distributed training fundamentals
3. ✅ Implementing Ray Train architecture
4. ✅ Building the training loop progressively
5. ✅ Optimizing NCCL for maximum performance
6. ✅ Implementing fault-tolerant checkpointing
