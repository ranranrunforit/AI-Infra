#!/bin/bash
# Quick training script
docker exec ray-head python -m src.training.distributed_trainer \
    --model resnet18 \
    --dataset cifar10 \
    --epochs ${EPOCHS:-10} \
    --batch-size ${BATCH_SIZE:-256} \
    --mixed-precision fp16 \
    "$@"
