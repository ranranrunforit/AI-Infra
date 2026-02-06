#!/bin/bash
# Monitor GPU usage
watch -n 1 'nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv'
