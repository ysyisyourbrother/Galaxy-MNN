#!/bin/bash

while true; do
    busy_percentage=$(cat /sys/class/kgsl/kgsl-3d0/gpu_busy_percentage)
    echo "GPU Busy Percentage: $busy_percentage%"
    sleep 0.0001 # 每隔 1 秒更新一次
done