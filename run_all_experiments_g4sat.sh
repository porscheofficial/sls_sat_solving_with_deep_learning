#!/bin/bash
for config in experiments/configs/g4sat/*.yaml; do
    printf "Running experiment with config: $config\n"
    python -m python.src.train --config $config
done