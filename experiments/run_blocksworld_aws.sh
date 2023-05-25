#!/bin/bash
for config in experiments/configs_aws/blocksworld/benchmark_lcg_vcg/*.yaml; do
    python -m python.src.train --config $config
done