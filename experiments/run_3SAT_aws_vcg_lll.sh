#!/bin/bash
for config in experiments/configs_aws/random_3_SAT/benchmark_vcg_lll/*.yaml; do
    python -m python.src.train --config $config
done