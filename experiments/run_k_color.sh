#!/bin/bash
for config in experiments/configs_aws/configs_k_color/*.yaml; do
    python -m python.src.train --config $config
done