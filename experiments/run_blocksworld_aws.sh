#!/bin/bash
for config in experiments/configs_aws/blocksworld/*.yaml; do
    python -m python.src.train --config $config
done