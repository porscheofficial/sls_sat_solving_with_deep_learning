#!/bin/bash
for config in experiments/configs_aws/blocksworld/right_inv_temp/*.yaml; do
    python -m python.src.train --config $config
done