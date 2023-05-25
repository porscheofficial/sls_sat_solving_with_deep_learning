#!/bin/bash
for config in experiments/configs_aws/blocksworld/choosing_the_right_inv_temp/*.yaml; do
    python -m python.src.train --config $config
done