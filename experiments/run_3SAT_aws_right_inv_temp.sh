#!/bin/bash
for config in experiments/configs_aws/random_3_SAT/right_inv_temp/*.yaml; do
    python -m python.src.train --config $config
done