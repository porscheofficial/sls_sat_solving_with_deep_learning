#!/bin/bash
for config in experiments/configs_aws/random_3_SAT/layer_number_LLL/*.yaml; do
    python -m python.src.train --config $config
done