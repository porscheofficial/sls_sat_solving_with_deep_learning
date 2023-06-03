#!/bin/bash
for config in experiments/configs_aws/random_3_SAT/right_layer_number/config_vcg*.yaml; do
    python -m python.src.train --config $config
done