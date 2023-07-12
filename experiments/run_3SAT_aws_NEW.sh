#!/bin/bash
for config in experiments/configs_aws/random_3SAT_NEW/*.yaml; do
    python -m python.src.train --config $config
done