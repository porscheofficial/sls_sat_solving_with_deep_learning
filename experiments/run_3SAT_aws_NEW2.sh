#!/bin/bash
for config in experiments/configs_aws/random_3SAT_NEW2/*.yaml; do
    python -m python.src.train --config $config
done