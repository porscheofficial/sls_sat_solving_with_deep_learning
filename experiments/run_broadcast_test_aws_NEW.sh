#!/bin/bash
for config in experiments/configs_aws/broadcast_test_NEW/*.yaml; do
    python -m python.src.train --config $config
done