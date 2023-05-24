#!/bin/bash
for config in experiments/configs_aws/blocksworld/try_batchsizes/*.yaml; do
    python -m python.src.train --config $config
done