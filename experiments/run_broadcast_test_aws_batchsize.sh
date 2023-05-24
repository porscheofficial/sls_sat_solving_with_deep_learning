#!/bin/bash
for config in experiments/configs_aws/broadcast_test/try_batchsizes/*.yaml; do
    python -m python.src.train --config $config
done