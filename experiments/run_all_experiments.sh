#!/bin/bash
for config in experiments/configs/*.yaml; do
    python -m python.src.train --config $config
done