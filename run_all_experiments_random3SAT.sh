#!/bin/bash
for config in experiments/configs/random_3SAT/*.yaml; do
    python -m python.src.train --config $config
done