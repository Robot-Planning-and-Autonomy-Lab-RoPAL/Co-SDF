#!/bin/bash
set -e

# Run escape.py for multiple environments
# block_world1.yaml
python3 escape.py --world-file config/block_world1.yaml --model-path output/block_world5/model_final.pt

# block_world2.yaml
if [ -f "config/block_world2.yaml" ]; then
    python3 escape.py --world-file config/block_world2.yaml --model-path output/block_world5/model_final.pt
fi

# block_world5.yaml
python3 escape.py --world-file config/block_world5.yaml --model-path output/block_world5/model_final.pt

# block_world10.yaml
if [ -f "config/block_world10.yaml" ]; then
    python3 escape.py --world-file config/block_world10.yaml --model-path output/block_world5/model_final.pt
fi

# block_world20.yaml
if [ -f "config/block_world20.yaml" ]; then
    python3 escape.py --world-file config/block_world20.yaml --model-path output/block_world5/model_final.pt
fi

echo "All escape runs completed."
