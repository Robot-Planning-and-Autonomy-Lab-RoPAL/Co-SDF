# Neural-Distance-Fields-for-Configuration-Space-Shape-Modeling

This repository provides tools for modeling Configuration Space (C-space) shapes using 1-Lipschitz Neural Distance Fields (NDFs). It includes scripts for data extraction, model training, and visualization, specifically demonstrated on a 2D arm configuration space.

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Project Structure

- `2Dexample/`: Contains the core scripts and configurations for the 2D arm environment.
  - `extract_dataset_pointcloud.py`: Script to generate C-space datasets.
  - `train_lip.py`: Script to train the 1-Lipschitz NDF model.
  - `config/`: Configuration files for the world environment.
  - `common/`: Shared utilities for models, training, and visualization.
- `requirements.txt`: Python dependencies.

## Data Extraction

Before training, you need to generate a C-space dataset. The `extract_dataset_pointcloud.py` script performs Gaussian sampling near the boundaries of the configuration space to create a point cloud dataset.

To generate the dataset:

```bash
cd 2Dexample
python extract_dataset_pointcloud.py --world config/block_world.yaml --n-train 5000 --n-test 1000 --near-ratio 0.7
```

- This will create an `inputs/` directory (by default) containing `.npy` files for training and testing.
- It also generates a visualization of the sampled points: `inputs/block_world_cspace_5000_samples.png`.

## Training the Model

Once the dataset is generated, you can train the 1-Lipschitz NDF model using `train_lip.py`.

### Running the Training Script

```bash
cd 2Dexample
python train_lip.py --dataset block_world --data inputs -o block_world_cspace_demo -model sll -n-layers 20 -n-hidden 128 -ne 200 -bs 200
```

### Explanation of Command Arguments:

- `--dataset block_world`: The prefix of the dataset files to load (e.g., `block_world_Xtrain_in.npy`).
- `--data inputs`: The directory where the generated dataset files are located.
- `-o block_world_cspace_demo`: The name of the output folder within `output/` where checkpoints and results will be saved.
- `-model sll`: Specifies the 1-Lipschitz architecture (SLL model).
- `-n-layers 20`: Number of layers in the network.
- `-n-hidden 128`: Hidden layer width.
- `-ne 200`: Number of training epochs.
- `-bs 200`: Training batch size.

## Visualization and Results

Training progress and results are saved in the `2Dexample/output/<output-name>/` directory:
- `log.csv`: Training metrics.
- `pc_0.geogram_ascii`: Initial point cloud visualization.
- `model_final.pt`: The trained model weights.
- Visualization images (if enabled) showing the NDF boundary during training.