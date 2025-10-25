# 1-Lipschitz-Neural-Distance-Fields-for-Configuration-Space-Shape-Modeling

Training the Model

This section describes how to train the 1-Lipschitz architecture on a block_world dataset using the provided training script.

Running the Training Script

To train the model, run the following command:

python train_lip.py block_world --data-dir inputs_cspace -o block_world_cspace_5000 -model sll -n-layers 20 -n-hidden 128 -ne 200 -bs 200

Explanation of the Command Arguments:

block_world: The name of the dataset to train on. This will be used to locate the training data files (e.g., block_world_Xtrain_in.npy, block_world_Xtrain_out.npy).

--data-dir inputs_cspace: Specifies the directory where the dataset files are located. In this case, the inputs_cspace directory.

-o block_world_cspace_5000: Specifies the output directory where the training results (such as model checkpoints) will be saved. The output will be stored in output/block_world_cspace_5000.

-model sll: Specifies the type of Lipschitz architecture to use. In this case, it will use the SLL model (as described in the research paper).

-n-layers 20: Defines the number of layers in the neural network. In this case, the network will have 20 layers.

-n-hidden 128: Defines the size of each hidden layer. Each layer will have 128 units.

-ne 200: Specifies the number of training epochs. In this case, the model will be trained for 200 epochs.

-bs 200: Sets the batch size for training. In this case, 200 samples will be processed in each batch.