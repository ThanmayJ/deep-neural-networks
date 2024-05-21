# cs6910_assignment2

This is the submission for partA section of Assignment 2.

`train.py`: Main code to train the CNN model

`question4.py`: solution for question 4 to plot 30 sample predictions and actual labels.

`utils.py`: Boiler plate functions

`Model.py`: Model containing the CNN architecture to be built after passing the CNN architecture details as command-line arguments.

### Command-line Arguments

    --wandb_project: Project name used for experiment tracking in Weights & Biases dashboard.
    --wandb_entity: Wandb Entity used for experiment tracking.
    --in_height, --in_width: Height and width of the input image in pixels.
    --in_channels: Number of channels in the input image.
    --activation: Activation function to be used.
    --filters: Number of filters to be used per layer.
    --num_layers: Number of convolutional layers in the network.
    --num_classes: Number of output classes.
    --dense_features: Number of neurons in the dense layer.
    --kernel_size: Size of the filter to be used in convolution and pooling.
    --stride: Stride to be used in convolution and pooling.
    --padding: Padding to be used in convolution and pooling.
    --dropout: Dropout rate after the dense layer.
    --use_batchnorm: Enable batch normalization after convolution and dense layer.
    --augmentation: Enable data augmentation.
    --batch_size: Batch size for training.
    --learning_rate: Learning rate for Adam optimizer.
    --epochs: Number of epochs to train the neural network.
    --no_wandb: Disable WandB if set True.
    --use_fp16: Use fp16 with mixed precision.
    --data_dir: Root directory of the dataset.
    --maintain_dist_after_split: Maintain class distributions when splitting train and validation sets.
    --manual_layers: Manually provide the number of filters per layer.
    --save_best_model: Save model with the best validation loss after training.
    --pretrained: Directory containing pre-trained model and optimizer.
