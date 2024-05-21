# Feedforward Neural Networks

This is the implementation of a feedforward neural network using NumPy

The neural network architecture is flexible, allowing you to customize various parameters such as the number of layers, activation functions, optimizers, and more.

## Usage
To get started, clone this repository to your local machine:

The neural network can be trained using the train.py script. Below are the available command-line arguments:

    -wp, --wandb_project: Project name used to track experiments in Weights & Biases dashboard.
    -we, --wandb_entity: Wandb Entity used to track experiments in the Weights & Biases dashboard.
    -d, --dataset: Dataset to use (fashion_mnist or mnist).
    -e, --epochs: Number of epochs to train the neural network.
    -b, --batch_size: Batch size used for training.
    -l, --loss: Loss function (mean_squared_error or cross_entropy).
    -o, --optimizer: Optimizer to use (sgd, momentum, nag, rmsprop, adam, or nadam).
    -lr, --learning_rate: Learning rate used for optimization.
    -m, --momentum: Momentum used by momentum and nag optimizers.
    -beta, --beta: Beta used by RMSprop optimizer.
    -beta1, --beta1: Beta1 used by Adam and Nadam optimizers.
    -beta2, --beta2: Beta2 used by Adam and Nadam optimizers.
    -eps, --epsilon: Epsilon used by optimizers.
    -w_d, --weight_decay: Weight decay used by optimizers.
    -w_i, --weight_init: Weight initialization method (random or xavier).
    -nhl, --num_layers: Number of hidden layers in the neural network.
    -sz, --hidden_size: Number of neurons in each hidden layer.
    -a, --activation: Activation function (identity, sigmoid, tanh, or relu).
    -nw, --no_wandb: If set, wandb won't get logged and outputs are printed on terminal itself.

Example usage:

```python train.py -wp myprojectname -we myname -d fashion_mnist -e 10 -b 32 -l cross_entropy -o adam -lr 0.001 -nhl 2 -sz 64 -a relu```

The default hyperparameter values have been set to the best hyperparameters that I got use wandb sweeps.

## Note
If you need to add additional optimizers, loss functions, activation functions, refer to `functional.py` and add those in the arguments in `train.py`. Ensure that the name of the function/class is (or includes) the name of the argument, as it is called using Python's `globals()` function in `train.py`. You can refer to the exisiting implementations in `functional.py` script to understand this better.
