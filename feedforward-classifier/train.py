import wandb
import pickle

import argparse

from keras.datasets import fashion_mnist, mnist
import numpy as np

from functional import *

np.random.seed(seed=187)

def normalize_data(x):
    x = (x - x.mean()) / x.std()
    return x

def get_validation_from_train(x_train, y_train, x=10):
    num_validation_samples = int(0.01*x*len(x_train)) # x percent of train set is taken for validaiton
    validation_indices = np.random.choice(len(x_train), num_validation_samples, replace=False) # random indices to be taken for validation set according to seed

    x_valid = x_train[validation_indices]
    y_valid = y_train[validation_indices]

    x_train = np.delete(x_train, validation_indices, axis=0)
    y_train = np.delete(y_train, validation_indices, axis=0)

    return x_train, y_train, x_valid, y_valid

def get_one_hot_vector(y):
    outputs = np.zeros((np.max(y)+1,len(y)))
    for i in range(len(y)):
        outputs.T[i][y[i]] = 1
    return outputs


class MultiClassClassifier:
    def __init__(self, n, nonlinear, weight_init, loss_fn, optimizer, lr, momentum, beta, beta1, beta2, epsilon, weight_decay):
      self.n = n
      self.nonlinear = globals()[nonlinear]
      self.nonlinear_prime = globals()[nonlinear+'_prime']
      
      self.initialize_weights(weight_init)
      self.optimizer = globals()["optimizer_"+optimizer](self.n)
      self.loss_fn = globals()[loss_fn+"_loss"]

      self.lr = lr
      self.momentum = momentum
      self.beta = beta
      self.beta1 = beta1
      self.beta2 = beta2
      self.epsilon = epsilon
      self.weight_decay = weight_decay

    def store_to_path(self, path):
        with open(path, 'wb') as f:
            pickle.dump({"W":self.W, "b":self.b, "n":self.n}, f)
    
    def load_from_path(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.n = data["n"]
        self.W = data["W"]
        self.b = data["b"]

    def initialize_weights(self, weight_init):
        self.W = {}
        self.b = {}

        for i in range(1, len(self.n), 1):
            self.W[i], self.b[i] = globals()[weight_init+"_init"](self.n[i-1], self.n[i])



    def Linear(self, i, final=False):
        if self.optimizer == 'nag':
            self.W[i] = self.W[i] - self.optimizer.W_velocity[i]
            self.b[i] = self.b[i] - self.optimizer.b_velocity[i]

        self.A[i] = linear(self.H[i-1], self.W[i], self.b[i])
        # print("Layer", i)
        # print("W", self.W[i].shape, self.W[i])
        # print("b", self.b[i].shape, self.b[i])
        # print("Z", self.A[i].shape, self.A[i])
        if final==False:
            self.H[i] = self.nonlinear(self.A[i])
        else:
            self.H[i] = globals()['softmax'](self.A[i])
        # print("A", self.H[i].shape, self.H[i])
    
    def forward(self, X):
        self.A = {}
        self.H = {}
        # X.shape = (num_features, num_examples)
        self.H[0] = X

        # Feedforward upto layer L-1
        for i in range(1, len(self.n)-1, 1):
            self.Linear(i)

        # Feedforward to final (output) layer to get predictions using softmax
        L = len(self.n)-1
        self.Linear(L, final=True)

        out = self.H[L]
        return out
    
    def backward(self, y, pred):
        m = y.shape[1]
        dLdW = {}
        dLdb = {}

        dLdA = {}
        dHdA = {} # dHdA[i] = H'(A[i])
        dAdH = self.W
        dAdW = {i:self.H[i-1] for i in range(1,len(self.n),1)} # dAdW[i] = H[i-1]
        dAdb = {i:np.ones((1, m)) for i in range(1,len(self.n),1)} # dAdb[i] = 1
        
        # Backpropagate for W and b in last layer (L)
        L = len(self.n)-1
        
        if self.loss_fn.__name__ == "cross_entropy_loss":
            dLdA[L] = (pred - y) # shape (n[L]=10, bs)
        elif self.loss_fn.__name__ == "mean_squared_error_loss":
            k = y.shape[0] # Num of Classes
            dLdA[L] = np.matmul( softmax_prime(self.A[L]), (2 / m) * (pred - y) )
            # dLdA[L] = ((pred - y)*pred*(1- pred)) / m
        else:
            raise NotImplementedError("Only supports cross_entropy_loss/mean_squared_error_loss")

        # print("dLdZ[L]",dLdA[L])
        dLdW[L] = np.matmul(dLdA[L], dAdW[L].T) / m
        dLdb[L] = np.matmul(dLdA[L], dAdb[L].T)  / m
        dLdW[L] += self.weight_decay * self.W[L] # L2 Regularization
        dLdb[L] += self.weight_decay * self.b[L] # L2 Regularization
        # print("dLdW[L]",dLdW[L]) 
        # print("dLdb[L]",dLdb[L]) 
        # print(dLdA[L].shape) # shape (n[L]=10, bs)
        # print(dAdW[L].shape) # shape (n[L-1], bs)
        # print(dLdW[L].shape) # shape (n[L]=10, n[L-1])
        
        # Backpropagate for W and b from L-1 th layer to first layer
        for i in range(L-1, 0, -1):
            dHdA[i] = self.nonlinear_prime(self.A[i]) 
            # print(f"dAdZ[{i}]",dHdA[i]) 
            # print(dLdA[i+1].shape) # shape (n[i+1], bs)
            # print(dAdH[i+1].shape) # shape (n[i+1], n[i])
            # print(dHdA[i].shape) # shape (n[i], bs)
            dLdA[i] = np.matmul(dLdA[i+1].T, dAdH[i+1]).T * dHdA[i]
            # print(f"dLdZ[{i}]",dLdA[i]) 
            # print(dLdA[i].shape) # shape (n[i], bs)
            dLdW[i] = np.matmul(dLdA[i], dAdW[i].T) / m
            dLdb[i] = np.matmul(dLdA[i], dAdb[i].T) / m
            # print(f"dLdW[{i}]",dLdW[i]) 
            # print(f"dLdb[{i}]",dLdb[i])

            dLdW[i] += self.weight_decay * self.W[i] # L2 Regularization
            dLdb[i] += self.weight_decay * self.b[i] # L2 Regularization

 
        
        self.dLdW = dLdW
        self.dLdb = dLdb

        # return dLdW, dLdb
    
    def Optimize(self):
        self.W, self.b = self.optimizer.optimize(self.W, self.b, self.dLdW, self.dLdb, self.lr, self.momentum, self.beta, self.beta1, self.beta2, self.epsilon, self.weight_decay)

def train_model(model, x, y_train, normalize, num_epochs, batch_size, valid_inputs, y_valid, log_wandb=True):
    y = get_one_hot_vector(y_train)
    valid_outputs = get_one_hot_vector(y_valid)

    if normalize == True:
        # x = x/255
        x = normalize_data(x)
        valid_inputs = normalize_data(valid_inputs)
        # valid_inputs = valid_inputs / 255
    
    for epoch in range(1, num_epochs+1,1):
        if not log_wandb:
            print("-"*100)
            print(" "*45, f"[Epoch {epoch}]", " "*45)
            print("-"*100)
        train_accuracies, train_losses = [], []
        val_accuracies, val_losses = [], [] 
        
        for i in range(0, x.shape[-1], batch_size):
            x_batch = x[:,i:i+batch_size]
            # x_batch = normalize_data(x_batch) # Batch Normalization
            y_batch = y[:,i:i+batch_size]

            out = model.forward(x_batch)
            predictions = np.argmax(out, axis=0)
            accuracy = (np.sum(y_train[i:i+batch_size]==predictions)/len(predictions))*100
            # print(out[:,0])
            loss = model.loss_fn(y_batch, out)
            train_losses.append(loss)
            model.backward(y_batch, out)
            # print(model.W.keys())
            # print(model.b.keys())
            model.Optimize()
            # for i in range(1, len(model.n), 1):
            #     print(model.W[i].shape, model.b[i].shape)
            # print("Predictions")
            # print(out)
            # print("References")
            # print(y)

            # Validation
            predictions = model.forward(valid_inputs)
            predictions = np.argmax(predictions, axis=0)
            val_accuracy = (np.sum(y_valid==predictions)/len(predictions))*100
            val_loss = model.loss_fn(valid_outputs, predictions)
            
            if not log_wandb:
                print(f"\tBatches {i} to {i+batch_size} |  Loss: {loss:.4} | Train Accuracy: {accuracy:.4}")
                print(f"\tValidation Accuracy: {val_accuracy:.4} | Validation Loss: {val_loss:.4}")
                print()
            train_losses.append(loss)
            val_losses.append(val_loss)
            train_accuracies.append(accuracy)
            val_accuracies.append(val_accuracy)
    

        
        if log_wandb:
            wandb.log({"accuracy": np.mean(train_accuracies), "loss": loss, "val_accuracy":val_accuracy, "val_loss":val_loss, "epoch":epoch}, step=epoch, commit=True)
        else:
            print(f"Epoch {epoch} Done |  Loss: {np.mean(train_losses):.4} | Train Accuracy: {np.mean(train_accuracies):.4} | Val Accuracy: {np.mean(val_accuracies):.4}")

    return model

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-wp', '--wandb_project', type=str, default="myprojectname",
                        help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument('-we', '--wandb_entity', type=str, default="myname",
                        help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument('-d', '--dataset', type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"],
                        help="dataset to be used. Choices: mnist/fashion_mnist")
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help="Number of epochs to train neural network.")
    parser.add_argument('-b', '--batch_size', type=int, default=4,
                        help="Batch size used to train neural network.")
    parser.add_argument('-l', '--loss', type=str, default="cross_entropy", choices=["mean_squared_error", "cross_entropy"],
                        help="Loss to be used. Choices: mean_squared_error/cross_entropy")
    parser.add_argument('-o', '--optimizer', type=str, default="sgd", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam", "adagrad"],
                        help="Optimizer to be used. Choices: sgd/momentum/nag/rmsprop/adam/nadam/adagrad")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1,
                        help="Learning rate used to optimize model parameters")
    parser.add_argument('-m', '--momentum', type=float, default=0.5,
                        help="Momentum used by momentum and nag optimizers.")
    parser.add_argument('-beta', '--beta', type=float, default=0.5,
                        help="Beta used by rmsprop optimizer")
    parser.add_argument('-beta1', '--beta1', type=float, default=0.5,
                        help="Beta1 used by adam and nadam optimizers.")
    parser.add_argument('-beta2', '--beta2', type=float, default=0.5,
                        help="Beta2 used by adam and nadam optimizers.")
    parser.add_argument('-eps', '--epsilon', type=float, default=0.000001,
                        help="Epsilon used by optimizers.")
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0,
                        help="Weight decay used by optimizers.")
    parser.add_argument('-w_i', '--weight_init', type=str, default="random", choices=["random", "xavier", "he"],
                        help="Weight initialization method. Choices: random/Xavier/He")
    parser.add_argument('-nhl', '--num_layers', type=int, default=2,
                        help="Number of hidden layers used in feedforward neural network.")
    parser.add_argument('-sz', '--hidden_size', type=int, default=128,
                        help="Number of hidden neurons in a feedforward layer.")
    parser.add_argument('-a', '--activation', type=str, default="sigmoid", choices=["identity", "sigmoid", "tanh", "relu"],
                        help="Activation function used in the feedforward neural network. Choices: identity/sigmoid/tanh/relu")
    parser.add_argument('-nw', '--no_wandb', default=False, action='store_true',
                        help="Disable WandB if set True")
    
    args = parser.parse_args()
    
    if not args.no_wandb:
        wandb.init(entity=args.wandb_entity, project=args.wandb_project)
        wandb.run.name = f"nhl_{args.num_layers}-sz{args.hidden_size}-bs_{args.batch_size}-optimizer_{args.optimizer}_lr-{args.learning_rate}_activation-{args.activation}_wi-{args.weight_init}_loss-{args.loss}"
    (x_train, y_train), (x_test, y_test) = globals()[args.dataset].load_data()

    x_train, y_train, x_valid, y_valid = get_validation_from_train(x_train, y_train, x=10) # 10 percent is taken from train set for validation
    test_inputs = np.array(x_test).T.reshape(28*28,-1)
    valid_inputs = np.array(x_valid).T.reshape(28*28,-1)


    model = MultiClassClassifier([28*28]+[args.hidden_size]*args.num_layers+[10], 
                                 nonlinear=args.activation, 
                                 weight_init=args.weight_init,
                                 loss_fn=args.loss,
                                 optimizer=args.optimizer,
                                 lr=args.learning_rate,
                                 momentum=args.momentum,
                                 beta=args.beta,
                                 beta1=args.beta1,
                                 beta2=args.beta2,
                                 epsilon=args.epsilon,
                                 weight_decay=args.weight_decay)
    
    train_inputs = np.array(x_train).T.reshape(28*28,-1)
    
    model = train_model(model, train_inputs, y_train, normalize=True, num_epochs=args.epochs, batch_size=args.batch_size, valid_inputs=valid_inputs, y_valid=y_valid, log_wandb=not(args.no_wandb))