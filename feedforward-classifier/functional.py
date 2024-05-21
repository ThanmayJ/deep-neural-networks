import numpy as np



def linear(input : np.ndarray, weight: np.ndarray , bias : np.ndarray ) -> np.ndarray:
    if bias is None:
        bias = 0
    # print(weight.shape, input.shape, bias.shape)
    return np.matmul(weight, input) + bias

# ----------------------------------------------------
#                   Weight Initialization
# ----------------------------------------------------


def he_init(fan_in, fan_out):
    W = np.random.randn(fan_out, fan_in) * np.sqrt(2 / (fan_in))
    b = np.random.randn(fan_out, 1)
    return W,b

def xavier_init(fan_in, fan_out):
    W = np.random.randn(fan_out, fan_in) * np.sqrt(2 / (fan_in + fan_out))
    b = np.random.randn(fan_out, 1)
    return W,b

def random_init(fan_in, fan_out):
    W = np.random.randn(fan_out, fan_in)
    b = np.random.randn(fan_out, 1)
    return W,b


# ----------------------------------------------------
#                   Activation Functions
# ----------------------------------------------------


def sigmoid(input : np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-input))

def relu(input : np.ndarray) -> np.ndarray:
    return np.maximum(0, input)

def tanh(input: np.ndarray) -> np.ndarray:
    return np.tanh(input)

def identity(input: np.ndarray) -> np.ndarray:
    return input

def softmax(input: np.ndarray) -> np.ndarray:
    return np.exp(input-np.max(input))/np.sum(np.exp(input-np.max(input)),axis=0, keepdims=True)

# def softmax(input: np.ndarray) -> np.ndarray:
#     return np.exp(input-np.max(input, axis=0, keepdims=True))/np.sum(np.exp(input),axis=0, keepdims=True)







# ----------------------------------------------------
#           Activation Function Derivatives
# ----------------------------------------------------

def sigmoid_prime(input : np.ndarray) -> np.ndarray:
    return sigmoid(input)*(1 - sigmoid(input))

def relu_prime(input : np.ndarray) -> np.ndarray:
    input[input <= 0] = 0
    input[input > 0] = 1
    return input

def tanh_prime(input: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(input)**2

def identity_prime(input: np.ndarray) -> np.ndarray:
    return np.ones_like(input)

def softmax_prime(input: np.ndarray) -> np.ndarray:
    pred = softmax(input)
    jacobian = -np.matmul(pred,pred.T)
    np.fill_diagonal(jacobian, pred*(1-pred))
    return jacobian







# ----------------------------------------------------
#                       Optimizers
# ----------------------------------------------------

class optimizer_sgd:
    def __init__(self, n):
        self.n = n
    
    def optimize(self, W, b, dLdW, dLdb, lr, momentum, beta, beta1, beta2, epsilon, weight_decay):
        for i in range(1, len(self.n), 1):
            W[i] = W[i] - lr * dLdW[i]
            b[i] = b[i] - lr * dLdb[i]
        return W, b
    
class optimizer_momentum:
    def __init__(self, n):
        self.n = n
        self.W_velocity, self.b_velocity = {}, {}
        for i in range(1, len(self.n), 1):
            self.W_velocity[i], self.b_velocity[i] = 0,0
    
    def optimize(self, W, b, dLdW, dLdb, lr, momentum, beta, beta1, beta2, epsilon, weight_decay):
        for i in range(1, len(self.n), 1):
            self.W_velocity[i] = momentum * self.W_velocity[i] + dLdW[i]
            self.b_velocity[i] = momentum * self.b_velocity[i] + dLdb[i]

            W[i] = W[i] - lr * self.W_velocity[i]
            b[i] = b[i] - lr * self.b_velocity[i]
        return W, b


class optimizer_nag:
    def __init__(self, n):
        self.n = n
        self.W_velocity, self.b_velocity = {}, {}
        for i in range(1, len(self.n), 1):
            self.W_velocity[i], self.b_velocity[i] = 0,0
    
    def optimize(self, W, b, dLdW_lookahead, dLdb_lookahead, lr, momentum, beta, beta1, beta2, epsilon, weight_decay):
        for i in range(1, len(self.n), 1):
            self.W_velocity[i] = momentum * self.W_velocity[i] + dLdW_lookahead[i]
            self.b_velocity[i] = momentum * self.b_velocity[i] + dLdb_lookahead[i]

            W[i] = W[i] - lr * self.W_velocity[i]
            b[i] = b[i] - lr * self.b_velocity[i]
        return W, b


class optimizer_rmsprop:
    def __init__(self, n):
        self.n = n
        self.W_rmsprop, self.b_rmsprop = {}, {}
        for i in range(1, len(self.n), 1):
            self.W_rmsprop[i], self.b_rmsprop[i] = 0, 0
    
    def optimize(self, W, b, dLdW, dLdb, lr, momentum, beta, beta1, beta2, epsilon, weight_decay):
        for i in range(1, len(self.n), 1):
            self.W_rmsprop[i] = beta * self.W_rmsprop[i] + (1 - beta) * np.square(dLdW[i])
            self.b_rmsprop[i] = beta * self.b_rmsprop[i] + (1 - beta) * np.square(dLdb[i])
            
            W[i] = W[i] - lr * (dLdW[i] / (np.sqrt(self.W_rmsprop[i]) + epsilon))
            b[i] = b[i] - lr * (dLdb[i] / (np.sqrt(self.b_rmsprop[i]) + epsilon))
        return W, b


class optimizer_adam:
    def __init__(self, n):
        self.n = n
        self.t = 0 # time step
        self.mW, self.mb = {}, {}
        self.vW, self.vb = {}, {}
        for i in range(1, len(self.n), 1):
            self.mW[i], self.mb[i] = 0, 0
            self.vW[i], self.vb[i] = 0, 0
    
    def optimize(self, W, b, dLdW, dLdb, lr, momentum, beta, beta1, beta2, epsilon, weight_decay):
        self.t += 1 
        for i in range(1, len(self.n), 1):
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * dLdW[i]
            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * dLdb[i]
            
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * np.square(dLdW[i])
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * np.square(dLdb[i])
            
            mW_hat = self.mW[i] / (1 - beta1 ** self.t)
            mb_hat = self.mb[i] / (1 - beta1 ** self.t)
            vW_hat = self.vW[i] / (1 - beta2 ** self.t)
            vb_hat = self.vb[i] / (1 - beta2 ** self.t)
            
            W[i] = W[i] - lr * (mW_hat / (np.sqrt(vW_hat) + epsilon))
            b[i] = b[i] - lr * (mb_hat / (np.sqrt(vb_hat) + epsilon))
        return W, b

class optimizer_nadam:
    def __init__(self, n):
        self.n = n
        self.t = 0  #   time step
        self.mW, self.mb = {}, {}
        self.vW, self.vb = {}, {}
        for i in range(1, len(self.n), 1):
            self.mW[i], self.mb[i] = 0, 0
            self.vW[i], self.vb[i] = 0, 0
    
    def optimize(self, W, b, dLdW, dLdb, lr, momentum, beta, beta1, beta2, epsilon, weight_decay):
        self.t += 1 
        for i in range(1, len(self.n), 1):
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * dLdW[i]
            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * dLdb[i]
            
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * np.square(dLdW[i])
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * np.square(dLdb[i])
            
            mW_hat = self.mW[i] / (1 - beta1 ** self.t)
            mb_hat = self.mb[i] / (1 - beta1 ** self.t)
            vW_hat = self.vW[i] / (1 - beta2 ** self.t)
            vb_hat = self.vb[i] / (1 - beta2 ** self.t)
            
            mW_nesterov = beta1 * mW_hat + ((1 - beta1) * dLdW[i]) / (1 - beta1 ** self.t)
            mb_nesterov = beta1 * mb_hat + ((1 - beta1) * dLdb[i]) / (1 - beta1 ** self.t)
            
            W[i] = W[i] - lr * ((mW_nesterov / (np.sqrt(vW_hat) + epsilon)))
            b[i] = b[i] - lr * ((mb_nesterov / (np.sqrt(vb_hat) + epsilon)))
        return W, b
    
class optimizer_adagrad:
    def __init__(self, n):
        self.n = n
        self.W_velocity, self.b_velocity = {}, {}
        for i in range(1, len(self.n), 1):
            self.W_velocity[i], self.b_velocity[i] = 0, 0
    
    def optimize(self, W, b, dLdW, dLdb, lr, momentum, beta, beta1, beta2, epsilon, weight_decay):
        for i in range(1, len(self.n), 1):
            self.W_velocity[i] += np.square(dLdW[i])
            self.b_velocity[i] += np.square(dLdb[i])
            
            W[i] = W[i] - lr * (dLdW[i] / (np.sqrt(self.W_velocity[i]) + epsilon))
            b[i] = b[i] - lr * (dLdb[i] / (np.sqrt(self.b_velocity[i]) + epsilon))
        return W, b








# ----------------------------------------------------
#                      Loss Functions
# ----------------------------------------------------

    
def cross_entropy_loss(y : np.ndarray, pred : np.ndarray) -> float:
    m = y.shape[1] # num. of training examples
    # ce_loss = -np.sum(y*np.log(pred), axis=0) # Cross entropy loss for m examples
    # ce_loss = np.mean(ce_loss) # Cross entropy loss as averaged across all the m examples
    ce_loss = -np.mean(y*np.log(pred+1e-10))
    return ce_loss

def mean_squared_error_loss(y: np.ndarray, pred: np.ndarray) -> float:
    m = y.shape[1] # num. of training examples
    k = y.shape[0] # num of classes
    mse_loss = np.mean(np.square(y-pred))
    return mse_loss

