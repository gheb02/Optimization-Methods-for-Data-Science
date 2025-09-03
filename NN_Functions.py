import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from typing import List
from scipy.stats import truncnorm
from sklearn.model_selection import train_test_split, KFold
import time


def mape(y_real, y_pred):
    """
    y_real: true target values
    y_pred: predicted values
    """
    return np.mean(np.abs((y_real - y_pred) / (y_real + 1e-8))) * 100
    
def dloss_dypred(y_real, y_pred):
    """
    y_real: true target values
    y_pred: predicted values
    """
    return (y_pred-y_real)/y_real.shape[0]

def mae(y_real, y_pred):
    """
    y_real: true target values
    y_pred: predicted values
    """
    return np.mean(np.abs(y_pred-y_real))


class NeuralNetwork:
    def __init__(self, neurons: List, act='tanh', init=None, alpha=0.5):
        """
        neurons: list with the number of neurons in each layer.
                      e.g. [512, 16, 8, 1]
        activation: string, 'tanh' or 'sigmoid'
        init: string, 'glorot_unif', 'glorot_normal', 'he_unif', 'he_normal' or None
        alpha: float
        """
        self.neurons = neurons
        self.layers = len(neurons) - 1  # number of layers (excluding input)
        self.act = act
        self.alpha = alpha

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        for i in range(self.layers):
            in_dim = neurons[i]
            out_dim = neurons[i + 1]

            if init is not None and init not in ["glorot_unif", "glorot_normal", "he_unif", "he_normal"]:
                raise ValueError('Select a initialization method between: None, "glorot_unif", "glorot_normal", "he_unif" and "he_normal".')
            if init is None:
                self.init = "Random Initialization"
                W = np.random.randn(in_dim, out_dim) * 0.01
            elif init == "glorot_unif":
                self.init = "Glorot Uniform"
                unif_params = 6/np.sqrt(in_dim+out_dim)
                W = np.random.uniform(-unif_params, unif_params, size = (in_dim, out_dim))
            elif init == "glorot_normal":
                self.init = "Glorot Normal"
                std = 2/np.sqrt(in_dim+out_dim)
                l, u = -2, 2
                W = truncnorm.rvs(l, u, loc=0, scale=std, size=(in_dim, out_dim))
            elif init == "he_unif":
                self.init = "He Uniform"
                unif_params = 2/np.sqrt(in_dim)
                W = np.random.uniform(-unif_params, unif_params, size=(in_dim, out_dim))
            elif init == "he_normal":
                self.init = "He Normal"
                std = 2/np.sqrt(in_dim)
                l, u = -2, 2
                W = truncnorm.rvs(l, u, loc=0, scale=std, size=(in_dim, out_dim))
            b = np.zeros((1, out_dim))
            self.weights.append(W)
            self.biases.append(b)

    def activation(self, x):
        """
        x: float
        """
        if self.act == 'tanh':
            return np.tanh(x)
        elif self.act == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError("Unsupported activation function")
        
    def derivate_activation(self, a):
        if self.act == 'tanh':
            return 1 - a**2
        elif self.act == 'sigmoid':
            return a * (1 - a)

    def forward(self, X):
        """
        Performs a forward pass through the network.
        X: input data of shape (n_samples, n_features)
        Returns:
            Output prediction of shape (n_samples, 1)
        """
        a = X
        zs = []
        activations = [X]  # Store input as the first activation
        
        for i in range(self.layers - 1):  # Hidden layers
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.activation(z)
            zs.append(z)
            activations.append(a)

        # Output layer (no activation)
        output = np.dot(a, self.weights[-1]) + self.biases[-1]
        zs.append(output)
        
        return output, activations, zs

    def get_params_vector(self):
        """Returns all weights and biases flattened into a single vector."""
        params = []
        for W, b in zip(self.weights, self.biases):
            params.append(W.flatten())
            params.append(b.flatten())
        return np.concatenate(params)
    

    def set_params_vector(self, flat_params):
        """Set the weights and biases from a flat parameter vector."""
        idx = 0
        self.weights = []
        self.biases = []
        for i in range(self.layers):
            in_dim = self.neurons[i]
            out_dim = self.neurons[i + 1]
            w_size = in_dim * out_dim
            b_size = out_dim

            W = flat_params[idx:idx + w_size].reshape((in_dim, out_dim))
            idx += w_size
            b = flat_params[idx:idx + b_size].reshape(1, out_dim)
            idx += b_size

            self.weights.append(W)
            self.biases.append(b)

    def mse_loss(self, y_real, y_pred, test=False):
        """
        y_real: true target values
        y_pred: predicted values
        test: test MSE doesn't have regularization 
        """
        weights = self.weights

        loss = np.mean((y_real - y_pred)**2)/2
        if test:
            return loss
        else:
            reg = sum([np.sum(w**2) for w in weights])
            return loss + self.alpha * reg
    
    
    def backward(self, X, y):
        """
        Performs backpropagation

        """
        y_pred, activations, zs = self.forward(X)
        deltas = [None] * self.layers
        grads_W, grads_b = [], []

        # delta output layer
        dL_dy = dloss_dypred(y, y_pred)  # shape (batch, out_dim)
        deltas[-1] = dL_dy  # ultimo layer: no activation

        # hidden layers backwards
        for layer in reversed(range(self.layers - 1)):
            a = activations[layer + 1]
            da_dz = self.derivate_activation(a)
            deltas[layer] = (deltas[layer+1] @ self.weights[layer+1].T) * da_dz

        # grad W, b
        for layer in range(self.layers):
            a_prev = activations[layer]
            grad_W = (a_prev.T @ deltas[layer])+2*self.alpha*self.weights[layer]
            grads_W.append(grad_W)
            grads_b.append(np.sum(deltas[layer], axis=0, keepdims=True))

        return grads_W, grads_b
    
    def update(self, grads_W, grads_b, lr=1e-3, method="vanilla", beta1=0.9, beta2=0.999, eps=1e-8, t=1):
        """
        Update weights and biases using either SGD or Adam.
        
        grads_W: list of gradients for weights
        grads_b: list of gradients for biases
        lr: learning rate
        method: "vanilla" or "adam"
        beta1, beta2: Adam hyperparameters
        eps: numerical stability constant
        t: current timestep (needed for Adam bias correction)
        """

        if not hasattr(self, "m_weights"):
            # Initialize Adam moment estimates if not already done
            self.m_weights = [np.zeros_like(W) for W in self.weights]
            self.v_weights = [np.zeros_like(W) for W in self.weights]
            self.m_biases = [np.zeros_like(b) for b in self.biases]
            self.v_biases = [np.zeros_like(b) for b in self.biases]

        for i in range(self.layers):
            if method == "vanilla":
                # Standard gradient descent
                self.weights[i] -= lr * grads_W[i]
                self.biases[i]  -= lr * grads_b[i]

            elif method == "adam":
                # Adam optimizer update
                self.m_weights[i] = beta1 * self.m_weights[i] + (1 - beta1) * grads_W[i]
                self.v_weights[i] = beta2 * self.v_weights[i] + (1 - beta2) * (grads_W[i]**2)

                self.m_biases[i] = beta1 * self.m_biases[i] + (1 - beta1) * grads_b[i]
                self.v_biases[i] = beta2 * self.v_biases[i] + (1 - beta2) * (grads_b[i]**2)

                # Bias correction
                m_hat_W = self.m_weights[i] / (1 - beta1**t)
                v_hat_W = self.v_weights[i] / (1 - beta2**t)
                m_hat_b = self.m_biases[i] / (1 - beta1**t)
                v_hat_b = self.v_biases[i] / (1 - beta2**t)

                # Update parameters
                self.weights[i] -= lr * m_hat_W / (np.sqrt(v_hat_W) + eps)
                self.biases[i]  -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

            else:
                raise ValueError("Unsupported update method. Choose 'sgd' or 'adam'.")
    


    def train(self, X, y, lr, epochs=200, method="Batch", batch_size=None, 
          X_val=None, y_val=None, optimizer="vanilla", beta1=0.9, beta2=0.999, eps=1e-8,
          early_stopping=None, verbose=True):

        if method not in ["Batch", "Mini Batch", "SGD"]:
            raise ValueError('Select a method between: "Batch", "Mini Batch", and "SGD".')

        if optimizer not in ["vanilla", "adam"]:
            raise ValueError('Select optimizer between: "vanilla" and "adam".')
        

        if method == "Mini Batch" and batch_size is None:
            raise ValueError("You must specify a batch_size for mini-batch training.")
        
        if verbose:
            print('The model is initialized with the following hyperparameters:\n')
            print(f'    - Number of hidden layers: {self.layers-1}')
            print(f'    - Number of neurons in each hidden layer: {self.neurons[1:-1]}')
            print(f'    - Activation function: {self.act}')
            print(f'    - Weights initializated using: {self.init}')
            print(f'    - Regularization Term: {self.alpha}')
            print(f'    - Learning Rate: {lr}')
            print(f'    - Optimization Method: {method}')
            print(f'    - Optimizer: {optimizer}')
            print(f'    - Maximum Number of epochs: {epochs}')
            if batch_size:
                print(f'    - Batch Size: {batch_size}')
            if early_stopping:
                print(f'    - Early Stopping Patience: {early_stopping}')
            

        t = 1  # Adam time step

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            if verbose:
                print(f"\nEpoch {epoch + 1}/{epochs}")

            if method == "Batch":
                X_batch, y_batch = X, y

                y_pred, _, _ = self.forward(X_batch)
                loss = self.mse_loss(y_batch, y_pred)

                grads_W, grads_b = self.backward(X_batch, y_batch)
                self.update(grads_W, grads_b, lr, method=optimizer, beta1=beta1, beta2=beta2, eps=eps, t=t)
                if verbose:
                    print(f"[Batch] Loss: {loss:.6f}")
                t += 1

            elif method == "SGD":
                indices = np.random.permutation(len(X))
                X_shuffled, y_shuffled = X[indices], y[indices]

                total_loss = 0
                for i in range(len(X_shuffled)):
                    xi = X_shuffled[i].reshape(1, -1)
                    yi = y_shuffled[i].reshape(1, -1)

                    y_pred, _, _ = self.forward(xi)
                    loss = self.mse_loss(yi, y_pred)

                    grads_W, grads_b = self.backward(xi, yi)
                    self.update(grads_W, grads_b, lr, method=optimizer, beta1=beta1, beta2=beta2, eps=eps, t=t)

                    total_loss += loss
                    t += 1

                avg_loss = total_loss / len(X_shuffled)
                if verbose:
                    print(f"[SGD] Average Loss: {avg_loss:.6f}")

            elif method == "Mini Batch":
                indices = np.random.permutation(len(X))
                X_shuffled, y_shuffled = X[indices], y[indices]

                total_loss = 0
                num_batches = 0

                for i in range(0, len(X_shuffled), batch_size):
                    X_batch = X_shuffled[i:i+batch_size]
                    y_batch = y_shuffled[i:i+batch_size]

                    y_pred, _, _ = self.forward(X_batch)
                    loss = self.mse_loss(y_batch, y_pred)

                    grads_W, grads_b = self.backward(X_batch, y_batch)
                    self.update(grads_W, grads_b, lr, method=optimizer, beta1=beta1, beta2=beta2, eps=eps, t=t)

                    total_loss += loss
                    num_batches += 1
                    t += 1

                avg_loss = total_loss / num_batches
                if verbose:
                    print(f"[Mini-batch] Average Loss: {avg_loss:.6f}")

            # --- Validation & Early Stopping ---
            if X_val is not None and y_val is not None:
                y_pred_val, _, _ = self.forward(X_val)
                val_loss = self.mse_loss(y_val, y_pred_val)
                val_mape = mape(y_val, y_pred_val)
                val_mae = mae(y_val, y_pred_val)
                if verbose:
                    print(f"Validation Loss: {val_loss:.6f} | Validation MAPE: {val_mape:.2f}% | Validation MAE: {val_mae:.2f} years")

                if early_stopping is not None:
                    if val_loss < best_val_loss - 1e-6:  # improvement
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:  # no improvement
                        patience_counter += 1
                        if patience_counter >= early_stopping:
                            print(f"\nEarly stopping triggered at epoch {epoch+1}!")
                            break
                print(f"\nTraining completed!\nFinal training loss: {avg_loss:.6f}\nFinal validation loss: {val_loss:.6f} \nFinal validation MAPE: {val_mape:.2f}% \nFinal validation MAE: {val_mae:.2f}")
            else:
                print(f"\nTraining completed!\nFinal training loss: {avg_loss:.6f}")


def test_model(nn, X_test, y_test):
    """
    Evaluate a trained NeuralNetwork instance on test data.

    Args:
        nn: trained NeuralNetwork instance
        X_test, y_test: test data
        alpha: regularization weight (set to 0 for pure performance)
        verbose: whether to print the result

    Returns:
        test_loss: MSE (with optional reg)
        test_mape: Mean Absolute Percentage Error
    """
    y_pred, _, _ = nn.forward(X_test)
    
    test_loss = nn.mse_loss(y_test, y_pred, test=True)
    test_mape = mape(y_test, y_pred)
    test_mae = mae(y_test, y_pred)
    
    print(f"Test Loss: {test_loss:.6f} | Test MAPE: {test_mape:.2f}% | Test MAE: {test_mae:.2f}")
    
    return test_loss, test_mape, test_mae


def k_fold_cv(X_train, y_train, X_val, y_val, param_grid, k=5, epochs=200, early_stopping=10):
    """
    Performs k-fold cross-validation on the training set, then evaluates on a fixed validation set.

    X_train, y_train: training set (used for k-fold CV)
    X_val, y_val: held-out validation set (fixed)
    param_grid: dict with hyperparameters to search over
    k: number of folds
    epochs: training epochs
    early_stopping: patience for early stopping

    Returns:
        results: list of dicts with hyperparams + avg validation loss
        best_params: dict with best hyperparameters
    """

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    results = []

    # Generate all possible hyperparameter combinations
    from itertools import product
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    print(f'Trying {len(param_combinations)} combination of parameters')
    comb = 0
    start = time.time()
    for params in param_combinations:
        print(f"Combination {comb+1}/{len(param_combinations)}")
        comb += 1
        print(f"\nEvaluating params: {params}")
        fold_losses = []

        neurons = params.get("neurons", [X_train.shape[1], 32, 16, 1])

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            print(f" Fold {fold+1}/{k}")

            X_tr, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_fold_val = y_train[train_idx], y_train[val_idx]

            # Build a fresh network for each fold
            model = NeuralNetwork(
                neurons,
                act=params.get("activation", "tanh"),
                init=params.get("init", None),
                alpha=params.get("alpha", 0.0)
            )

            model.train(
                X_tr, y_tr,
                epochs=epochs,
                lr=params["lr"],
                method="Mini Batch" if "batch_size" in params else "Batch",
                batch_size=params.get("batch_size", None),
                X_val=X_fold_val, y_val=y_fold_val,
                optimizer=params["optimizer"],
                early_stopping=early_stopping,
                verbose=False
            )

            y_pred_val, _, _ = model.forward(X_fold_val)
            val_loss = model.mse_loss(y_fold_val, y_pred_val)
            fold_losses.append(val_loss)

        # average loss across folds
        avg_cv_loss = np.mean(fold_losses)
        end = time.time()

        total_time = end-start
        print(f"\nTotal Validation Time: {total_time:.2f} seconds | Average Validation Time: {(total_time/len(param_combinations)):.2f}")

        # retrain with full training set and evaluate on fixed validation set
        final_model = NeuralNetwork(
            neurons=params.get('neurons'),
            act=params.get("activation", "tanh"),
            init=params.get("init", None),
            alpha=params.get("alpha", 0.0)
        )
        
        final_model.train(
            X_train, y_train,
            epochs=epochs,
            lr=params["lr"],
            method="Mini Batch" if "batch_size" in params else "Batch",
            batch_size=params.get("batch_size", None),
            X_val=X_val, y_val=y_val,
            optimizer=params["optimizer"],
            early_stopping=early_stopping,
            verbose=False
        )

        y_pred_val, _, _ = final_model.forward(X_val)
        fixed_val_loss = final_model.mse_loss(y_val, y_pred_val)

        print(f" Avg CV Loss: {avg_cv_loss:.6f} | Fixed Validation Loss: {fixed_val_loss:.6f}")

        results.append({**params, "cv_loss": avg_cv_loss, "val_loss": fixed_val_loss})

    # Choose best params by fixed validation loss
    best_params = min(results, key=lambda x: x["val_loss"])
    print("\nBest Hyperparameters:", best_params)

    print(f"\nBest Validation Loss: {best_params['val_loss']:.4f}")
    return results, best_params
