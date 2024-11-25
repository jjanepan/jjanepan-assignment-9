import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr
        self.activation_fn = activation
        self.activations = {
            'tanh': np.tanh,
            'relu': lambda x: np.maximum(0, x),
            'sigmoid': lambda x: 1 / (1 + np.exp(-x))
        }
        self.activation_derivatives = {
            'tanh': lambda x: 1 - np.tanh(x) ** 2,
            'relu': lambda x: (x > 0).astype(float),
            'sigmoid': lambda x: self.activations['sigmoid'](x) * (1 - self.activations['sigmoid'](x))
        }
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.activations[self.activation_fn](self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.out = self.activations['sigmoid'](self.Z2)
        return self.out

    def backward(self, X, y):
        m = X.shape[0]
        dZ2 = self.out - y
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.activation_derivatives[self.activation_fn](self.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

def generate_data(n_samples=200):
    np.random.seed(42)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0]**2 + X[:, 1]**2 > 1).astype(int).reshape(-1, 1)  # Circular boundary
    return X, y * 2 - 1  # Convert to -1, 1 labels

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Train for multiple steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    # Visualize hidden layer
    hidden_features = mlp.A1
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title("Hidden Layer Features")
    ax_hidden.set_xlim(-1, 1)
    ax_hidden.set_ylim(-1, 1)

    # Decision boundary
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100),
        np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 100)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid).reshape(xx.shape)
    ax_input.contourf(xx, yy, preds, levels=50, cmap="bwr", alpha=0.7)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="bwr", edgecolor="k")
    ax_input.set_title("Input Space Decision Boundary")

    # Gradient visualization
    gradient_magnitude = np.linalg.norm(mlp.W1, axis=0)
    ax_gradient.bar(range(len(gradient_magnitude)), gradient_magnitude, color='blue', alpha=0.6)
    ax_gradient.set_title("Gradient Magnitudes")
    ax_gradient.set_xlabel("Weights")
    ax_gradient.set_ylabel("Magnitude")
    ax_gradient.set_ylim(0, np.max(gradient_magnitude) * 1.2)

def visualize(activation, lr, step_num, hidden_dim=3):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=hidden_dim, output_dim=1, lr=lr, activation=activation)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_hidden, ax_input, ax_gradient = axes

    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), 
                        frames=step_num // 10, repeat=False)
    ani.save(os.path.join(result_dir, f"visualize_{activation}.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    visualize(activation="tanh", lr=0.01, step_num=500, hidden_dim=4)
