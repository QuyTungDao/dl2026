import math
import random
import matplotlib.pyplot as plt

class Matrix:
    def __init__(self, rows, cols, data=None):
        self.rows = rows
        self.cols = cols
        if data is None:
            self.data = [[0.0 for _ in range(cols)] for _ in range(rows)]
        else:
            self.data = data
            
    def randomize(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = random.random() * 2 - 1 # -1 to 1
                
    def to_array(self):
        arr = []
        for i in range(self.rows):
            for j in range(self.cols):
                arr.append(self.data[i][j])
        return arr
        
    def add(self, n):
        if isinstance(n, Matrix):
            if self.rows != n.rows or self.cols != n.cols:
                raise ValueError("Matrix dimensions must match for addition")
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] + n.data[i][j]
            return result
        else:
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] + n
            return result
            
    def sub(self, n):
        if isinstance(n, Matrix):
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] - n.data[i][j]
            return result
        else:
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] - n
            return result
            
    def mul(self, n):
        # Element-wise multiplication (Hadamard product) or scalar
        if isinstance(n, Matrix):
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] * n.data[i][j]
            return result
        else:
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] * n
            return result
            
    def dot(self, n):
        if self.cols != n.rows:
            raise ValueError(f"Columns of A ({self.cols}) must match rows of B ({n.rows})")
        result = Matrix(self.rows, n.cols)
        for i in range(result.rows):
            for j in range(result.cols):
                sum_val = 0
                for k in range(self.cols):
                    sum_val += self.data[i][k] * n.data[k][j]
                result.data[i][j] = sum_val
        return result
        
    def transpose(self):
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[j][i] = self.data[i][j]
        return result
        
    def map(self, func):
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                val = self.data[i][j]
                result.data[i][j] = func(val)
        return result

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-x))

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        self.W1 = Matrix(self.input_nodes, self.hidden_nodes)
        self.W1.randomize()
        self.b1 = Matrix(1, self.hidden_nodes)
        self.b1.randomize()
        
        self.W2 = Matrix(self.hidden_nodes, self.output_nodes)
        self.W2.randomize()
        self.b2 = Matrix(1, self.output_nodes)
        self.b2.randomize()
        
        self.learning_rate = 0.1
        
    def feedforward(self, input_array):
        X = Matrix(1, len(input_array))
        for i in range(len(input_array)):
            X.data[0][i] = input_array[i]
            
        Z1 = X.dot(self.W1).add(self.b1)
        A1 = Z1.map(sigmoid)
        
        Z2 = A1.dot(self.W2).add(self.b2)
        A2 = Z2.map(sigmoid)
        
        return A2.to_array()

    def calculate_gradients(self, input_array, target_array):
        X = Matrix(1, len(input_array))
        for i in range(len(input_array)):
            X.data[0][i] = input_array[i]
            
        Y = Matrix(1, len(target_array))
        for i in range(len(target_array)):
            Y.data[0][i] = target_array[i]
            
        # Feedforward
        Z1 = X.dot(self.W1).add(self.b1)
        A1 = Z1.map(sigmoid)
        
        Z2 = A1.dot(self.W2).add(self.b2)
        A2 = Z2.map(sigmoid)
        
        # Loss
        y_val = Y.data[0][0]
        y_hat = A2.data[0][0]
        epsilon = 1e-15
        y_hat_clip = max(epsilon, min(1 - epsilon, y_hat))
        loss = - (y_val * math.log(y_hat_clip) + (1 - y_val) * math.log(1 - y_hat_clip))
        
        # Backpropagation
        dZ2 = A2.sub(Y)
        
        dW2 = A1.transpose().dot(dZ2)
        db2 = dZ2
        
        dA1 = dZ2.dot(self.W2.transpose())
        
        ones = Matrix(A1.rows, A1.cols)
        for i in range(ones.rows):
            for j in range(ones.cols):
                ones.data[i][j] = 1.0
        dZ1 = dA1.mul(A1.mul(ones.sub(A1)))
        
        dW1 = X.transpose().dot(dZ1)
        db1 = dZ1

        return loss, dW1, db1, dW2, db2

    def train_batch(self, inputs, targets):
        N = len(inputs)
        total_loss = 0.0
        total_dW1 = Matrix(self.W1.rows, self.W1.cols)
        total_db1 = Matrix(self.b1.rows, self.b1.cols)
        total_dW2 = Matrix(self.W2.rows, self.W2.cols)
        total_db2 = Matrix(self.b2.rows, self.b2.cols)

        for i in range(N):
            loss, dW1, db1, dW2, db2 = self.calculate_gradients(inputs[i], targets[i])
            total_loss += loss
            total_dW1 = total_dW1.add(dW1)
            total_db1 = total_db1.add(db1)
            total_dW2 = total_dW2.add(dW2)
            total_db2 = total_db2.add(db2)

        scale = 1.0 / N
        self.W2 = self.W2.sub(total_dW2.mul(self.learning_rate * scale))
        self.b2 = self.b2.sub(total_db2.mul(self.learning_rate * scale))
        self.W1 = self.W1.sub(total_dW1.mul(self.learning_rate * scale))
        self.b1 = self.b1.sub(total_db1.mul(self.learning_rate * scale))

        return total_loss / N

    def print_parameters(self):
        print("\nOptimized weights and biases:")
        print(f"W1 = {self.W1.data}")
        print(f"b1 = {self.b1.data}")
        print(f"W2 = {self.W2.data}")
        print(f"b2 = {self.b2.data}")

def load_data(filename):
    inputs = []
    targets = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                inputs.append([float(parts[0]), float(parts[1])])
                targets.append([float(parts[2])])
    return inputs, targets

def train_network(filename, title, epochs=10000, lr=0.1, seed=1):
    inputs, targets = load_data(filename)
    
    if len(inputs) > 0:
        max_f1 = max([x[0] for x in inputs])
        max_f2 = max([x[1] for x in inputs])
        
        if max_f1 > 1.0 or max_f2 > 1.0:
            for i in range(len(inputs)):
                inputs[i][0] /= max_f1
                inputs[i][1] /= max_f2
                
    random.seed(seed)
    nn = NeuralNetwork(2, 2, 1)
    nn.learning_rate = lr
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = nn.train_batch(inputs, targets)
        
        if epoch % max(1, (epochs // 20)) == 0:
            losses.append(epoch_loss)
            print(f"Epoch {epoch:5d}, Loss: {epoch_loss:.4f}")
            
    print(f"Final Loss: {epoch_loss:.4f}")
    nn.print_parameters()
    
    print("\nPredictions after training:")
    for i in range(len(inputs)):
        pred = nn.feedforward(inputs[i])[0]
        decision = 1 if pred >= 0.5 else 0
        print(f"Input: {inputs[i]}, Target: {targets[i][0]}, Pred: {pred:.4f}, Decision: {decision}")
        
    return nn, losses, inputs, targets

def plot_results(nn, losses, inputs, targets, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(losses, color='blue', linewidth=2)
    ax1.set_title(f"Loss over time - {title}")
    ax1.set_xlabel("Epochs (sampled)")
    ax1.set_ylabel("Binary Cross-Entropy Loss")
    ax1.grid(True)
    
    x_min = min(x[0] for x in inputs) - 0.1
    x_max = max(x[0] for x in inputs) + 0.1
    y_min = min(x[1] for x in inputs) - 0.1
    y_max = max(x[1] for x in inputs) + 0.1
    
    xx_pts = 50
    yy_pts = 50
    
    db_x = []
    db_y = []
    db_c = []
    
    dx = (x_max - x_min) / xx_pts
    dy = (y_max - y_min) / yy_pts
    
    for i in range(xx_pts):
        for j in range(yy_pts):
            px = x_min + i * dx
            py = y_min + j * dy
            pred = nn.feedforward([px, py])[0]
            db_x.append(px)
            db_y.append(py)
            db_c.append(pred)
            
    ax2.scatter(db_x, db_y, c=db_c, cmap='coolwarm', alpha=0.3, marker='s')
    
    class_0_x = [inputs[i][0] for i in range(len(inputs)) if targets[i][0] == 0]
    class_0_y = [inputs[i][1] for i in range(len(inputs)) if targets[i][0] == 0]
    
    class_1_x = [inputs[i][0] for i in range(len(inputs)) if targets[i][0] == 1]
    class_1_y = [inputs[i][1] for i in range(len(inputs)) if targets[i][0] == 1]
    
    ax2.scatter(class_0_x, class_0_y, color='blue', edgecolor='k', label='Class 0', s=100)
    ax2.scatter(class_1_x, class_1_y, color='red', edgecolor='k', label='Class 1', s=100)
    
    ax2.set_title(f"Decision Boundary - {title}")
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")
    ax2.legend()
    
    plt.tight_layout()
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, f"{title.replace(' ', '_').lower()}.png"))

if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=== Training XOR ===")
    xor_path = os.path.join(script_dir, 'xor.csv')
    nn_xor, losses_xor, X_xor, Y_xor = train_network(xor_path, 'XOR Dataset', epochs=15000, lr=0.5, seed=1)
    plot_results(nn_xor, losses_xor, X_xor, Y_xor, "XOR Dataset")
    
    print("\n=== Training House Price-Size ===")
    house_path = os.path.join(script_dir, 'house.csv')
    nn_house, losses_house, X_house, Y_house = train_network(house_path, 'House Dataset', epochs=15000, lr=0.1, seed=1)
    plot_results(nn_house, losses_house, X_house, Y_house, "House Dataset")
