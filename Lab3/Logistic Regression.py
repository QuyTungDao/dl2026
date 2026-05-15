import math
import matplotlib.pyplot as plt


def load_data(file_path):
  X = []
  Y = []
  with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines[1:]:
      parts = line.strip().split(',')
      if len(parts) == 3:
        exp = float(parts[0].strip())
        sal = float(parts[1].strip())
        loan = int(float(parts[2].strip()))
        X.append([sal, exp])
        Y.append(loan)
  return X, Y


def sigmoid(z):
  z = max(-500, min(500, z))
  return 1.0 / (1.0 + math.exp(-z))


def predict_probability(x, w0, w1, w2):
  z = w1 * x[0] + w2 * x[1] + w0
  return sigmoid(z)


def compute_loss(X, Y, w0, w1, w2):
  N = len(Y)
  total_loss = 0.0
  epsilon = 1e-15
  for i in range(N):
    y_hat = predict_probability(X[i], w0, w1, w2)
    y_hat = max(epsilon, min(1.0 - epsilon, y_hat))
    total_loss += -(Y[i] * math.log(y_hat) + (1.0 - Y[i]) * math.log(1.0 - y_hat))
  return total_loss / N


def train(X, Y, lr, epochs, verbose=True):
  w0, w1, w2 = 0.0, 1.0, 2.0
  N = len(Y)
  loss_history = []

  for epoch in range(epochs):
    dw0, dw1, dw2 = 0.0, 0.0, 0.0

    for i in range(N):
      y_hat = predict_probability(X[i], w0, w1, w2)
      error = y_hat - Y[i]
      dw0 += error
      dw1 += error * X[i][0]
      dw2 += error * X[i][1]

    w0 -= lr * (dw0 / N)
    w1 -= lr * (dw1 / N)
    w2 -= lr * (dw2 / N)

    loss = compute_loss(X, Y, w0, w1, w2)
    loss_history.append(loss)

    if verbose and (epoch % 10000 == 0 or epoch == epochs - 1):
      print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Weights: w0={w0:.3f}, w1={w1:.3f}, w2={w2:.3f}")

  return w0, w1, w2, loss_history


def make_decision(x, w0, w1, w2, threshold=0.2):
  prob = predict_probability(x, w0, w1, w2)
  return 1 if prob >= threshold else 0


if __name__ == "__main__":
  X_train, Y_train = load_data('loan2.csv')

  learning_rate = 0.1
  epochs = 190000
  w0, w1, w2, losses = train(X_train, Y_train, learning_rate, epochs)

  test_profile = [7.0, 0.5]
  t = 0.5

  prob = predict_probability(test_profile, w0, w1, w2)
  decision = make_decision(test_profile, w0, w1, w2, threshold=t)

  print(f"\nPrediction Probability (y_hat): {prob:.4f}")
  print(f"Final Decision: {'LOAN' if decision == 1 else 'REFUSE'}\n")

  X_sal = [x[0] for x in X_train]
  X_exp = [x[1] for x in X_train]

  sal_loan = [X_sal[i] for i in range(len(Y_train)) if Y_train[i] == 1]
  exp_loan = [X_exp[i] for i in range(len(Y_train)) if Y_train[i] == 1]
  sal_refuse = [X_sal[i] for i in range(len(Y_train)) if Y_train[i] == 0]
  exp_refuse = [X_exp[i] for i in range(len(Y_train)) if Y_train[i] == 0]

  plt.figure(figsize=(10, 6))
  plt.scatter(sal_loan, exp_loan, color='red', label='Loan (1)', s=60)
  plt.scatter(sal_refuse, exp_refuse, color='blue', label='Refuse (0)', s=60)

  min_sal, max_sal = min(X_sal) - 0.5, max(X_sal) + 0.5
  sal_line = [min_sal, max_sal]
  exp_line = [(-w1 * min_sal - w0) / w2, (-w1 * max_sal - w0) / w2]

  plt.plot(sal_line, exp_line, color='green', linewidth=3, label='Decision Boundary')
  plt.xlabel('Salary (million)')
  plt.ylabel('Experience (years)')
  plt.title('Data & Decision Boundary')
  plt.legend()
  plt.grid(True, linestyle='--', alpha=0.6)

  y_min, y_max = min(X_exp) - 0.5, max(X_exp) + 0.5
  plt.ylim(y_min, y_max)
  plt.xlim(min_sal, max_sal)

  plt.tight_layout()
  plt.show()
