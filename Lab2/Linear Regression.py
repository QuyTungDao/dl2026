import matplotlib.pyplot as plt

def read_data(filename):
  X = []
  Y = []
  with open(filename, 'r') as f:
    for line in f:
      line = line.strip()
      if line:
        parts = line.split(',')
        if len(parts) >= 2:
          X.append(float(parts[0]))
          Y.append(float(parts[1]))
  return X, Y


def gradient_descent_linear_regression(X, Y, lr, threshold):
  w = 0.0
  b = 0.0
  N = len(X)

  step = 0

  while True:
    dw = 0.0
    db = 0.0
    loss = 0.0

    for i in range(N):
      y_pred = w * X[i] + b
      error = y_pred - Y[i]

      loss += error ** 2

      dw += (2 / N) * error * X[i]
      db += (2 / N) * error

    loss = loss / N

    if abs(loss) < threshold:
      break

    w = w - lr * dw
    b = b - lr * db

    print(f"Step {step}: w = {w:.4f}, b = {b:.4f}, Loss = {loss:.4f}")

    step += 1

  return w, b


if __name__ == "__main__":
  X, Y = read_data('lr.csv')

  learning_rate = 0.0001
  threshold = 19

  w, b = gradient_descent_linear_regression(X, Y, learning_rate, threshold)
  print(f"w = {w:.4f}, b = {b:.4f}")

  x_line = [0, max(X) + 10]
  y_line = [w * x + b for x in x_line]

  plt.figure(figsize=(10, 6))

  plt.scatter(X, Y, color='blue', label='Actual Data', zorder=2)

  plt.plot(x_line, y_line, color='red', label=f'y = {w:.2f}x + {b:.2f}', zorder=1)

  plt.title("Linear Regression")
  plt.xlabel("X")
  plt.ylabel("Y")
  plt.legend()
  plt.grid(True)
  plt.show()
