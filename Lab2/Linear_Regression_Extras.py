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

def gradient_descent_extras(X, Y, lr, threshold):
    w0 = 0.0
    w1 = 1.0
    N = len(X)

    step = 0
    while True:
        J = 0.0
        dw0_sum = 0.0
        dw1_sum = 0.0

        for i in range(N):
            y_hat = w1 * X[i] + w0

            L_i = 0.5 * (y_hat - Y[i]) ** 2
            J += L_i

            dL_dw0 = y_hat - Y[i]
            dL_dw1 = X[i] * (y_hat - Y[i])

            dw0_sum += dL_dw0
            dw1_sum += dL_dw1

        J = J / N

        if J < threshold:
            break

        w0 = w0 - lr * (dw0_sum / N)
        w1 = w1 - lr * (dw1_sum / N)

        if step % 1000 == 0:
            print(f"Step {step}: w0 = {w0:.4f}, w1 = {w1:.4f}, Loss J = {J:.4f}")

        step += 1

    return w0, w1

if __name__ == "__main__":
    X, Y = read_data('lr.csv')

    learning_rate = 0.0001

    threshold = 9.5

    w0, w1 = gradient_descent_extras(X, Y, learning_rate, threshold)
    print(f"\nw0 (bias) = {w0:.4f}, w1 (slope) = {w1:.4f}")

    x_line = [0, max(X) + 10]
    y_line = [w1 * x + w0 for x in x_line]

    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, color='blue', label='Actual Data', zorder=2)
    plt.plot(x_line, y_line, color='red', label=f'y = {w1:.2f}x + {w0:.2f}', zorder=1)
    plt.title("Linear Regression (Extras)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()
