def f(x):
  return x * x


def df(x):
  return 2 * x


def GradientDescent(x, lr, thread_hold):
  count = 0
  while abs(f(x) > thread_hold):
    x = x - lr * df(x)
    count += 1
    print(f"{count} {x} {df(x)} {f(x)}")


if __name__ == "__main__":
  x = 10
  lr = 0.9
  thread_hold = 0.01
  GradientDescent(x, lr, thread_hold)
