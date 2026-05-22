import math
import random


def sigmoid(z):
  return 1.0 / (1.0 + math.exp(-z))


class Neuron:
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias

  def feedforward(self, inputs):
    z = self.bias
    for i in range(len(inputs)):
      z += inputs[i] * self.weights[i]
    return sigmoid(z)


class Layer:
  def __init__(self, neurons):
    self.neurons = neurons

  def feedforward(self, inputs):
    outputs = []
    for neuron in self.neurons:
      outputs.append(neuron.feedforward(inputs))
    return outputs

  def feedforward_thresholded(self, inputs, threshold):
    outputs = []
    raw_outputs = []
    for neuron in self.neurons:
      raw_output = neuron.feedforward(inputs)
      raw_outputs.append(raw_output)
      outputs.append(1.0 if raw_output >= threshold else 0.0)
    return outputs, raw_outputs


class NeuralNetwork:
  def __init__(self, layer_sizes):
    self.layer_sizes = layer_sizes
    self.layers = []

  @staticmethod
  def from_architecture_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
      lines = [line.strip() for line in f.readlines() if line.strip()]

    number_of_layers = int(lines[0])
    layer_sizes = []

    for i in range(1, number_of_layers + 1):
      layer_sizes.append(int(lines[i]))

    return NeuralNetwork(layer_sizes)

  def initialize_random(self):
    self.layers = []

    for layer_index in range(1, len(self.layer_sizes)):
      previous_layer_size = self.layer_sizes[layer_index - 1]
      current_layer_size = self.layer_sizes[layer_index]
      neurons = []

      for _ in range(current_layer_size):
        weights = []
        for _ in range(previous_layer_size):
          weights.append(random.random())
        bias = random.random()
        neurons.append(Neuron(weights, bias))

      self.layers.append(Layer(neurons))

  def initialize_from_weight_file(self, file_path):
    with open(file_path, "r", encoding="utf-8") as f:
      raw_lines = f.readlines()

    lines = []
    for line in raw_lines:
      line = line.strip()
      if line and not line.startswith("#"):
        lines.append(line)

    self.layers = []
    index = 0

    for layer_index in range(1, len(self.layer_sizes)):
      expected_neurons = self.layer_sizes[layer_index]
      expected_inputs = self.layer_sizes[layer_index - 1]

      if lines[index] != f"layer {layer_index}":
        raise ValueError(f"Expected layer {layer_index}")
      index += 1

      neurons = []
      for neuron_index in range(1, expected_neurons + 1):
        if lines[index] != f"neuron {neuron_index}":
          raise ValueError(f"Expected neuron {neuron_index} in layer {layer_index}")
        index += 1

        bias_parts = lines[index].split()
        if len(bias_parts) != 2 or bias_parts[0] != "bias":
          raise ValueError("Invalid bias line")
        bias = float(bias_parts[1])
        index += 1

        weight_parts = lines[index].split()
        if len(weight_parts) != expected_inputs + 1 or weight_parts[0] != "weights":
          raise ValueError("Invalid weights line")
        weights = []
        for value in weight_parts[1:]:
          weights.append(float(value))
        index += 1

        neurons.append(Neuron(weights, bias))

      self.layers.append(Layer(neurons))

  def feedforward(self, inputs):
    activations = inputs
    all_activations = [inputs]

    for layer in self.layers:
      activations = layer.feedforward(activations)
      all_activations.append(activations)

    return activations, all_activations

  def feedforward_thresholded(self, inputs, threshold=0.5):
    activations = inputs
    all_activations = [inputs]
    all_raw_activations = [inputs]

    for layer in self.layers:
      activations, raw_activations = layer.feedforward_thresholded(activations, threshold)
      all_activations.append(activations)
      all_raw_activations.append(raw_activations)

    return activations, all_activations, all_raw_activations

  def print_summary(self):
    print("Network architecture")
    for i in range(len(self.layer_sizes)):
      print(f"Layer {i}: {self.layer_sizes[i]} neuron(s)")

    for layer_index in range(len(self.layers)):
      print(f"\nLayer {layer_index + 1}")
      for neuron_index in range(len(self.layers[layer_index].neurons)):
        neuron = self.layers[layer_index].neurons[neuron_index]
        print(
          f"  Neuron {neuron_index + 1}: "
          f"weights = {neuron.weights}, bias = {neuron.bias}"
        )


def run_xor_experiment():
  network = NeuralNetwork.from_architecture_file("network_architecture.txt")
  network.initialize_from_weight_file("xor_weights.txt")
  network.print_summary()

  test_inputs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
  ]

  print("\nXOR feedforward result")
  print("x1 x2 hidden_1 hidden_2 y_hat decision expected")

  expected_outputs = [0, 1, 1, 0]
  for i in range(len(test_inputs)):
    output, activations = network.feedforward(test_inputs[i])
    y_hat = output[0]
    decision = 1 if y_hat >= 0.5 else 0
    hidden_1 = activations[1][0]
    hidden_2 = activations[1][1]
    print(
      f"{test_inputs[i][0]:.0f}  {test_inputs[i][1]:.0f}  "
      f"{hidden_1:.4f}   {hidden_2:.4f}   "
      f"{y_hat:.4f}   {decision}        {expected_outputs[i]}"
    )

  print("\nXOR gate-style feedforward result with threshold = 0.5")
  print("x1 x2 h1 h2 y decision expected")

  for i in range(len(test_inputs)):
    output, activations, raw_activations = network.feedforward_thresholded(test_inputs[i])
    decision = int(output[0])
    print(
      f"{test_inputs[i][0]:.0f}  {test_inputs[i][1]:.0f}  "
      f"{activations[1][0]:.0f}  {activations[1][1]:.0f}  "
      f"{output[0]:.0f}  {decision}        {expected_outputs[i]}"
    )


if __name__ == "__main__":
  run_xor_experiment()
